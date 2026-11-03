"""四帧原始多模态输入加一枚可学习Query；不使用Reader或预编码历史特征。"""
import torch


def four_frame_inputs(frames, config):
    """按样本重排四帧patch，保留一份完整语言提示；返回Qwen原生多图输入。"""
    if len(frames)!=4:
        raise ValueError('Direct VLM Query requires chronological four-frame inputs')
    batch=frames[-1]['input_ids'].shape[0]
    start,end,image=config.vision_start_token_id,config.vision_end_token_id,config.image_token_id
    spans=[];patches=[];grids=[];prompts=[]
    for frame in frames:
        if frame['input_ids'].shape[0]!=batch or frame['image_grid_thw'].shape!=(batch,3):
            raise ValueError('Each frame must contain exactly one image per sample')
        grid=frame['image_grid_thw'];counts=grid.prod(-1).tolist()
        pixels=frame['pixel_values']
        # 原生collator将等尺寸图像stack为[B,Npatch,Dpatch]；也接受processor的扁平patch表。
        if pixels.ndim==3 and pixels.shape[0]==batch:pixels=pixels.flatten(0,1)
        if sum(counts)!=len(pixels) or pixels.ndim!=2:
            raise ValueError('Image patch/grid mismatch')
        patches.append(pixels.split(counts));grids.append(grid)
        frame_spans=[];frame_prompts=[]
        for row,mask in zip(frame['input_ids'],frame['attention_mask']):
            ids=row[mask.bool()]
            begin=torch.where(ids==start)[0];finish=torch.where(ids==end)[0]
            if len(begin)!=1 or len(finish)!=1 or finish[0]<=begin[0]+1:
                raise ValueError('Expected one complete image placeholder span per frame')
            a,b=int(begin[0]),int(finish[0])+1
            if not (ids[a+1:b-1]==image).all():
                raise ValueError('Only image tokens are supported inside each vision span')
            frame_spans.append(ids[a:b]);frame_prompts.append((ids,a,b))
        spans.append(frame_spans);prompts.append(frame_prompts)
    sequences=[];pixels=[];image_grids=[]
    for sample in range(batch):
        current,a,b=prompts[-1][sample]
        # 与原生4-image chat模板一致：前三帧只插入视觉span，语言仅保留当前提示一次。
        sequences.append(torch.cat([current[:a],*[spans[f][sample] for f in range(4)],current[b:]]))
        for f in range(4):
            pixels.append(patches[f][sample]);image_grids.append(grids[f][sample])
    length=max(map(len,sequences));device=sequences[0].device
    # padding id只供RoPE计算；mask=0位置不参与attention，不修改tokenizer词表。
    ordinary=next(i for i in range(8) if i not in (start,end,image,config.video_token_id))
    ids=torch.full((batch,length),ordinary,dtype=sequences[0].dtype,device=device)
    mask=torch.zeros_like(ids)
    for i,sequence in enumerate(sequences):ids[i,:len(sequence)]=sequence;mask[i,:len(sequence)]=1
    return dict(input_ids=ids,attention_mask=mask,pixel_values=torch.cat(pixels),image_grid_thw=torch.stack(image_grids))


def query_output(vlm, inputs, query, max_length):
    """图像/语言/Query同一次VLM前向；显式原生mRoPE避免inputs_embeds路径丢失视觉位置。"""
    ids,mask=inputs['input_ids'],inputs['attention_mask']
    if ids.shape[1]+1>max_length:raise ValueError('Four-frame Query exceeds model context budget')
    q=query.to(vlm.get_input_embeddings().weight.dtype).expand(len(ids),-1,-1)
    embeds=torch.cat([vlm.get_input_embeddings()(ids),q],1)
    ext_mask=torch.cat([mask,torch.ones_like(mask[:,:1])],1)
    ext_ids=torch.cat([ids,torch.zeros_like(ids[:,:1])],1)
    # Query位置使用一个普通占位id计算，不加入词表；实际输入向量始终是learned parameter。
    ordinary=next(i for i in range(8) if i not in (vlm.config.image_token_id,vlm.config.video_token_id,
                                               vlm.config.vision_start_token_id,vlm.config.vision_end_token_id))
    ext_ids[:,-1]=ordinary
    positions,_=vlm.model.get_rope_index(ext_ids,inputs['image_grid_thw'],attention_mask=ext_mask)
    output=vlm(inputs_embeds=embeds,attention_mask=ext_mask,position_ids=positions,
        pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw'],use_cache=False,
        output_hidden_states=True,return_dict=True,logits_to_keep=1)
    return output.hidden_states[-1][:,-1]
