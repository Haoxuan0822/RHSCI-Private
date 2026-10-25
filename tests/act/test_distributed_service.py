import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rsci.backbones.act.model import masked_ardy_loss


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.weight.expand(2, 128)


def gloo_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + str(Path(root) / "init"), rank=rank, world_size=2
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(Scale())
        prediction = model()
        target = (
            torch.tensor([1.0, 3.0])[:, None].expand_as(prediction)
            if rank
            else torch.full_like(prediction, float("nan"))
        )
        loss = masked_ardy_loss(prediction, target, torch.tensor([bool(rank)] * 2))
        loss.backward()
        # 全局有效目标[1,3]的MSE在p=0的导数=-4；空rank不能使分母扩大一倍。
        torch.testing.assert_close(model.module.weight.grad, torch.tensor(-4.0), rtol=0, atol=0)
        (Path(root) / f"rank-{rank}.json").write_text(
            json.dumps({"gradient": model.module.weight.grad.item()})
        )
    finally:
        dist.destroy_process_group()


class DistributedAndService(unittest.TestCase):
    def test_two_rank_actual_gloo_empty_rank_global_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(gloo_worker, args=(root,), nprocs=2, join=True)
            self.assertTrue(all((Path(root) / f"rank-{rank}.json").is_file() for rank in range(2)))

    def test_http_uses_native_serialization_full_chunk_and_reset(self):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from rsci.backbones.act.serve import app_for

        class FakePolicy:
            model = SimpleNamespace(variant="EGQ-AE")
            task = "test-task"
            lock = threading.Lock()
            resets = 0

            def predict(self, image, states):
                self.asserted_image = np.asarray(image)
                return np.arange(16 * 36, dtype=np.float32).reshape(16, 36)

            def reset(self):
                self.resets += 1

        policy = FakePolicy()
        with TestClient(app_for(policy)) as client:
            image = np.zeros((224, 224, 3), np.uint8)
            request = RequestMessage(
                {"egocentric": image}, "unused language", {}, {}, {}, [], "simple", "0"
            )
            response = client.post("/act", json=request.serialize())
            self.assertEqual(response.status_code, 200)
            actions = ResponseMessage.deserialize(response.json()).action
            np.testing.assert_array_equal(actions, policy.predict(None, None))
            self.assertEqual(client.post("/reset").status_code, 200)
            self.assertEqual(policy.resets, 1)
            request.image = {"a": image, "b": image}
            self.assertEqual(client.post("/act", json=request.serialize()).status_code, 400)


if __name__ == "__main__":
    unittest.main()
