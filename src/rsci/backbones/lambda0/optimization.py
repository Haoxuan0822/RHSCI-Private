"""展开原生参数默认值，并检查实际优化器覆盖与逐组有效参数。"""

from collections import Counter


def native_arguments(settings):
    from lambda0.training import engine

    args = engine.get_args_parser().parse_args(
        ["--models", settings["models"], "--train_metas_path", settings["train_metas_path"]]
    )
    for key, value in settings.items():
        setattr(args, key, value)
    return args


def effective_arguments(settings, variant):
    args = native_arguments(settings)
    if variant == "VGQ-AE":
        args.use_proprio = False
        args.state_conditioning_mode = "continuous_action_expert"
    return vars(args)


def optimizer_report(model, optimizer):
    named = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in named.items()}
    expected = {id(parameter) for parameter in named.values() if parameter.requires_grad}
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    counts = Counter(map(id, parameters))
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError(
            "Optimizer must cover every trainable parameter exactly once, with no frozen parameters"
        )
    groups = []
    for group in optimizer.param_groups:
        values = {key: value for key, value in group.items() if key != "params"}
        values.update(
            tensors=len(group["params"]),
            numel=sum(parameter.numel() for parameter in group["params"]),
            parameter_dtypes=dict(Counter(str(parameter.dtype) for parameter in group["params"])),
            auxiliary_parameters=[
                names[id(parameter)]
                for parameter in group["params"]
                if names[id(parameter)].startswith("rsci_")
            ],
        )
        groups.append(values)
    return dict(
        implementation=f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        constructor_defaults_not_group_overrides=optimizer.defaults,
        effective_groups=groups,
        allocated_state_entries=len(optimizer.state),
        trainable_numel=sum(parameter.numel() for parameter in parameters),
        frozen_numel=sum(
            parameter.numel() for parameter in named.values() if not parameter.requires_grad
        ),
        lazy_state_note="Adam moments are allocated on the first gradient-bearing update; no FP32 master copy is added",
    )
