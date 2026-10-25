"""Build the native GR00T HTTP server with current-frame RHSCI inference."""


def build_server(checkpoint, device="cuda:0", *, deterministic_eval=False, eval_seed=0):
    from gr00t.deploy.gr00t_serve_simple import Server, ServerConfig

    from rsci.backbones.gr00t.data import TAG

    server = Server(
        ServerConfig(
            model_path=str(checkpoint), embodiment_tag=TAG, device=device, action_exec_horizon=16
        )
    )
    server.modality_configs["video"].delta_indices = [0]
    policy = server.policy.policy
    policy.modality_configs["video"].delta_indices = [0]
    if policy.processor.training or policy.model.training:
        raise ValueError("GR00T serving requires eval mode")
    from rsci.inference.action_noise import install_server

    return install_server(server, enabled=deterministic_eval, base_seed=eval_seed)
