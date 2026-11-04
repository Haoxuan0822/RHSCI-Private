"""仅诊断：按独立流复用原生RequestActionNoise，配对不同布局的动作噪声。"""

import functools
import inspect
import json
from concurrent.futures import ThreadPoolExecutor


def install_client(seed, stream):
    from simple.baselines.client import HttpActionClient

    from rhsci.inference.action_noise import PROTOCOL

    original = HttpActionClient.query_action
    signature = inspect.signature(original)

    @functools.wraps(original)
    def query(client, *args, **kwargs):
        bound = signature.bind(client, *args, **kwargs)
        history = dict(bound.arguments.get("history") or {})
        if "psi_eval_deterministic" in history:
            raise ValueError("Refuse to overwrite pre-existing action-noise metadata")
        index = getattr(client, "_paired_noise_index", 0)
        metadata = dict(
            protocol=PROTOCOL, episode=0, inference_index=index, episode_seed=seed, stream=stream
        )
        history["psi_eval_deterministic"] = metadata
        bound.arguments["history"] = history
        result = original(*bound.args, **bound.kwargs)
        client._paired_noise_index = index + 1
        print("DIAGNOSTIC_PAIRED_NOISE", json.dumps(metadata), flush=True)
        return result

    HttpActionClient.query_action = query


def create_app(server):
    from rhsci.inference.action_noise import RequestActionNoise
    from rhsci.inference.http_app import create_policy_app

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paired-policy-inference")
    controllers = {}

    def infer(payload):
        metadata = payload.get("history", {}).get("psi_eval_deterministic", {})
        stream, seed = metadata.get("stream"), metadata.get("episode_seed")
        if not isinstance(stream, str) or not stream or type(seed) is not int or seed < 0:
            raise ValueError("Paired-noise diagnosis requires an explicit stream and seed")
        if stream not in controllers:
            controllers[stream] = RequestActionNoise(True, seed)
        controller = controllers[stream]
        if controller.base_seed != seed:
            raise ValueError("A stream must retain its original seed")
        # 串行线程内切换各流的原生校验/噪声控制器；模型与采样公式不变。
        server.action_noise = controller
        return server.predict_action(payload)

    app = create_policy_app(lambda payload: executor.submit(infer, payload).result())
    app.add_event_handler("shutdown", executor.shutdown)
    return app


if __name__ == "__main__":
    from rhsci.inference import http_server

    http_server.create_app = create_app
    http_server.main()
