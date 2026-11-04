"""Keep shared compiled-model requests on one inference thread; reuse the native server CLI."""

from concurrent.futures import ThreadPoolExecutor

from rhsci.inference import http_server
from rhsci.inference.http_app import create_policy_app


def create_app(server):
    """多个仿真复用一份模型时，所有 CUDA Graph 推理留在同一工作线程。"""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")

    def predict(payload):
        return executor.submit(server.predict_action, payload).result()

    app = create_policy_app(predict)
    app.add_event_handler("shutdown", executor.shutdown)
    return app


if __name__ == "__main__":
    http_server.create_app = create_app
    http_server.main()
