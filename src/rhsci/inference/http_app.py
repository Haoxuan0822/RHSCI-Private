"""SIMPLE HTTP 路由；预测、状态说明与 reset 由策略适配器提供。"""

import threading


def create_policy_app(
    predict_action,
    *,
    health=None,
    reset=None,
    lock=None,
    error_status=422,
    errors=(ValueError, KeyError),
):
    """返回 FastAPI 应用；串行执行策略调用，保留调用方的错误与 reset 语义。"""
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    lock = lock if lock is not None else threading.Lock()

    @app.post("/act")
    def act(payload: dict):
        try:
            with lock:
                return predict_action(payload)
        except errors as error:
            raise HTTPException(status_code=error_status, detail=str(error)) from error

    @app.get("/health")
    def get_health():
        return health() if health is not None else {"status": "ok"}

    if reset is not None:

        @app.post("/reset")
        def reset_policy():
            with lock:
                reset()
            return {"status": "ok"}

    return app
