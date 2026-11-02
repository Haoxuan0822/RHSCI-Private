"""SIMPLE HTTP 路由；预测、状态说明与 reset 由策略适配器提供。"""

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from fastapi import FastAPI


class ActionPolicy(Protocol):
    """HTTP 适配器需要的最小策略接口；动作分块和状态由模型自己管理。"""

    def predict_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """接收 SIMPLE 请求字典，返回已编码的 SIMPLE 响应字典。"""
        ...


def create_policy_app(
    predict_action: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    health: Callable[[], dict[str, Any]] | None = None,
    reset: Callable[[], None] | None = None,
    lock: AbstractContextManager[Any] | None = None,
    error_status: int = 422,
    errors: tuple[type[Exception], ...] = (ValueError, KeyError),
) -> "FastAPI":
    """用预测函数构造 /act、/health 和可选 /reset；创建应用时不调用策略。

    /act 和 /reset 共用锁。只把 errors 中的异常转成指定 HTTP 状态，其他异常沿用
    FastAPI 行为；不添加模型检查、不改变预测结果或请求中的状态。
    """
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
