"""使用固定官方源码的真实 HTTP 客户端核验 RHSCI 编解码，不加载仿真或模型。"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from check_simple import check

from rsci.inference.history_client import CausalRgbHistory


@unittest.skipUnless(os.environ.get("SIMPLE_HOME"), "设置 SIMPLE_HOME 指向官方固定源码")
class OfficialProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ["SIMPLE_HOME"])
        check(source)
        # 单独加载官方纯 Python 客户端，不伪装或导入不可用的 Isaac/WBC 模块。
        spec = importlib.util.spec_from_file_location(
            "official_simple_client", source / "src/simple/baselines/client.py"
        )
        cls.client_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.client_module)

    def roundtrip(self, condition):
        from fastapi.testclient import TestClient

        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from rsci.inference.http_server import create_app

        received = []
        frame = np.full((3, 4, 3), 7, dtype=np.uint8)
        state = np.arange(32, dtype=np.float32)[None]
        actions = np.arange(16 * 36, dtype=np.float32).reshape(16, 36)

        class Server:
            def predict_action(self, payload):
                message = RequestMessage.deserialize(payload)
                received.append(message)
                return ResponseMessage(action=actions, err=0.0).serialize()

        with TestClient(create_app(Server())) as http:
            # 真实官方 query_action 生成请求；仅将网络投递替换为本地 ASGI 服务。
            def post(url, json):
                self.assertEqual(url, "http://127.0.0.1:8125/act")
                return http.post("/act", json=json)

            with patch.object(self.client_module.requests, "post", side_effect=post):
                action, error, image = self.client_module.HttpActionClient(
                    "127.0.0.1", 8125
                ).query_action(
                    {"rgb_head_stereo_left": frame},
                    "pick the box",
                    {"states": state},
                    condition,
                    history={"reset": True},
                    dataset="simple",
                )
        np.testing.assert_array_equal(action, actions)
        self.assertEqual(error, 0.0)
        self.assertEqual(image.shape, (1, 1, 3))
        message = received[0]
        np.testing.assert_array_equal(message.image["rgb_head_stereo_left"], frame)
        np.testing.assert_array_equal(message.state["states"], state)
        self.assertEqual(message.history, {"reset": True})
        self.assertEqual(message.dataset_name, "simple")
        return message

    def test_current_frame_request_and_action_response(self):
        message = self.roundtrip({})
        self.assertEqual(message.condition, {})

    def test_causal_rgb_condition_survives_official_wire_format(self):
        history = CausalRgbHistory()
        for step in range(8):
            history.append(np.full((3, 4, 3), step, dtype=np.uint8))
        message = self.roundtrip(history.condition())
        self.assertEqual(message.condition["rsci_history_protocol"], "rsci-control-rgb-50hz-v1")
        self.assertEqual(message.condition["rsci_rgb_frames"][:, 0, 0, 0].tolist(), [1, 3, 5, 7])
        self.assertTrue(message.condition["rsci_history_valid"].all())


if __name__ == "__main__":
    unittest.main()
