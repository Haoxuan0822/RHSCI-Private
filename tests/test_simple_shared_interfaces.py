"""共享 HTTP 包装与原生部署兼容接口的端到端 CPU 回归。"""

import json
import threading
import unittest
from http.server import HTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
from fastapi.testclient import TestClient

from rhsci.inference.http_app import create_policy_app
from rhsci.inference.protocol import RequestMessage, ResponseMessage


class SimpleInterfaceTests(unittest.TestCase):
    def test_native_imports_are_the_shared_wire_classes(self):
        from psi.deploy import helpers

        self.assertIs(helpers.RequestMessage, RequestMessage)
        self.assertIs(helpers.ResponseMessage, ResponseMessage)
        # 历史原生服务器使用 wildcard import，保留这些公开名字。
        for name in ("torch", "np", "JSONResponse", "Dict", "Path"):
            self.assertTrue(hasattr(helpers, name))

    def test_noncontiguous_numpy_roundtrip_and_optional_http_routes(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[:, ::-1]
        states = np.arange(32, dtype=np.float32)[None]
        action = np.arange(16 * 36, dtype=np.float32).reshape(16, 36)
        resets = []

        def predict(payload):
            request = RequestMessage.deserialize(payload)
            np.testing.assert_array_equal(request.image["rgb"], image)
            np.testing.assert_array_equal(request.state["states"], states)
            return ResponseMessage(action, 0.0).serialize()

        request = RequestMessage(
            {"rgb": image}, "lift", {}, {"states": states}, {}, [], "simple", "0"
        )
        app = create_policy_app(
            predict, health=lambda: {"model": "fixture"}, reset=lambda: resets.append(True)
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"model": "fixture"})
            response = client.post("/act", json=request.serialize())
            self.assertEqual(response.status_code, 200)
            np.testing.assert_array_equal(
                ResponseMessage.deserialize(response.json()).action, action
            )
            self.assertEqual(client.post("/reset").status_code, 200)
            self.assertEqual(resets, [True])
        with TestClient(create_policy_app(predict)) as client:
            self.assertEqual(client.post("/reset").status_code, 404)
            self.assertEqual(client.post("/act", json={}).status_code, 422)

    def test_egovla_standard_http_handler_uses_the_same_wire_format(self):
        from human_plan.rhsci import serve

        action = np.arange(30 * 36, dtype=np.float32).reshape(30, 36)
        image = np.zeros((2, 3, 3), np.uint8)
        request = RequestMessage(
            {"rgb_head_stereo_left": image}, "lift", {}, {}, {}, [], "simple", "0"
        )

        def predict(model, processor, payload):
            np.testing.assert_array_equal(payload["image"]["rgb_head_stereo_left"], image)
            return action

        server = HTTPServer(
            ("127.0.0.1", 0), serve.make_handler(SimpleNamespace(variant="VGQ-AE"), None)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch.object(serve, "predict", side_effect=predict):
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/act"
                body = json.dumps(request.serialize()).encode()
                with urlopen(
                    Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=5
                ) as response:
                    payload = json.load(response)
                np.testing.assert_array_equal(ResponseMessage.deserialize(payload).action, action)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
