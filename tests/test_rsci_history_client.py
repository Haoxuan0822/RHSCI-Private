"""实际逐控制步历史、HTTP序列化和RHSCI服务消费契约回归。"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rsci.inference.history_client import CausalRgbHistory, install


class HistoryTests(unittest.TestCase):
    def test_padding_copy_and_reset(self):
        history = CausalRgbHistory()
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        history.append(frame)
        frame[:] = 99
        condition = history.condition()
        self.assertTrue((condition["rsci_rgb_frames"] == 0).all())
        self.assertEqual(condition["rsci_history_valid"].tolist(), [False, False, False, True])
        history.reset()
        with self.assertRaises(RuntimeError):
            history.condition()

    def test_action_queue_steps_and_serialization(self):
        captures = []

        class Client:
            def query_action(self, image_dict, instruction, state_dict, condition_dict, **kwargs):
                captures.append(condition_dict)
                return "unchanged-actions"

        class Agent:
            _control_dt = 0.02

            def __init__(self):
                self.client = Client()
                self.counter = 0

            def reset(self):
                self.counter = 0

            def get_action(self, observation):
                # 模拟chunk执行期间没有HTTP请求，但每个控制步仍须记下RGB。
                result = None
                if self.counter % 24 == 0:
                    result = self.client.query_action(
                        {"rgb_head_stereo_left": observation["head_stereo_left"]}, "task", {}, {}
                    )
                self.counter += 1
                return result

        install(Agent, Client)
        wrapped = Agent.get_action
        install(Agent, Client)
        self.assertIs(Agent.get_action, wrapped)
        agent = Agent()
        agent.reset()
        for step in range(25):
            result = agent.get_action({"head_stereo_left": np.full((2, 3, 3), step, np.uint8)})
        self.assertEqual(result, "unchanged-actions")
        condition = captures[-1]
        self.assertEqual(condition["rsci_rgb_frames"][:, 0, 0, 0].tolist(), [18, 20, 22, 24])
        self.assertTrue(condition["rsci_history_valid"].all())
        # Exercise the server's real HTTP codec without a local simulator checkout.
        from psi.deploy.helpers import convert_numpy_in_dict, numpy_deserialize, numpy_serialize

        decoded = convert_numpy_in_dict(
            json.loads(json.dumps(convert_numpy_in_dict(condition, numpy_serialize))),
            numpy_deserialize,
        )
        for key in ("rsci_rgb_frames", "rsci_history_valid", "rsci_frame_offsets"):
            np.testing.assert_array_equal(condition[key], decoded[key])
        agent.reset()
        agent.get_action({"head_stereo_left": np.full((2, 3, 3), 100, np.uint8)})
        self.assertEqual(captures[-1]["rsci_history_valid"].tolist(), [False, False, False, True])
        agent._control_dt = 0.04
        with self.assertRaises(ValueError):
            agent.get_action({"head_stereo_left": np.zeros((2, 3, 3), np.uint8)})


if __name__ == "__main__":
    unittest.main()
