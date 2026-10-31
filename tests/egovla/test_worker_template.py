"""模拟spawn重建模块全局后，Processor仍恢复训练时的同一对话模板。"""

import pickle
import sys
import types
import unittest
from unittest.mock import patch

from human_plan.rsci.processing import Processor


class WorkerTemplateTests(unittest.TestCase):
    def test_pickled_processor_restores_template_before_each_prompt(self):
        conversation = types.ModuleType("llava.conversation")
        llava = types.ModuleType("llava")
        llava.conversation = conversation
        preprocessing = types.ModuleType("human_plan.preprocessing.preprocessing")
        prompting = types.ModuleType("human_plan.preprocessing.prompting_format")
        prompting.preprocess_language_instruction = lambda text, *args: text
        preprocessing.preprocess_multimodal_vla = lambda text, *args: text
        seen = []

        def preprocess(*args, **kwargs):
            seen.append(conversation.default_conversation)
            return {"input_ids": [1, 2]}

        preprocessing.preprocess_vla = preprocess
        processor = object.__new__(Processor)
        processor.conversation = {"template": "training"}
        processor.data_args = None
        processor.action_tokenizer = None
        processor.tokenizer = None
        restored = pickle.loads(pickle.dumps(processor))
        with patch.dict(
            sys.modules,
            {
                "llava": llava,
                "llava.conversation": conversation,
                "human_plan.preprocessing.preprocessing": preprocessing,
                "human_plan.preprocessing.prompting_format": prompting,
            },
        ):
            for _ in range(2):
                conversation.default_conversation = {"template": "fresh-worker-default"}
                restored.prompt("pick the box")
        self.assertEqual(seen, [{"template": "training"}] * 2)


if __name__ == "__main__":
    unittest.main()
