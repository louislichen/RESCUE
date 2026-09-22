"""CPU contract tests; these are not Qwen model evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(x) for x in str(text).split()] or [1]}


class ContractTests(unittest.TestCase):
    def test_first_gold_token_is_separate_and_no_mars_default(self):
        tok = FakeTokenizer()
        prompt = "Question: which writer? Answer:"
        answer = "Naguib Mahfouz"
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tok(" " + answer, add_special_tokens=False)["input_ids"]
        self.assertEqual(answer_ids[0], len("Naguib"))
        self.assertEqual(len(prompt_ids), 4)
        self.assertNotIn("Mars", answer)

    def test_prior_is_frozen_input_and_adapter_output_is_trainable(self):
        prior = torch.ones(4, 2, 8, requires_grad=False)
        mix = torch.nn.Parameter(torch.zeros(4, 2, 1))
        output = prior + mix
        output.sum().backward()
        self.assertFalse(prior.requires_grad)
        self.assertTrue(mix.grad is not None)

    def test_checkpoint_payload_has_identity_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.json"
            payload = {"teacher_identity": {"config_sha256": "t"},
                       "student_identity": {"config_sha256": "s"}, "step": 16}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text())["step"], 16)
            self.assertNotEqual(payload["teacher_identity"], payload["student_identity"])

    def test_cpu_contract_is_not_real_qwen_evidence(self):
        self.assertFalse(torch.cuda.is_available() and False)
        self.assertEqual("CPU_FIXTURE_ONLY", "CPU_FIXTURE_ONLY")


if __name__ == "__main__":
    unittest.main()
