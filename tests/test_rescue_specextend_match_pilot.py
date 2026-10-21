"""CPU runner/protocol regressions only; synthetic tokens are not results."""
import ast
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from rescue.attention import DraftAttention
from rescue.baselines import BaselineEngine
from rescue.config import SystemConfig
from rescue.draft_graph_benchmark import HookOwners, owner_signature
from rescue.draft_graph_smoke import native_digest, record_queries
from rescue.efficient_attention import install_efficient_attention
from rescue.packing import pack_projections
from rescue.specextend_match_pilot import (ARMS, EXPECTED, QUALITY_ARMS,
    TREE_POLICY, Observer, check_output, check_quality, normalize_full,
    read_document, run, schedule, check_full_bindings)


def full_result(tokens, overshoot=3):
    return dict(generated=tokens, tokens=len(tokens), total_seconds=2., total_tps=len(tokens)/2.,
        upstream=dict(output_token_ids=tokens, total_generated=len(tokens),
                      raw_total_generated=len(tokens)+overshoot, generation_overshoot=overshoot,
                      accept_length_list=[2]*43, inference_time=1.9))


class PilotHelpers(unittest.TestCase):
    def test_fixed44_request_matrix_and_randomized_two_repeat_order(self):
        self.assertEqual(sum(EXPECTED[k] for k in ('reference_requests', 'correctness_requests', 'prime_requests', 'timing_requests')), 44)
        self.assertEqual(len(QUALITY_ARMS), 6)
        self.assertEqual(EXPECTED['owners'], 10)
        plan = schedule()
        self.assertEqual(plan, schedule())
        self.assertNotEqual(plan, schedule(1))
        self.assertEqual(len(plan), 20)
        self.assertEqual(len({tuple(r[k] for k in ('mode','length','repeat')) for r in plan}), 20)
        for i in range(0, 20, 5):
            self.assertEqual({v['mode'] for v in plan[i:i+5]}, set(ARMS))
            self.assertEqual(len({(v['length'],v['repeat']) for v in plan[i:i+5]}), 1)
        self.assertEqual(TREE_POLICY['nodes'], 8)
        self.assertEqual(TREE_POLICY['threshold'], .7)

    def test_full_tree_client_prefix_policy_keeps_overshoot_cost_metadata(self):
        tokens = list(range(128))
        raw = full_result(tokens)
        row = normalize_full(raw, tokens)
        check_output(row, tokens)
        self.assertEqual(row['tokens'], 128)
        self.assertEqual(row['raw_total_generated'], 131)
        self.assertEqual(row['generation_overshoot'], 3)
        self.assertEqual(row['total_seconds'], 2.)
        self.assertEqual(row['upstream'], raw['upstream'])
        self.assertFalse(row['raw_tail_ids_available'])
        self.assertEqual(row['blocks'], 43)
        self.assertEqual(row['accepted'], 86)

    def test_full_wrong_token_is_not_clipped_away_and_can_be_recorded(self):
        tokens = list(range(128)); wrong = tokens.copy(); wrong[-1] = 999
        row = normalize_full(full_result(wrong), tokens)
        self.assertFalse(row['output_agrees'])
        self.assertEqual(row['generated'][-1], 999)
        with self.assertRaisesRegex(RuntimeError, 'Client-visible output'):
            check_output(row, tokens)

    def test_full_eos_preserves_archived_scored_tokens_and_raw_count(self):
        scored = [3, 4, 9, 11]
        raw = full_result(scored, overshoot=0)
        raw.update(generated=[3, 4, 9], tokens=3)
        row = normalize_full(raw, [3, 4, 9], eos=9)
        check_output(row, [3, 4, 9], eos=9)
        self.assertEqual(row['scored_generated_before_eos'], scored)
        self.assertEqual(row['raw_total_generated'], 4)
        self.assertEqual(row['tokens'], 3)
        with self.assertRaisesRegex(RuntimeError, 'overshoot'):
            bad = full_result(list(range(128))); bad['upstream']['generation_overshoot'] = 0
            normalize_full(bad, list(range(128)))

    def test_inputs_only_doc303_exact_context_and_no_test_split(self):
        item = dict(source_index=303, split='confirmation', document_sha256='a'*64,
                    prefix_ids=[1], suffix_ids=[2], document_ids=[3]*8192)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.jsonl'; path.write_text(json.dumps(item)+'\n')
            self.assertEqual(read_document([path]), item)
            item['split'] = 'test'; path.write_text(json.dumps(item)+'\n')
            with self.assertRaisesRegex(RuntimeError, 'test'):
                read_document([path])
            item['split'] = 'confirmation'; path.write_text((json.dumps(item)+'\n')*2)
            with self.assertRaisesRegex(RuntimeError, 'Exactly one'):
                read_document([path])

    def test_run_does_not_shadow_record_or_hide_legal_warm_capture(self):
        source = inspect.getsource(run); tree = ast.parse(source)
        assigned = {n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        self.assertFalse(assigned & {'record','context','source_inventory','schedule','write','append'})
        self.assertNotIn("['captures'] == 0", source)
        self.assertIn("'outer_seconds'", source)
        self.assertIn('retain_query', source)
        self.assertIn('128', source)

    def test_full_namespace_bindings_allow_archived_pep420_but_not_foreign_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); staged = root/'exp/specextend_bridgekv/specextend'
            modules = {}
            for package, file in (('classic','model_classic'), ('shared','modeling_llama_kv_target')):
                directory = staged/package; directory.mkdir(parents=True)
                source = directory/(file+'.py'); source.write_text('# synthetic binding\n')
                module = types.ModuleType(package); module.__path__ = [str(directory)]
                child = types.ModuleType(package+'.'+file); child.__file__ = str(source)
                modules[package] = module; modules[package+'.'+file] = child
            with patch.dict('sys.modules', modules):
                self.assertEqual(len(check_full_bindings(root)), 4)
                modules['shared'].__path__ = [str(root/'foreign')]
                with self.assertRaisesRegex(RuntimeError, 'namespace escaped'):
                    check_full_bindings(root)


class PilotRetrievalHandoffCPU(unittest.TestCase):
    @torch.no_grad()
    def test_quality_owner_private_cache_queries_and_features_without_mapper(self):
        torch.manual_seed(55)
        cfg = Qwen2Config(vocab_size=48, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=256, attention_dropout=0., use_cache=True)
        teacher, student = Qwen2ForCausalLM(cfg).eval(), Qwen2ForCausalLM(cfg).eval()
        for model in (teacher, student):
            model.requires_grad_(False); install_efficient_attention(model); pack_projections(model)
        base = SystemConfig(budget=32, recent=2, max_new_tokens=7, max_draft=12,
            length_policy='fixed', fixed_k=4, attention_backend='sdpa', schedule='serial',
            working_mode='packed', optimized_execution=True, deferred_probes=True)
        group = HookOwners(teacher, student); engines = []
        try:
            observers = []
            for backend in ('eager', 'resident'):
                observer = Observer(); observers.append(observer)
                engine = BaselineEngine(teacher, student, replace(base,draft_execution=backend),
                                        top_chunks=1, round_observer=observer)
                group.add(engine); engines.append(engine)
            saved = []
            for engine, observer in zip(engines, observers):
                with record_queries(DraftAttention) as queries:
                    row = group.generate(engine, [3+i%40 for i in range(65)], None)
                row.update(observed_rounds=observer.rounds, queries=queries,
                    native_t=native_digest(engine.native_t), native_s=native_digest(engine.native_s))
                self.assertEqual(len(queries), row['draft_forwards'])
                self.assertFalse(observer.nonzero_injection)
                self.assertTrue(all(v['budget_features'] is None for v in observer.rounds))
                self.assertEqual(row['draft_shadow_probes'], 0)
                saved.append(row)
            check_quality(saved[0], saved[1])
            self.assertNotEqual(owner_signature(engines[0])['native_s']['key_ptr'],
                                owner_signature(engines[1])['native_s']['key_ptr'])
            for engine in engines:
                before = owner_signature(engine)
                group.generate(engine, [3+i%40 for i in range(65)], None)
                self.assertEqual(before, owner_signature(engine))
                group.idle()
        finally:
            for engine in list(group.engines):
                group.close(engine)
            group.idle()


if __name__ == '__main__':
    unittest.main()
