"""CPU-only scope/dispatch proofs using metadata fakes, never CUDA/Triton."""
import hashlib
import importlib
import inspect
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from exp.round_overhead import fp32_splitkv_target_adapter_v1 as adapter


class FakeTensor:
    def __init__(self, shape, dtype='fp32', device='cuda'):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = types.SimpleNamespace(type=device)


class Attention:
    def __setattr__(self, name, value):
        if name == 'forward' and getattr(self, 'reject_private', False) and isinstance(value, types.MethodType):
            if isinstance(value.__func__.__globals__.get('F'), adapter._FunctionalProxy):
                raise RuntimeError('injected assignment failure')
        object.__setattr__(self, name, value)


def fixtures():
    torch = types.ModuleType('torch')
    torch.float32, torch.bfloat16, torch.Tensor = 'fp32', 'bf16', FakeTensor
    torch.grad = False
    torch.is_grad_enabled = lambda: torch.grad
    torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                                          cudnn=types.SimpleNamespace(allow_tf32=False))
    functional = types.ModuleType('torch.nn.functional')
    original_calls, candidate_calls, factories = [], [], []
    original_result, candidate_result = object(), object()
    def original(*args, **kwargs):
        original_calls.append((args, kwargs)); return original_result
    functional.scaled_dot_product_attention = original
    def candidate(*args, **kwargs):
        candidate_calls.append((args, kwargs)); return candidate_result
    def factory(fallback, *, enabled, cache_dir):
        factories.append((fallback, enabled, cache_dir)); return candidate
    path = adapter.ROOT/'rescue/efficient_attention.py'
    code = adapter._reference_forward_code(path, str(path))
    namespace = dict(torch=torch, F=functional, __name__='rescue.efficient_attention')
    model = types.SimpleNamespace(config=types.SimpleNamespace(model_type='qwen2', num_hidden_layers=28,
        num_attention_heads=28, num_key_value_heads=4, hidden_size=3584, head_dim=128))
    layers, old = [], []
    for index in range(28):
        attention = Attention()
        attention._rescue_efficient = True
        attention.layer_idx, attention.num_heads, attention.num_key_value_heads = index, 28, 4
        attention.num_key_value_groups, attention.head_dim, attention.hidden_size = 7, 128, 3584
        source = types.FunctionType(code, namespace, 'forward', (None, None, None, False, False, original))
        source.__qualname__ = 'install_efficient_attention.<locals>.forward'
        source.__module__ = 'rescue.efficient_attention'
        source.__annotations__ = {'return': 'unchanged'}
        attention.forward = types.MethodType(source, attention)
        old.append(attention.forward)
        layers.append(types.SimpleNamespace(self_attn=attention))
    model.model = types.SimpleNamespace(layers=layers)
    return types.SimpleNamespace(torch=torch, F=functional, factory=factory, model=model, old=old,
        original_calls=original_calls, candidate_calls=candidate_calls, factories=factories,
        original_result=original_result, candidate_result=candidate_result, namespace=namespace)


def operands(q=3, n=4096, dtype='fp32', device='cuda'):
    return (FakeTensor((4, 7, q, 128), dtype, device), FakeTensor((4, 7, n, 128), dtype, device),
            FakeTensor((4, 7, n, 128), dtype, device))


class TargetScopeTests(unittest.TestCase):
    def activate(self, fixture, scope):
        return patch.object(adapter, '_dependencies', return_value=(fixture.torch, fixture.F, fixture.factory))

    def assert_restored(self, fixture):
        for layer, old in zip(fixture.model.model.layers, fixture.old):
            self.assertIs(layer.self_attn.forward, old)
            self.assertIs(old.__func__.__globals__['F'], fixture.F)

    def test_default_off_no_import_model_access_or_factory(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError('model touched: '+name)
        model = Untouchable()
        with patch.object(adapter.importlib, 'import_module', side_effect=AssertionError('heavy import')):
            scope = adapter.TargetAttentionScope(model)
            with scope as actual:
                self.assertIs(actual, model)
            self.assertEqual(scope.stats()['calls'], 0)
            self.assertEqual(scope.stats()['state'], 'closed')
        code = ('import sys; from exp.round_overhead.fp32_splitkv_target_adapter_v1 import TargetAttentionScope; '
                'm=object(); s=TargetAttentionScope(m); assert s.__enter__() is m; s.__exit__(None,None,None); '
                'assert not any(x=="torch" or x.startswith("torch.") or x=="triton" or x.startswith("triton.") for x in sys.modules)')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=adapter.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_has_no_runtime_factory_injection_or_cuda_contract_bypass(self):
        self.assertEqual(set(inspect.signature(adapter.TargetAttentionScope).parameters), {'model', 'enabled', 'cache_dir'})
        with self.assertRaises(ValueError):
            adapter.TargetAttentionScope(object(), enabled=1)

    def test_all28_private_globals_clone_properties_and_strict_restore(self):
        f = fixtures()
        before_globals = dict(f.namespace)
        draft = types.SimpleNamespace(forward=object())
        draft_forward = draft.forward
        source_sha = hashlib.sha256((adapter.ROOT/'rescue/efficient_attention.py').read_bytes()).hexdigest()
        scope = adapter.TargetAttentionScope(f.model, enabled=True, cache_dir='/owned/example')
        with self.activate(f, scope), scope as model:
            self.assertIs(model, f.model)
            self.assertEqual(len(f.factories), 1)
            self.assertEqual(f.factories[0], (f.F.scaled_dot_product_attention, True, '/owned/example'))
            for layer, old in zip(model.model.layers, f.old):
                new = layer.self_attn.forward
                self.assertIsNot(new, old)
                self.assertIs(new.__self__, old.__self__)
                self.assertIs(new.__func__.__code__, old.__func__.__code__)
                self.assertIs(new.__func__.__defaults__, old.__func__.__defaults__)
                self.assertIs(new.__func__.__closure__, old.__func__.__closure__)
                self.assertIs(new.__func__.__kwdefaults__, old.__func__.__kwdefaults__)
                self.assertEqual(new.__func__.__annotations__, old.__func__.__annotations__)
                self.assertIsNot(new.__func__.__globals__, old.__func__.__globals__)
                self.assertIsInstance(new.__func__.__globals__['F'], adapter._FunctionalProxy)
                self.assertIs(old.__func__.__globals__['F'], f.F)
            self.assertIs(draft.forward, draft_forward)
        self.assert_restored(f)
        self.assertEqual(f.namespace, before_globals)
        self.assertEqual(hashlib.sha256((adapter.ROOT/'rescue/efficient_attention.py').read_bytes()).hexdigest(), source_sha)

    def test_metadata_dispatch_candidate_q2to5_and_original_q1_prefill_bf16_cpu_causal(self):
        f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
        with self.activate(f, scope), scope:
            proxy = f.model.model.layers[9].self_attn.forward.__func__.__globals__['F']
            for q in (2, 3, 4, 5):
                args, kw = operands(q), dict(attn_mask=None, dropout_p=0., is_causal=False)
                self.assertIs(proxy.scaled_dot_product_attention(*args, **kw), f.candidate_result)
                self.assertEqual(f.candidate_calls[-1], (args, kw))
            for args, kw, reason in ((operands(1), {}, 'query_length_1'), (operands(2048), {}, 'query_length_out_of_scope'),
                                    (operands(dtype='bf16'), {}, 'non_fp32'), (operands(device='cpu'), {}, 'non_cuda'),
                                    (operands(), {'is_causal': True}, 'causal')):
                self.assertIs(proxy.scaled_dot_product_attention(*args, **kw), f.original_result)
                self.assertEqual(scope.stats()['records'][-1]['fallback_reason'], reason)
                self.assertEqual(f.original_calls[-1], (args, kw))
            self.assertEqual(scope.stats()['candidate_calls'], 4)
            self.assertEqual(scope.stats()['original_calls'], 5)
            self.assertTrue(all(row['layer'] == 9 and row['outcome'] == 'returned' for row in scope.stats()['records']))
        self.assert_restored(f)

    def test_original_output_attentions_fallback_unchanged_no_proxy_call(self):
        f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
        with self.activate(f, scope), scope:
            result = f.model.model.layers[0].self_attn.forward(object(), output_attentions=True)
            self.assertIs(result, f.original_result)
            self.assertEqual(scope.stats()['calls'], 0)
        self.assert_restored(f)

    def test_candidate_error_propagates_no_silent_fallback_and_restores(self):
        f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
        error = RuntimeError('candidate failed')
        def failure(*args, **kwargs):
            raise error
        f.factory = lambda *args, **kwargs: failure
        with self.assertRaises(RuntimeError) as caught:
            with self.activate(f, scope), scope:
                proxy = f.model.model.layers[0].self_attn.forward.__func__.__globals__['F']
                proxy.scaled_dot_product_attention(*operands())
        self.assertIs(caught.exception, error)
        self.assertEqual(len(f.original_calls), 0)
        self.assertEqual(scope.stats()['records'][-1]['outcome'], 'error')
        self.assert_restored(f)

    def test_partial_installation_error_restores_all_original_bound_methods(self):
        f = fixtures(); f.model.model.layers[11].self_attn.reject_private = True
        scope = adapter.TargetAttentionScope(f.model, enabled=True)
        with self.activate(f, scope), self.assertRaisesRegex(RuntimeError, 'assignment failure'):
            scope.__enter__()
        self.assertEqual(scope.state, 'failed')
        self.assert_restored(f)

    def test_factory_failure_and_invalid_model_never_mutate_forward(self):
        for failure in ('factory', 'config', 'layer', 'source_code', 'source_sha', 'module_global'):
            with self.subTest(failure=failure):
                f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
                if failure == 'factory':
                    f.factory = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('factory'))
                elif failure == 'config':
                    f.model.config.num_attention_heads = 12
                elif failure == 'layer':
                    f.model.model.layers[-1].self_attn._rescue_efficient = False
                elif failure == 'source_code':
                    old = f.model.model.layers[-1].self_attn.forward
                    fn = types.FunctionType((lambda self: None).__code__, old.__func__.__globals__, 'forward')
                    fn.__qualname__ = old.__func__.__qualname__
                    f.model.model.layers[-1].self_attn.forward = types.MethodType(fn, old.__self__)
                    f.old[-1] = f.model.model.layers[-1].self_attn.forward
                elif failure == 'module_global':
                    f.namespace['F'] = types.ModuleType('other')
                with self.activate(f, scope):
                    if failure == 'source_sha':
                        with patch.object(adapter, 'EFFICIENT_SOURCE_SHA256', '0'*64), self.assertRaises(ValueError):
                            scope.__enter__()
                    else:
                        with self.assertRaises((ValueError, RuntimeError)):
                            scope.__enter__()
                for layer, old in zip(f.model.model.layers, f.old):
                    self.assertIs(layer.self_attn.forward, old)

    def test_no_grad_tf32_required_at_entry_and_each_dispatch(self):
        for flag in ('grad', 'matmul', 'cudnn'):
            f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
            if flag == 'grad': f.torch.grad = True
            elif flag == 'matmul': f.torch.backends.cuda.matmul.allow_tf32 = True
            else: f.torch.backends.cudnn.allow_tf32 = True
            with self.activate(f, scope), self.assertRaises(ValueError):
                scope.__enter__()
            self.assert_restored(f)
        f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
        with self.activate(f, scope), scope:
            proxy = f.model.model.layers[0].self_attn.forward.__func__.__globals__['F']
            f.torch.grad = True
            with self.assertRaises(ValueError):
                proxy.scaled_dot_product_attention(*operands())
            self.assertEqual(scope.stats()['calls'], 0)
        self.assert_restored(f)

    def test_nested_scope_rejected_without_harming_outer_then_two_arms_sequential(self):
        f = fixtures()
        for arm in ('rescue_graph', 'retrieval32_k4_graph'):
            outer = adapter.TargetAttentionScope(f.model, enabled=True)
            with self.activate(f, outer), outer:
                current = [layer.self_attn.forward for layer in f.model.model.layers]
                inner = adapter.TargetAttentionScope(f.model, enabled=True)
                with self.activate(f, inner), self.assertRaises(ValueError):
                    inner.__enter__()
                self.assertTrue(all(layer.self_attn.forward is old for layer, old in zip(f.model.model.layers, current)))
            self.assert_restored(f)
            with self.assertRaises(ValueError):
                outer.__enter__()

    def test_stats_are_detached_cpu_metadata_and_never_tensor_values(self):
        f = fixtures(); scope = adapter.TargetAttentionScope(f.model, enabled=True)
        with self.activate(f, scope), scope:
            proxy = f.model.model.layers[27].self_attn.forward.__func__.__globals__['F']
            proxy.scaled_dot_product_attention(*operands(5, 8192))
            stats = scope.stats()
            self.assertEqual(stats['records'], [dict(layer=27, query_tokens=5, key_tokens=8192,
                action='candidate', fallback_reason=None, outcome='returned')])
            stats['records'][0]['layer'] = -1
            self.assertEqual(scope.stats()['records'][0]['layer'], 27)
            self.assertEqual((stats['tensor_readbacks'], stats['tensor_allocations_by_adapter']), (0, 0))
            self.assertFalse(stats['cuda_or_model_validated'])
            self.assertEqual(stats['token_gate'], 'NOT_RUN')


if __name__ == '__main__':
    unittest.main()
