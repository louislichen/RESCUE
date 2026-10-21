"""CPU-only contracts/static checks. NO CUDA compile, numeric or speed proof."""
import ast
import builtins
import inspect
import os
from pathlib import Path
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

import torch

from exp.round_overhead import fp32_splitkv_candidate_v1 as candidate


def tensors(q=3, n=129):
    query = torch.empty((1, q, 28, 128)).transpose(1, 2).reshape(4, 7, q, 128)
    key = torch.empty((4, 1, n, 128)).expand(4, 7, n, 128)
    value = torch.empty_like(key[:, :1]).expand_as(key)
    return query, key, value


def contract(query, key, value, mask=None, **kwargs):
    options = dict(dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, require_cuda=False)
    options.update(kwargs)
    return candidate._contract(query, key, value, mask, **options)


class SplitKvCpuContractTests(unittest.TestCase):
    def test_disabled_callable_identity_no_import_allocation_env_or_files(self):
        def fallback(*args, **kwargs):
            return args, kwargs
        state = dict(os.environ)
        with patch.object(candidate.importlib, 'import_module', side_effect=AssertionError('unexpected import')), \
             patch.object(torch, 'empty', side_effect=AssertionError('unexpected Tensor allocation')), \
             patch.object(candidate.Path, 'stat', side_effect=AssertionError('unexpected filesystem access')):
            actual = candidate.build_candidate(fallback, cache_dir='/does/not/exist')
        self.assertIs(actual, fallback)
        self.assertEqual(actual(3, marker=True), ((3,), {'marker': True}))
        self.assertEqual(dict(os.environ), state)
        for enabled in (0, 1, None, 'yes'):
            with self.assertRaises(ValueError):
                candidate.build_candidate(fallback, enabled=enabled)

    def test_module_import_itself_is_lazy_and_contains_no_torch_or_triton_import(self):
        source = Path(candidate.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(not name.name.startswith(('torch', 'triton')) for name in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('torch', 'triton')))
        namespace = {'__name__': 'isolated_splitkv_contract_import', '__file__': candidate.__file__}
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(('torch', 'triton')):
                raise AssertionError('Heavy import at module load')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            exec(compile(source, candidate.__file__, 'exec'), namespace)
            token = lambda: None
            self.assertIs(namespace['build_candidate'](token), token)

    def test_shape_q2_to5_and_broadcast_head_zero_stride(self):
        for q in range(2, 6):
            query, key, value = tensors(q=q)
            meta = contract(query, key, value)
            self.assertEqual((meta['queries'], meta['n']), (q, 129))
            self.assertEqual(meta['key_strides'][1], 0)
            self.assertEqual(meta['value_strides'][1], 0)
            self.assertEqual(meta['mask_kind'], 0)

    def test_float_and_bool_rank4_masks_broadcast_without_materializing(self):
        inputs = tensors(q=3, n=129)
        for dtype, kind in ((torch.float32, 1), (torch.bool, 2)):
            for shape in ((1, 1, 3, 129), (4, 7, 3, 129), (1, 7, 1, 129), (1, 1, 1, 1)):
                mask = torch.empty(shape, dtype=dtype)
                meta = contract(*inputs, mask)
                expected = tuple(0 if size == 1 else stride for size, stride in zip(shape, mask.stride()))
                self.assertEqual(meta['mask_strides'], expected)
                self.assertEqual(meta['mask_kind'], kind)
        for mask in (torch.empty(3, 129), torch.empty(1, 1, 2, 129),
                     torch.empty(1, 1, 3, 128), torch.empty(1, 1, 3, 129, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                contract(*inputs, mask)

    def test_runtime_length_and_length_dependent_strides_are_not_static_signature_constants(self):
        names = inspect.signature(candidate._stage_one).parameters
        required = {'N', 'K0', 'K1', 'K2', 'V0', 'V1', 'V2', 'M0', 'M1', 'M2', 'M3'}
        self.assertTrue(required <= set(candidate.DYNAMIC_STAGE_ONE))
        for name in candidate.DYNAMIC_STAGE_ONE:
            self.assertIn(name, names)
            self.assertEqual(names[name].annotation, inspect.Parameter.empty)
        for name in ('QUERIES', 'HEADS', 'D', 'SPLIT_COUNT', 'TILE_K', 'MASK_KIND', 'SCALE'):
            self.assertEqual(names[name].annotation, 'tl.constexpr')
        first = contract(*tensors(n=127), torch.empty(1, 1, 3, 127))
        second = contract(*tensors(n=129), torch.empty(1, 1, 3, 129))
        self.assertNotEqual(first['key_strides'][0], second['key_strides'][0])
        self.assertNotEqual(first['mask_strides'][2], second['mask_strides'][2])
        self.assertEqual(set(candidate.DYNAMIC_STAGE_ONE), set(names)-{
            'Q', 'K', 'V', 'MASK', 'PM', 'PL', 'PO', 'QUERIES', 'HEADS', 'D',
            'SPLIT_COUNT', 'TILE_K', 'MASK_KIND', 'SCALE'})

    def test_jit_factory_uses_private_globals_and_both_dynamic_guards(self):
        calls, tl_module = [], types.ModuleType('mock_triton_language')
        def jit(**kwargs):
            def decorate(function):
                calls.append((function, kwargs))
                return function
            return decorate
        default_cache = type('FileCacheManager', (), {})
        cache_module = types.SimpleNamespace(FileCacheManager=default_cache,
            **{'__cache_cls': default_cache, '__cache_cls_nme': 'DEFAULT'})
        modules = {'triton': types.SimpleNamespace(jit=jit), 'triton.language': tl_module,
                   'triton.runtime.cache': cache_module}
        with patch.object(candidate.importlib, 'import_module', side_effect=lambda name: modules[name]):
            kernels = candidate._load_kernels()
        self.assertEqual(len(kernels), 2)
        self.assertNotIn('tl', candidate.__dict__)
        for index, (function, kwargs) in enumerate(calls):
            self.assertIs(function.__globals__['tl'], tl_module)
            self.assertIsNot(function.__globals__, candidate.__dict__)
            expected = candidate.DYNAMIC_STAGE_ONE if index == 0 else ()
            self.assertEqual(kwargs, dict(do_not_specialize=expected, do_not_specialize_on_alignment=expected))

    def test_sticky_custom_cache_manager_rejected_after_environment_is_cleared(self):
        default, custom = type('FileCacheManager', (), {}), type('CustomCacheManager', (), {})
        for state, name, accepted in ((default, 'DEFAULT', True), (custom, 'custom:Cache', False),
                                      (custom, 'DEFAULT', False), (default, 'custom:Cache', False),
                                      (None, 'DEFAULT', False)):
            module = types.SimpleNamespace(FileCacheManager=default,
                **{'__cache_cls': state, '__cache_cls_nme': name})
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(candidate.importlib, 'import_module', return_value=module):
                if accepted:
                    candidate._assert_default_cache_manager()
                else:
                    with self.assertRaisesRegex(ValueError, 'untouched default'):
                        candidate._assert_default_cache_manager()
            # Validation is read-only, including in the sticky failure case.
            self.assertIs(getattr(module, '__cache_cls'), state)
            self.assertEqual(getattr(module, '__cache_cls_nme'), name)
        with patch.object(candidate.importlib, 'import_module', return_value=types.SimpleNamespace()):
            with self.assertRaisesRegex(ValueError, 'untouched default'):
                candidate._assert_default_cache_manager()
        # A stale custom manager is rejected by the real loader before tl/JIT
        # can be accessed, even though there is no environment override now.
        stale = types.SimpleNamespace(FileCacheManager=default,
            **{'__cache_cls': custom, '__cache_cls_nme': 'custom:Cache'})
        names = []
        def modules(name):
            names.append(name)
            return types.SimpleNamespace() if name == 'triton' else stale
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(candidate.importlib, 'import_module', side_effect=modules):
            with self.assertRaisesRegex(ValueError, 'untouched default'):
                candidate._load_kernels()
        self.assertEqual(names, ['triton', 'triton.runtime.cache'])
        source = inspect.getsource(candidate._Candidate.__call__)
        self.assertIn('else:\n                _assert_default_cache_manager()', source)

    def test_fixed_key_partition_covers_nonmultiples_and_empty_splits(self):
        self.assertEqual((candidate.SPLITS, candidate.BLOCK_K, candidate.HEAD_DIM, candidate.MAX_KEYS), (8, 128, 128, 65536))
        for n in (2, 3, 127, 128, 129, 1023, 1024, 1025, 4096, 8193, 16384, 16512, 65535, 65536):
            ranges = candidate.split_key_ranges(n)
            self.assertEqual(len(ranges), 8)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], n)
            self.assertEqual(sum(end-start for start, end in ranges), n)
            self.assertTrue(all(ranges[i][1] == ranges[i+1][0] for i in range(7)))
            self.assertTrue(all(start == n or start % 128 == 0 for start, _ in ranges))
        self.assertTrue(any(start == end for start, end in candidate.split_key_ranges(3)))
        for n in (0, -1, True, 65537, 2**31):
            with self.assertRaises(ValueError):
                candidate.split_key_ranges(n)
        query = torch.empty((4, 7, 3, 128))
        for n, accepted in ((65536, True), (65537, False), (2**31-1, False)):
            # Expanded views make these range-bound checks allocation-free and
            # distinguish N scope from the separate strided-address limit.
            key = torch.empty((4, 1, 1, 128)).expand(4, 7, n, 128)
            if accepted:
                self.assertEqual(contract(query, key, key)['n'], n)
            else:
                with self.assertRaisesRegex(ValueError, '65536 keys'):
                    contract(query, key, key)

    def test_unsupported_dtype_shape_stride_and_autograd_rejected(self):
        inputs = tensors()
        variants = [(inputs[0].to(torch.bfloat16), inputs[1], inputs[2]),
                    (torch.empty(4, 7, 1, 128), inputs[1], inputs[2]),
                    (torch.empty(4, 7, 6, 128), inputs[1], inputs[2]),
                    (torch.empty(1, 28, 3, 128), inputs[1], inputs[2]),
                    (inputs[0], inputs[1], torch.empty(4, 7, 128, 128)),
                    (torch.empty(4, 7, 3, 256)[..., ::2], inputs[1], inputs[2]),
                    (inputs[0].clone().requires_grad_(), inputs[1], inputs[2])]
        for values in variants:
            with self.assertRaises(ValueError):
                contract(*values)
        with self.assertRaises(ValueError):
            contract(object(), inputs[1], inputs[2])
        # Metadata-only oversized inputs avoid allocation and exercise address
        # arithmetic bounds, not CUDA correctness.
        oversized = [torch.empty((4, 7, 3, 128), device='meta'),
                     torch.empty((4, 7, 2**23, 128), device='meta'),
                     torch.empty((4, 7, 2**23, 128), device='meta')]
        with self.assertRaisesRegex(ValueError, 'int32 offset'):
            contract(*oversized)

    def test_dropout_causal_scale_and_unfolded_gqa_are_explicitly_rejected(self):
        values = tensors()
        for kwargs in ({'dropout_p': 0.1}, {'dropout_p': True}, {'is_causal': True},
                       {'is_causal': 0}, {'scale': 0.1}, {'enable_gqa': True}):
            with self.assertRaises(ValueError):
                contract(*values, **kwargs)

    def test_unresolved_negative_views_rejected_for_every_pointer_input_without_resolving(self):
        query, key, value = tensors()
        mask = torch.empty((1, 1, 3, 129))
        views = (torch._neg_view(query), torch._neg_view(key), torch._neg_view(value), torch._neg_view(mask))
        self.assertTrue(all(view.is_neg() and type(view) is torch.Tensor for view in views))
        with patch.object(torch.Tensor, 'resolve_neg', side_effect=AssertionError('unexpected resolve/copy')), \
             patch.object(torch.Tensor, 'resolve_conj', side_effect=AssertionError('unexpected resolve/copy')):
            for index, view in enumerate(views):
                inputs = [query, key, value, mask]
                inputs[index] = view
                with self.assertRaisesRegex(ValueError, 'negative/conjugate'):
                    contract(*inputs)
            # FP32 currently cannot obtain a logical conjugate bit through
            # conj(); simulate that metadata to verify conservative rejection.
            with patch.object(torch.Tensor, 'is_conj', new=lambda tensor: True):
                with self.assertRaisesRegex(ValueError, 'negative/conjugate'):
                    contract(query, key, value, mask)
            with patch.object(torch.Tensor, 'is_conj', new=lambda tensor: tensor is mask):
                with self.assertRaisesRegex(ValueError, 'negative/conjugate'):
                    # The first three inputs remain ordinary; only mask reports
                    # the conjugate metadata in this independent branch check.
                    contract(query, key, value, mask)

    def test_cache_must_already_exist_be_owned_canonical_and_env_matched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(root)}, clear=False):
                with patch.object(candidate.importlib, 'import_module', side_effect=AssertionError('unexpected import')):
                    result = candidate.build_candidate(lambda: None, enabled=True, cache_dir=root)
                self.assertIsInstance(result, candidate._Candidate)
                self.assertEqual(list(root.iterdir()), [])
                with self.assertRaises(ValueError):
                    candidate.build_candidate(lambda: None, enabled=True)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    candidate.build_candidate(lambda: None, enabled=True, cache_dir=root/'missing')
                with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(root/'different')}):
                    with self.assertRaises(ValueError):
                        candidate.build_candidate(lambda: None, enabled=True, cache_dir=root)
                root.chmod(0o777)
                with self.assertRaises(ValueError):
                    candidate.build_candidate(lambda: None, enabled=True, cache_dir=root)
                root.chmod(0o700)

    def test_enabled_cpu_call_fails_before_triton_allocation_or_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(root)}, clear=False):
                fallback = unittest.mock.Mock(side_effect=AssertionError('silent fallback'))
                operation = candidate.build_candidate(fallback, enabled=True, cache_dir=root)
                inputs = tensors()
                with patch.object(candidate, '_load_kernels', side_effect=AssertionError('Triton load')), \
                     patch.object(torch, 'empty', side_effect=AssertionError('unexpected allocation')):
                    with self.assertRaisesRegex(ValueError, 'requires CUDA'):
                        operation(*inputs)
                fallback.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])
                self.assertIsNone(operation._kernels)

    def test_compile_env_guards_do_not_mutate_or_create_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            clean = {'TRITON_CACHE_DIR': str(root), 'CUDA_CACHE_DISABLE': '1'}
            with patch.dict(os.environ, clean, clear=True):
                path, identity = candidate._cache_identity(root)
                original = dict(os.environ)
                candidate._check_compile_environment(path, identity)
                self.assertEqual(dict(os.environ), original)
                for name in ('TRITON_INTERPRET', 'TRITON_KERNEL_DUMP', 'TRITON_KERNEL_OVERRIDE',
                             'TRITON_CACHE_MANAGER', 'TRITON_REMOTE_CACHE_BACKEND', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR'):
                    with patch.dict(os.environ, {name: '1'}):
                        with self.assertRaises(ValueError):
                            candidate._check_compile_environment(path, identity)
                with patch.dict(os.environ, {'TRITON_CACHE_MANAGER': ''}):
                    with self.assertRaisesRegex(ValueError, 'must be absent'):
                        candidate._check_compile_environment(path, identity)
                with patch.dict(os.environ, {'CUDA_CACHE_DISABLE': '0'}):
                    with self.assertRaises(ValueError):
                        candidate._check_compile_environment(path, identity)
                with self.assertRaises(ValueError):
                    candidate._check_compile_environment(path, (0, 0, 0, 0))
                self.assertEqual(list(root.iterdir()), [])

    def test_kernel_source_fp32_reductions_no_dot_precision_lowering_and_masked_split_paths(self):
        for kernel in (candidate._stage_one, candidate._stage_two):
            tree = ast.parse(textwrap.dedent(inspect.getsource(kernel)))
            attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
            self.assertNotIn('dot', attributes)
            self.assertNotIn('float16', attributes)
            self.assertNotIn('bfloat16', attributes)
            self.assertNotIn('float64', attributes)
            reductions = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and node.func.attr == 'sum']
            self.assertTrue(reductions)
            for call in reductions:
                dtype = next(keyword.value for keyword in call.keywords if keyword.arg == 'dtype')
                self.assertEqual(ast.unparse(dtype), 'tl.float32')
        one, two = inspect.getsource(candidate._stage_one), inspect.getsource(candidate._stage_two)
        self.assertIn('while offset < stop:', one)
        self.assertIn("new_maximum == -float('inf')", one)
        self.assertIn('MASK_KIND == 2', one)
        self.assertIn("maximum == -float('inf')", two)
        self.assertIn('denominator == 0.0', two)
        self.assertEqual(inspect.getsource(candidate._Candidate.__call__).count('enable_fp_fusion=False'), 2)

    def test_no_proof_or_engine_integration_is_claimed(self):
        for name in ('cuda_numerics_verified', 'bitwise_equivalence_claim', 'speed_verified', 'engine_integrated'):
            self.assertIs(getattr(candidate._Candidate, name), False)
        source = Path(candidate.__file__).read_text()
        self.assertNotIn('from rescue', source)
        self.assertNotIn('import rescue', source)
        self.assertNotIn('os.environ[', source)
        self.assertNotIn('.mkdir(', source)
        # A fresh-subprocess *requirement* in documentation is not a launch.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name != 'subprocess' for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, 'subprocess')
            if isinstance(node, ast.Name):
                self.assertNotEqual(node.id, 'subprocess')


if __name__ == '__main__':
    unittest.main()
