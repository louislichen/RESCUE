"""CPU-only v3 startup ordering/failure tests; no actual GPU or process work."""
import ast
import contextlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import csgb_episode_runner as runner


UUID = 'GPU-0dab56d0-2236-4c0e-fc96-fe28081dba9b'


class StopBeforeRequests(Exception):
    pass


class HardwareGuardV3Tests(unittest.TestCase):
    def test_helper_calls_only_the_registered_readonly_guard_and_propagates_failure(self):
        args=SimpleNamespace(physical_gpu_index=0,expected_gpu_uuid=UUID)
        with patch.object(runner,'__file__','/synthetic/new_v3/specextend/csgb_episode_runner.py'),\
             patch.object(runner.subprocess,'run',return_value=object()) as run:
            runner.wait_for_preload_gpu(args)
            run.assert_called_once_with([sys.executable,'/synthetic/new_v3/csgb_episode_gpu_guard.py',
                                         '--index','0','--uuid',UUID],check=True)
        for error in (subprocess.CalledProcessError(1,['registered-guard']),
                      subprocess.TimeoutExpired(['registered-guard'],15),
                      FileNotFoundError('guard missing')):
            with self.subTest(error=type(error).__name__),\
                 patch.object(runner.subprocess,'run',side_effect=error),self.assertRaises(type(error)):
                runner.wait_for_preload_gpu(args)

    def test_static_guard_is_after_validation_imports_and_before_first_cuda(self):
        function=next(n for n in ast.parse(inspect.getsource(runner)).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
        positions=lambda name:[n.lineno for n in calls if ast.unparse(n.func)==name]
        gate=positions('wait_for_preload_gpu')
        self.assertEqual(len(gate),1)
        self.assertLess(positions('validate_checkpoint_receipt')[0],gate[0])
        imports=[n.lineno for n in function.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertLess(max(imports),gate[0])
        self.assertLess(gate[0],positions('torch.cuda.device_count')[0])
        self.assertLess(gate[0],positions('torch.empty')[0])
        load=positions('SPModel.from_pretrained')[0];convert=positions('model.base_model.float')[0]
        self.assertTrue(any(load<line<convert for line in positions('assert_gpu_boundary')))

    @contextlib.contextmanager
    def runtime(self,*,gate_error=None,postload_error=False):
        events=[]
        def checkpoint(*args):events.append('hash');return {},{}
        def wait(args):
            events.append('wait')
            if gate_error is not None:raise gate_error
        boundary_count=[0]
        def boundary(*args):
            boundary_count[0]+=1;events.append('boundary')
            if postload_error and boundary_count[0]==2:raise RuntimeError('foreign appeared during load')
        def convert():events.append('float');raise StopBeforeRequests()
        model=SimpleNamespace(base_model=SimpleNamespace(float=Mock(side_effect=convert)))
        model.eval=lambda:model
        def from_pretrained(**kwargs):events.append('load');return model
        model_loader=Mock(side_effect=from_pretrained)
        def probe(*args,**kwargs):events.append('probe');return object()
        cuda=SimpleNamespace(device_count=Mock(return_value=1),synchronize=Mock())
        torch=SimpleNamespace(cuda=cuda,empty=Mock(side_effect=probe),manual_seed=Mock(),float16='float16',
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True)),set_float32_matmul_precision=Mock())
        modules={'numpy':SimpleNamespace(random=SimpleNamespace(seed=Mock())), 'torch':torch,
            'baseline_profiler':SimpleNamespace(exact_input=Mock()),
            'classic.model_classic':SimpleNamespace(SPModel=SimpleNamespace(from_pretrained=model_loader)),
            'csgb_same_state':SimpleNamespace(release_request_caches=Mock()),
            'csgb_online_evaluate_migrated':SimpleNamespace(validate_physical_gpu=Mock(),
                validate_checkpoint_receipt=Mock(side_effect=checkpoint))}
        protocol={'method':{'seed':101},'hardware':{'physical_gpu_index':0,'expected_gpu_uuid':UUID},
            'source_sha256':'0'*64,'scaler_sha256':'0'*64,'checkpoint_receipt_sha256':'0'*64,
            'source_manifest':{},'checkpoint_training_root':'/synthetic/original'}
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack:
            out=Path(temp)/'must_not_exist'
            argv=['runner','--source','/synthetic/train24.jsonl','--scaler','/synthetic/scaler.json',
                '--protocol','/synthetic/protocol.json','--checkpoint-receipt','/synthetic/receipt.json',
                '--base','/synthetic/base','--draft','/synthetic/draft','--out',str(out),
                '--expected-gpu-uuid',UUID,'--physical-gpu-index','0','--seed','101','--smoke']
            stack.enter_context(patch.dict(sys.modules,modules))
            stack.enter_context(patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':UUID}))
            stack.enter_context(patch.object(sys,'argv',argv))
            stack.enter_context(patch.object(runner,'validate_pilot_protocol',return_value=protocol))
            stack.enter_context(patch.object(runner,'sha',return_value='0'*64))
            stack.enter_context(patch.object(runner,'load_train24',return_value=['synthetic']*24))
            stack.enter_context(patch.object(runner,'validate_pilot_scaler',return_value=({},{})))
            stack.enter_context(patch.object(runner,'verify_loaded_sources'))
            stack.enter_context(patch.object(runner,'wait_for_preload_gpu',side_effect=wait))
            stack.enter_context(patch.object(runner,'assert_gpu_boundary',side_effect=boundary))
            stack.enter_context(patch('builtins.print'))
            yield SimpleNamespace(events=events,torch=torch,loader=model_loader,model=model,out=out,argv=argv)

    def test_idle_guard_failure_after_long_hash_stops_before_any_cuda_model_or_output(self):
        error=subprocess.CalledProcessError(1,['registered-idle-guard'])
        with self.runtime(gate_error=error) as runtime:
            with self.assertRaises(subprocess.CalledProcessError):runner.main()
            self.assertEqual(runtime.events,['hash','wait'])
            runtime.torch.cuda.device_count.assert_not_called()
            runtime.torch.empty.assert_not_called();runtime.loader.assert_not_called()
            self.assertFalse(runtime.out.exists())

    def test_validate_only_remains_no_wait_cuda_or_output(self):
        with self.runtime() as runtime:
            runtime.argv.append('--validate-only')
            runner.main()
            self.assertEqual(runtime.events,['hash'])
            runtime.torch.empty.assert_not_called();runtime.loader.assert_not_called()
            self.assertFalse(runtime.out.exists())

    def test_load_then_exclusive_boundary_before_fp32_and_before_request_output(self):
        with self.runtime() as runtime:
            with self.assertRaises(StopBeforeRequests):runner.main()
            self.assertEqual(runtime.events,['hash','wait','probe','boundary','load','boundary','float'])
            self.assertFalse(runtime.out.exists())

    def test_foreign_observed_after_model_load_prevents_fp32_and_requests(self):
        with self.runtime(postload_error=True) as runtime:
            with self.assertRaisesRegex(RuntimeError,'foreign appeared during load'):runner.main()
            self.assertEqual(runtime.events,['hash','wait','probe','boundary','load','boundary'])
            runtime.model.base_model.float.assert_not_called()
            self.assertFalse(runtime.out.exists())


if __name__=='__main__':
    unittest.main()
