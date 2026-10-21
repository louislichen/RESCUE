"""Synthetic stdlib tests of the independent audit, never scientific results."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'exp/draft_graph/audit_baseline_terminal.py'
spec = importlib.util.spec_from_file_location('independent_terminal_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def metadata(root, token_as_dict=False):
    prefix = root/'asset_metadata/Qwen2-7B-Instruct';prefix.mkdir(parents=True)
    content = {'config.json':dict(eos_token_id=999),
        'tokenizer_config.json':dict(eos_token={'content':'SYNTHETIC_EOS'} if token_as_dict else 'SYNTHETIC_EOS',
                                    added_tokens_decoder={'999':{'content':'SYNTHETIC_EOS'}}),
        'tokenizer.json':dict(added_tokens=[dict(content='SYNTHETIC_EOS',id=999)])}
    collection = dict(files={}); protocol = dict(args={'teacher':'/frozen/synthetic/Target'},assets_sha256={})
    for name,value in content.items():
        path=prefix/name;path.write_text(json.dumps(value));sha=audit.digest(path)
        original='/frozen/synthetic/Target/'+name
        collection['files'][str(path.relative_to(root))]=dict(read_only_external_source=original,sha256=sha,bytes=path.stat().st_size)
        protocol['assets_sha256'][original]=sha
    return protocol,collection


class TerminalAuditHelpers(unittest.TestCase):
    def test_independent_fixed_menu_schedule_counts(self):
        arms=audit.menu();owners,requests=audit.planned(arms)
        self.assertEqual((len(arms),len(owners),len(requests)),(38,76,532))
        self.assertEqual(sum(r['phase']=='warm' for r in requests),456)
        tree={a['mode'] for a in arms if a['family']=='full_tree'}
        self.assertEqual(sum(r['mode'] in tree for r in requests),112)
        self.assertEqual((owners,requests),audit.planned(arms))
        for index,owner in enumerate(owners):
            block=requests[index*7:(index+1)*7]
            self.assertEqual(block[0],dict(owner,source_index=271,phase='cold_prime'))
            self.assertEqual({(r['source_index'],r['repeat']) for r in block[1:]},
                             {(i,k) for i in (271,272,288) for k in (0,1)})

    def test_canonical_first_eos_and_short_output_are_strict(self):
        audit.canonical([1]*128,999);audit.canonical([1,999],999)
        for ids in ([1],[],[999,1],[999,999],[True]*128):
            with self.subTest(ids=ids[:3]),self.assertRaises(RuntimeError):audit.canonical(ids,999)

    def test_eos_reconstructed_from_three_pinned_json_files_not_hash_only(self):
        for form in (False,True):
            with self.subTest(dict_form=form),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);protocol,collection=metadata(root,form)
                proof=audit.tokenizer_eos(root,protocol,collection)
                self.assertEqual(proof['eos_token_id'],999)
                self.assertEqual(len(proof['asset_metadata_sha256']),3)
                wrong=copy.deepcopy(protocol);wrong['assets_sha256']['/frozen/synthetic/Target/tokenizer.json']='a'*64
                with self.assertRaisesRegex(RuntimeError,'frozen Target asset'):audit.tokenizer_eos(root,wrong,collection)

    def test_eos_content_id_and_config_mismatch_fail_even_with_rebound_sha(self):
        for fault in ('config','token','decoder'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);protocol,collection=metadata(root)
                name='config.json' if fault=='config' else ('tokenizer.json' if fault=='token' else 'tokenizer_config.json')
                path=root/'asset_metadata/Qwen2-7B-Instruct'/name;value=json.loads(path.read_text())
                if fault=='config':value['eos_token_id']=998
                if fault=='token':value['added_tokens'][0]['content']='different'
                if fault=='decoder':value['added_tokens_decoder']['999']['content']='different'
                path.write_text(json.dumps(value));sha=audit.digest(path)
                protocol['assets_sha256']['/frozen/synthetic/Target/'+name]=sha
                collection['files'][str(path.relative_to(root))]['sha256']=sha
                with self.assertRaises(RuntimeError):audit.tokenizer_eos(root,protocol,collection)

    def test_incomplete_synthetic_root_never_creates_pass_or_source_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);analysis=root/'results/analysis';analysis.mkdir(parents=True)
            (root/'TERMINAL_COLLECTION.json').write_text(json.dumps(dict(status='PARTIAL',counts={})))
            with self.assertRaisesRegex(RuntimeError,'terminal export'):audit.audit(root,analysis)
            self.assertFalse((root/'INDEPENDENT_RAW_AUDIT.json').exists())
            self.assertFalse((root/'INDEPENDENT_RAW_AUDIT_SOURCE.py').exists())

    def test_local_resolver_rejects_external_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'safe').write_text('safe');(root/'link').symlink_to(root/'safe')
            self.assertEqual(audit.local(root,'safe'),(root/'safe').resolve())
            for value in ('../outside','/absolute','link'):
                with self.subTest(value=value),self.assertRaises(RuntimeError):audit.local(root,value)


if __name__=='__main__':unittest.main()
