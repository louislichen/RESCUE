from types import SimpleNamespace
import unittest
from csgb_episode_gpu_guard import observe

UUID='GPU-c7642b47-f924-b4b1-3e90-ac94a72f1830'


class GuardTests(unittest.TestCase):
    def query(self,*,apps='',memory=4,util=0,stderr='',uuid=UUID,total=49140):
        def run(args,**kwargs):
            output=apps if '--query-compute-apps=gpu_uuid,pid' in args else f'{uuid}, NVIDIA GeForce RTX 4090, {total}, {memory}, {util}\n'
            return SimpleNamespace(stdout=output,stderr=stderr)
        return run

    def test_empty_idle(self): self.assertTrue(observe(4,UUID,self.query())['idle'])
    def test_foreign_job_not_idle(self):
        self.assertFalse(observe(4,UUID,self.query(apps=f'{UUID}, 123\n'))['idle'])
    def test_unaccounted_memory_or_util_not_idle(self):
        self.assertFalse(observe(4,UUID,self.query(memory=1000))['idle'])
        self.assertFalse(observe(4,UUID,self.query(util=1))['idle'])
    def test_warning_or_bad_inventory_is_not_idle(self):
        for query in (self.query(stderr='warning'),self.query(apps='unknown'),self.query(total=24000)):
            with self.assertRaises(RuntimeError): observe(4,UUID,query)


if __name__=='__main__': unittest.main()
