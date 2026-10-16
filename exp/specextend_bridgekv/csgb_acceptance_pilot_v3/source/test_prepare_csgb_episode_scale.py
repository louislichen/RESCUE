import copy
import unittest
from prepare_csgb_episode_scale import population_scale


class ScaleTests(unittest.TestCase):
    def rows(self):
        return [(d,[1.,.5+.01*d,.01+.001*d,(.5+.01*d)*(.01+.001*d)]) for d in range(16)]

    def test_population_and_counts(self):
        result=population_scale(self.rows())
        self.assertEqual(result['count'],16)
        self.assertAlmostEqual(result['mean'][0],.575)
        self.assertTrue(all(v>0 for v in result['scale']))

    def test_monitor_rejected(self):
        with self.assertRaises(ValueError): population_scale(self.rows()+[(16,[1.,.6,.02,.012])])

    def test_missing_fit_document_rejected(self):
        with self.assertRaises(ValueError): population_scale(self.rows()[:-1])

    def test_raw_product_rejected(self):
        rows=copy.deepcopy(self.rows());rows[0][1][3]=0
        with self.assertRaises(ValueError): population_scale(rows)

    def test_degenerate_scale_rejected(self):
        with self.assertRaises(ValueError): population_scale([(d,[1.,.5,.1,.05]) for d in range(16)])


if __name__=='__main__': unittest.main()
