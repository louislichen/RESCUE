import unittest
from csgb_cache_geometry import root_readable_positions


class GeometryTests(unittest.TestCase):
    def test_new_committed_suffix_is_readable(self):
        self.assertEqual(root_readable_positions([(0,0,2),(2,4,6)],6,9,10),
                         [0,1,4,5,6,7,8])

    def test_next_uncommitted_root_is_excluded(self):
        self.assertEqual(root_readable_positions([(0,0,4)],4,4,5),[0,1,2,3])

    def test_stale_attention_does_not_fabricate_new_mass(self):
        self.assertEqual(root_readable_positions([(0,0,4),(2,8,10)],10,7,13),[0,1,2,3])

    def test_deduplicates_and_rejects_wrong_positions(self):
        self.assertEqual(root_readable_positions([(0,0,3),(1,2,4)],4,4,5),[0,1,2,3])
        for args in [([(0,0,6)],4,4,5), ([],6,5,5), ([],4,5,5)]:
            with self.assertRaises(ValueError): root_readable_positions(*args)
