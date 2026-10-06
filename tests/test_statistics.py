import unittest

from src.evaluation.statistics import bootstrap_mean_difference, paired_permutation_pvalue


class TestStatistics(unittest.TestCase):
    def test_bootstrap_difference(self):
        result = bootstrap_mean_difference([1, 1, 1, 1], [0, 0, 0, 0], n_bootstrap=100, seed=1)
        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["ci_low"], 1.0)
        self.assertEqual(result["ci_high"], 1.0)

    def test_permutation_identical_samples(self):
        p = paired_permutation_pvalue([1, 0, 1], [1, 0, 1], n_permutations=100, seed=1)
        self.assertEqual(p, 1.0)

    def test_paired_length_validation(self):
        with self.assertRaises(ValueError):
            bootstrap_mean_difference([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
