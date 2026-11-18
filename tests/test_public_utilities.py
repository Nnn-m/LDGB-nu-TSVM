import unittest
import numpy as np
from src.preprocessing import binary_label_mapping, map_labels, scale_train_and_test
from src.metrics import evaluate_binary
from experiments.experiment_protocol import corrupt_training_labels

class PublicUtilitiesTest(unittest.TestCase):
    def test_mapping(self):
        mapping = binary_label_mapping([20, 5, 20, 5])
        self.assertEqual(mapping, {5.0: -1, 20.0: 1})
        np.testing.assert_array_equal(map_labels([20, 5], mapping), [1, -1])

    def test_train_only_scaler(self):
        train, test, _ = scale_train_and_test([[0.0], [2.0]], [[100.0]])
        self.assertAlmostEqual(float(train.mean()), 0.0)
        self.assertGreater(float(test[0, 0]), 10.0)

    def test_f1_variants(self):
        metrics = evaluate_binary([1, 1, -1, -1], [1, -1, -1, -1])
        self.assertAlmostEqual(metrics["ACC_percent"], 75.0)
        self.assertAlmostEqual(metrics["F1_positive"], 2/3)
        self.assertAlmostEqual(metrics["F1_macro"], (2/3 + 4/5)/2)

    def test_noise_only_on_copy(self):
        original = np.array([-1] * 10 + [1] * 10)
        modified = corrupt_training_labels(original, 0.2, random_state=1)
        self.assertEqual(int(np.sum(original != modified)), 4)
        self.assertEqual(int(np.sum(original == 1)), 10)

if __name__ == "__main__":
    unittest.main()
