"""Small sanity checks: run with python -m unittest discover -s tests."""
import unittest

from main import build_models, load_data


class ModelSetupTests(unittest.TestCase):
    def test_dataset_shape_and_labels(self):
        X, y = load_data()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(int(y.sum()), 212)

    def test_all_required_models_exist(self):
        names = set(build_models().keys())
        self.assertEqual(names, {
            "Logistic Regression", "Decision Tree", "KNN",
            "Voting (Hard)", "Stacking", "AdaBoost",
        })


if __name__ == "__main__":
    unittest.main()
