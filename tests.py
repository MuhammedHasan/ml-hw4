import unittest
import NN_model
import numpy as np
from sklearn.metrics import accuracy_score


class TestNNModel(unittest.TestCase):

    def setUp(self):
        self.M = 120
        self.N = 128
        self.pic = np.ones((self.M, self.N))

    def test_compress_image(self):
        expected_matrix = np.ones((self.M / 2, self.N / 2))
        compressed_matrix = NN_model.compress_image(self.pic)
        self.assertEqual(compressed_matrix.shape, expected_matrix.shape)
        np.testing.assert_array_equal(compressed_matrix, expected_matrix)

    def test__labels_numbers_conversion(self):
        self.assertEqual(NN_model._labels_numbers_conversion(0), 'left')
        self.assertEqual(NN_model._labels_numbers_conversion('left'), 0)

    def test__model_training(self):
        self.assertIsNotNone(NN_model.model)

    def test_model_evaluation(self):
        self.assertIsNotNone(NN_model.model_evaluation(self.pic))

    def test_accurcy(self):
        (matrixs, labels) = zip(*NN_model.map_file_to_matrix())
        predicted_labels = [NN_model.model_evaluation(m) for m in matrixs]
        print('\ntrain accuracy: %s ' %
              accuracy_score(labels, predicted_labels))

        (matrixs, labels) = zip(*NN_model.map_file_to_matrix('TestSet'))
        predicted_labels = [NN_model.model_evaluation(m) for m in matrixs]
        print('\ntest accuracy: %s ' %
              accuracy_score(labels, predicted_labels))


if __name__ == '__main__':
    unittest.main()
