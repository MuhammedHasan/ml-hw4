import unittest
import logging

import numpy as np
from sklearn.metrics import accuracy_score

import NN_model
import face_direction_predictor as fdp
import emotion_felt_predictor as efp

LOG_FILENAME = 'logs/face-direction-predictor.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)


# class TestNNModel(unittest.TestCase):
#
#     def setUp(self):
#         self.M = 120
#         self.N = 128
#         self.pic = np.ones((self.M, self.N))
#
#     def test_compress_image(self):
#         expected_matrix = np.ones((self.M / 2, self.N / 2))
#         compressed_matrix = NN_model.compress_image(self.pic)
#         self.assertEqual(compressed_matrix.shape, expected_matrix.shape)
#         np.testing.assert_array_equal(compressed_matrix, expected_matrix)
#
#     def test__model_training(self):
#         self.assertIsNotNone(NN_model.ip)
#
#     def test_model_evaluation(self):
#         self.assertIsNotNone(NN_model.model_evaluation(self.pic))
#
#
# class TestFaceDirectionPredictor(unittest.TestCase):
#
#     def setUp(self):
#         self.M = 120
#         self.N = 128
#         self.pic = np.ones((self.M, self.N))
#         self.predictor = fdp.FaceDirectionPredictor()
#
#     def test_compress_image(self):
#         expected_matrix = np.ones((self.M / 2, self.N / 2))
#         compressed_matrix = self.predictor.compress_image(self.pic)
#         self.assertEqual(compressed_matrix.shape, expected_matrix.shape)
#         np.testing.assert_array_equal(compressed_matrix, expected_matrix)
#
#     def test__number_to_label(self):
#         self.assertEqual(self.predictor._number_to_label(0), 'left')
#
#     def test__labels_to_number(self):
#         self.assertEqual(self.predictor._labels_to_number('left'), 0)
#
#     def test__model_training(self):
#         self.predictor.fit()
#         self.assertIsNotNone(self.predictor._model)
#
#     def test_accurcy(self):
#         self.predictor.fit()
#         (matrixs, labels) = zip(*self.predictor._read_pics())
#         predicted_labels = [self.predictor.predict(m) for m in matrixs]
#         logging.info('train accuracy: %s ' %
#                      accuracy_score(labels, predicted_labels))
#
#         (matrixs, labels) = zip(*self.predictor._read_pics('TestSet'))
#         predicted_labels = [self.predictor.predict(m) for m in matrixs]
#         logging.info('test accuracy: %s ' %
#                      accuracy_score(labels, predicted_labels))

class TestFaceDirectionPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = efp.EmotionFeltPredictor()

    def test_accurcy(self):
        self.predictor.fit()
        (matrixs, labels) = zip(*self.predictor._read_pics())
        predicted_labels = [self.predictor.predict(m) for m in matrixs]
        logging.info('train accuracy: %s ' %
                     accuracy_score(labels, predicted_labels))

        (matrixs, labels) = zip(*self.predictor._read_pics('TestSet'))
        predicted_labels = [self.predictor.predict(m) for m in matrixs]
        logging.info('test accuracy: %s ' %
                     accuracy_score(labels, predicted_labels))

if __name__ == '__main__':
    unittest.main()
