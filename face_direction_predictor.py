import os
import pickle
from PIL import Image

import numpy as np

from NN_Implementation import NeuralNetMLP


class FaceDirectionPredictor(object):

    def __init__(self, labels=('left', 'straight', 'right',  'up')):
        self.file_label_number = 1
        self.labels = labels
        self.directions = {j: i for i, j in enumerate(self.labels)}
        self._model = object()
        self.l2 = 0.1
        self.l1 = 0.0
        self.epochs = 1000
        self.eta = 0.001
        self.alpha = 0.001
        self.decrease_const = 0.00001
        self.minibatches = 50

    def _labels_to_number(self, label):
        return self.directions[label]

    def _number_to_label(self, number):
        return self.labels[number]

    @staticmethod
    def compress_image(pic):
        """
        Compress image to half size
        pic: np array of dimensions M x N
        return: np array of dimensions M/2 x N/2
        """
        return np.array([
            pic[m:m + 2, n:n + 2].sum() / 4.0
            for m in range(0, pic.shape[0], 2)
            for n in range(0, pic.shape[1], 2)
        ]).reshape((pic.shape[0] / 2, pic.shape[1] / 2))

    def _pic_to_features(self, pic):
        '''
        Convert pictures to two time compressed feature array
        pic: np array of dimensions M x N
        return: np array of one dimensions with size M/4 * N/4
        '''
        return self.compress_image(self.compress_image(pic)).reshape(-1)

    def _read_pics(self, folder="TrainingSet"):
        '''
        Reads all image and labels from given folder
        folder: TrainingSet or TestSet
        return: list of tuple of label and image matrix
        '''
        return [(np.asarray(Image.open(folder + "/" + i).convert('L')),
                 i.split('_')[self.file_label_number])
                for i in os.listdir(folder)]

    def _pic_to_X_y(self, folder="TrainingSet"):
        '''
        Convert traing set pictures to X and y
        return: tuple of X, y
        '''
        return map(np.asarray, zip(*[
            (
                self._pic_to_features(pic),
                self._labels_to_number(label)
            )
            for pic, label in self._read_pics()
        ]))

    def fit(self, version=""):
        '''
        Fits data for traing set
        '''
        pickle_file = 'models/%s%s.p' % (self.__class__.__name__, version)

        self._model = NeuralNetMLP(l2=self.l2, l1=self.l1, epochs=self.epochs,
                                   eta=self.eta, alpha=self.alpha,
                                   decrease_const=self.decrease_const,
                                   minibatches=self.minibatches, shuffle=True,
                                   random_state=1, n_output=4, n_features=960)

        if os.path.isfile(pickle_file):
            self._model = pickle.load(open(pickle_file, 'rb'))
        else:
            (X, y) = self._pic_to_X_y()
            self._model.fit(X, y)
            pickle.dump(self._model, open(pickle_file, 'wb'))

    def predict(self, pic):
        '''
        pic: np array of dimensions 120 x 128 representing an image
        return: String specifying direction that the subject is facing
        '''
        x = self._pic_to_features(pic)
        return self._number_to_label(self._model.predict(np.array([x]))[0])

    def __str__(self):
        return ' '.join([
            str(self.labels),
            "l2:", str(self.l2),
            "l1:", str(self.l1),
            "epochs:", str(self.epochs),
            "eta:", str(self.eta),
            "alpha:", str(self.alpha),
            "decrease_const:", str(self.decrease_const),
            "minibatches:", str(self.minibatches)
        ])
