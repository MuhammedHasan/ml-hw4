import os
from PIL import Image

import numpy as np

from NN_Implementation import NeuralNetMLP


# part (a)
def compress_image(pic):
    """
    :param pic: np array of dimensions M x N
    :return: np array of dimensions M/2 x N/2 using the formula
             specified in the assignment description
    """
    return np.array([
        pic[m:m + 2, n:n + 2].sum() / 4.0
        for m in range(0, pic.shape[0], 2)
        for n in range(0, pic.shape[1], 2)
    ]).reshape((pic.shape[0] / 2, pic.shape[1] / 2))


def _map_img_to_x_array(pic):
    return compress_image(compress_image(pic)).reshape(-1)


def map_file_to_matrix(folder="TrainingSet"):
    return [(np.asarray(Image.open(folder + "/" + i).convert('L')),
             i.split('_')[1]) for i in os.listdir(folder)]


def pic_matrix_to_X_y():
    return map(np.asarray, zip(*[
        (
            _map_img_to_x_array(pic),
            _labels_numbers_conversion(label)
        )
        for pic, label in map_file_to_matrix()
    ]))


def _model_training():
    (X, y) = pic_matrix_to_X_y()
    nnm = NeuralNetMLP(l2=0.1, l1=0.0, epochs=1000, eta=0.001, alpha=0.001,
                       decrease_const=0.00001, minibatches=50,
                       shuffle=True, random_state=1,
                       n_output=4, n_features=960)
    nnm.fit(X, y, print_progress=True)
    return nnm


def _labels_numbers_conversion(label_or_number):
    labels = ['left', 'straight', 'right',  'up']
    # labels = ('up', 'straight', 'left', 'right')

    directions = {j: i for i, j in enumerate(labels)}
    if type(label_or_number) == str:
        return directions[label_or_number]
    return labels[label_or_number]


model = _model_training()


# part (b)
def model_evaluation(pic):
    """
    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying direction that the subject is facing
    """
    x = _map_img_to_x_array(pic)
    return _labels_numbers_conversion(model.predict(np.array([x]))[0])
