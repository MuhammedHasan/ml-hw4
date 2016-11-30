import itertools
import logging

from sklearn.metrics import accuracy_score

from face_direction_predictor import FaceDirectionPredictor
from emotion_felt_predictor import EmotionFeltPredictor


LOG_FILENAME = 'logs/optimezer.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)


class Optimezer:

    def __init__(self, predictor_class):
        self.predictor_class = predictor_class

    def small_scale(self):
        return [0] + [0.001 * 3**i for i in range(11)]

    def optimize(self):
        score = 0
        best_model = object()
        p = self.predictor_class()
        for l in itertools.permutations(p.labels):
            for l2 in [0.1, 0.3, 1, 10]:
                for l1 in [0, 0.1, 0.3, 1, 10]:
                    #         for eta in self.small_scale():
                        # for alpha in self.small_scale():
                    p = self.predictor_class(l)
                    p.l2 = l2
                    p.l1 = l1
                    # p.eta = eta
                    # p.alpha = alpha
                    p.fit()
                    t_score = self.accuracy(p)
                    if t_score > score:
                        score = t_score
                        best_model = p
        return best_model

    def accuracy(self, model):
        logging.info(str(model))

        (matrixs, labels) = zip(*model._read_pics())
        predicted_labels = [model.predict(m) for m in matrixs]
        score = accuracy_score(labels, predicted_labels)
        logging.info('train accuracy: %s ' % score)

        (matrixs, labels) = zip(*model._read_pics('TestSet'))
        predicted_labels = [model.predict(m) for m in matrixs]
        score = accuracy_score(labels, predicted_labels)
        logging.info('test accuracy: %s ' % score)
        return score


if __name__ == "__main__":
    o = Optimezer(EmotionFeltPredictor)
    print(o.optimize())
