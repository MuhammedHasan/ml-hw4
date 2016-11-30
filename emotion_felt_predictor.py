from face_direction_predictor import FaceDirectionPredictor
import numpy as np


class EmotionFeltPredictor(FaceDirectionPredictor):

    def __init__(self, labels=('neutral', 'sad', 'angry', 'happy')):
        super(self.__class__, self).__init__(labels)
        self.file_label_number = 2

    @staticmethod
    def compress_border(pic):
        x, y = pic.shape
        return pic[x / 4:x * 3 / 4, y / 4:y * 3 / 4]

    @staticmethod
    def compress_image(pic):
        """
        Compress image to half size
        pic: np array of dimensions M x N
        return: np array of dimensions M/2 x N/2
        """
        return np.array([
            pic[m:m + 2, n:n + 2].sum() / 4
            for m in range(0, pic.shape[0], 2)
            for n in range(0, pic.shape[1], 2)
        ]).reshape((pic.shape[0] / 2, pic.shape[1] / 2))

    def _pic_to_features(self, pic):
        '''
        Convert pictures to two time compressed feature array
        pic: np array of dimensions M x N
        return: np array of one dimensions with size M/4 * N/4
        '''
        return self.compress_image(self.compress_border(pic)).reshape(-1)
