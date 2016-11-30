from face_direction_predictor import FaceDirectionPredictor


class EmotionFeltPredictor(FaceDirectionPredictor):

    def __init__(self, labels=('neutral', 'sad', 'angry', 'happy')):
        super(self.__class__, self).__init__(labels)
        self.file_label_number = 2
        # self.l1 = 1
        # self.l2 = 1
        self.minibatches = 315
