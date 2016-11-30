from emotion_felt_predictor import EmotionFeltPredictor

efp = EmotionFeltPredictor()
efp.fit()


# part (d)
def model_evaluation(pic):
    """
    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying emotion the subject is feeling
    """
    return efp.predict(pic)
