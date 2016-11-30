from face_direction_predictor import FaceDirectionPredictor

fdp = FaceDirectionPredictor()
fdp.minibatches = 960
fdp.fit(version="1")


# part (c)
def model_evaluation(pic):
    """
    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying direction that the subject is facing
    """
    return fdp.predict(pic)
