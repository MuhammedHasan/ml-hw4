from face_direction_predictor import FaceDirectionPredictor


# part (a)
def compress_image(pic):
    """
    :param pic: np array of dimensions M x N
    :return: np array of dimensions M/2 x N/2 using the formula
             specified in the assignment description
    """
    return FaceDirectionPredictor.compress_image(pic)

ip = FaceDirectionPredictor()
ip.fit()


# part (b)
def model_evaluation(pic):
    """
    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying direction that the subject is facing
    """
    return ip.predict(pic)
