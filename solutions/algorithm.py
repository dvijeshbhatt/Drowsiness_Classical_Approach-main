jj# import the necessary packages
from skimage import feature
import numpy as np
import cv2
import pickle
import warnings
warnings.filterwarnings("ignore")


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist


# load model and make prediction
def get_Predictions(image: cv2.Mat, model_path: str = "./models/svm_blink.pkl") -> int:
    # load saved model
    with open('./models/svm_blink.pkl', 'rb') as f:
        model = pickle.load(f)

    # image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # local binary patterns
    lbp = LocalBinaryPatterns(24, 8)
    arr = lbp.describe(image_gray)

    prediction = model.predict(arr.reshape(1, -1))

    return prediction[0]


# load the svm model and pass EARs of 13 frames for predictions
def get_Blink_Prediction(features:list, model_path:str = "./models/svm_ear_blinks.pkl") -> str:
    pass