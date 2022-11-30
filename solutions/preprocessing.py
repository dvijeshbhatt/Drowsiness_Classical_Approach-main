from importlib.metadata import metadata
from operator import ge
from xmlrpc.client import boolean
import cv2
import ffmpeg
import os
import pandas as pd


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def correct_rotation(frame, rC):
    return cv2.rotate(frame, rC)


def checkRotation(file_name):
    print(f"Probing video: {file_name} for rotations")
    #metaData = ffmpeg.probe(file_name)
    metaData=ffmpeg.probe(file_name)

    stream_data = metaData.get('streams', [dict(tags=dict())])

    rC = None
    rotateCode = None

    for i in range(len(stream_data)):
        rotate = stream_data[i].get('tags', dict()).get('rotate', None)
        if rotate:
            rotateCode = int(rotate)

    if rotateCode in [90, -90, 270, -270]:
        rC = 1
        print(f"Found rotation: {rotateCode}")
    if rotateCode == 180:
        rC = None

    return rC


# save the image crop
def save_Crop(image, coords: tuple, w: int, h: int, path: str = "./data/eye/l_eye.jpg", get: boolean = False):
    image = image[coords[1]:coords[1]+h, coords[0]:coords[0]+w]
    # image = cv2.resize(image, (25, 40))

    if not get:
        cv2.imwrite(path, image)

    else:
        return image


# clean FILES
def clean_Folder(name: str, filename: str):

    if filename in os.listdir("./data"):
        os.system(f"rm {name}")
        print(f"REMOVED FILE {name}")

    if name not in os.listdir("./data"):
        df = pd.DataFrame(
            columns=['FN', 'EAR_L', 'EAR_R', 'EAR','EAR_Diff','NewEAR','MAR','MOEAR'])
        df.to_csv(name, index=None)
        print("FILE CREATED.\n\n")
