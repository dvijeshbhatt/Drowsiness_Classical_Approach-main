import numpy as np
import cv2

# get coordinates of the particular face landmark point
def get_arr_from_coordinates(landmarks, point) -> np.array:
    val = (landmarks[point].x, landmarks[point].y)
    val = np.array(val)

    return val


# caclulate the EAR for a person
def get_Aspect_Ratios(person1_landmarks, feature_name: str = "EAR") -> float:

    if feature_name == "EAR":
        # FOR EAR (EYE ASPECT RATIO)
        # Note: Refer "./data/EAR_ref.png" for the postion of these points

        # Left eye
        p1, p2 = get_arr_from_coordinates(
            person1_landmarks, 362), get_arr_from_coordinates(person1_landmarks, 385)
        p3, p4 = get_arr_from_coordinates(
            person1_landmarks, 387), get_arr_from_coordinates(person1_landmarks, 263)
        p5, p6 = get_arr_from_coordinates(
            person1_landmarks, 373), get_arr_from_coordinates(person1_landmarks, 380)

        # Right eye
        p1_r, p2_r = get_arr_from_coordinates(
            person1_landmarks, 133), get_arr_from_coordinates(person1_landmarks, 158)
        p3_r, p4_r = get_arr_from_coordinates(
            person1_landmarks, 160), get_arr_from_coordinates(person1_landmarks, 33)
        p5_r, p6_r = get_arr_from_coordinates(
            person1_landmarks, 144), get_arr_from_coordinates(person1_landmarks, 153)

        # Formula for calculating EAR

        # left
        EAR_L = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / \
            (2 * np.linalg.norm(p1 - p4))

        # right
        EAR_R = (np.linalg.norm(p2_r - p6_r) + np.linalg.norm(p3_r - p5_r)) / \
            (2 * np.linalg.norm(p1_r - p4_r))

        return EAR_L, EAR_R, (EAR_L + EAR_R) / 2,

    elif feature_name == "MAR":

        p_t, p_b = get_arr_from_coordinates(
            person1_landmarks, 0), get_arr_from_coordinates(person1_landmarks, 17)
        p_r, p_l = get_arr_from_coordinates(
            person1_landmarks, 62), get_arr_from_coordinates(person1_landmarks, 91)

        MAR = (np.linalg.norm(p_t - p_b)) / (np.linalg.norm(p_r - p_l))

        return MAR

    return None


# get the dist b/w eyelid top and the iris center or bottom
def get_Dist_Iris_Eyelid(person1_landmarks) -> float:
    iris_left_eye = get_arr_from_coordinates(person1_landmarks, 475)
    iris_right_eye = get_arr_from_coordinates(person1_landmarks, 470)

    eyelid_top_left = get_arr_from_coordinates(person1_landmarks, 386)
    eyelid_bottom_left = get_arr_from_coordinates(person1_landmarks, 374)

    eyelid_top_right = get_arr_from_coordinates(person1_landmarks, 159)
    eyelid_bottom_right = get_arr_from_coordinates(person1_landmarks, 145)

    # Distance b/w iris left eye and eyelid top point
    TD_R = np.linalg.norm(eyelid_top_right - eyelid_bottom_right)
    ID_R = np.linalg.norm(iris_right_eye - eyelid_bottom_right)

    TD_L = np.linalg.norm(eyelid_top_left - eyelid_bottom_left)
    ID_L = np.linalg.norm(iris_left_eye - eyelid_bottom_left)

    # thresh = 1 if ((TD_R + TD_L) / 2) < ((ID_R + ID_L) / 2) else 0
    thresh = ID_R / TD_R

    return thresh


# Get the crops of the left eye
def get_Crops_L_Eye(landmarks, width, height):
    x_lt, y_lt = int(get_arr_from_coordinates(landmarks, 464)[
                     0] * height), int(get_arr_from_coordinates(landmarks, 441)[1] * width)
    x_rb, y_rb = int(get_arr_from_coordinates(landmarks, 359)[
                     0] * height), int(get_arr_from_coordinates(landmarks, 450)[1] * width)

    w = x_rb - x_lt
    h = y_rb - y_lt

    return (x_lt, y_lt), w, h


# get normalized cross correlation
def get_NCC(image_1, template_image_path) -> float:
    try:
        t_image = cv2.imread(template_image_path)

        result = {}

        result["NCC"] = cv2.matchTemplate(
            image_1, t_image, cv2.TM_CCORR_NORMED)

        # result["NCC"] = ncc(t_image, image_1)

    except Exception as e:
        result = None

    return result


# def Normalised_Cross_Correlation(roi, target):
#     # Normalised Cross Correlation Equation
#     cor = np.sum(roi * target)
#     nor = np.sqrt((np.sum(roi ** 2))) * np.sqrt(np.sum(target ** 2))

#     return cor / nor

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    # return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


# Note: Get the minute number going on
def get_Minute(frame_count: int) -> int:
    if frame_count in range(0, 1801):
        return 1
    elif frame_count in range(1801, 3601):
        return 2
    elif frame_count in range(3601, 5401):
        return 3
    elif frame_count in range(5401, 7201):
        return 4
    elif frame_count in range(7201, 9001):
        return 5
    elif frame_count in range(9001, 10801):
        return 6
    elif frame_count in range(10801, 12601):
        return 7
    elif frame_count in range(12601, 14401):
        return 8
    elif frame_count in range(14401, 16201):
        return 9
    elif frame_count in range(16201, 18050):
        return 10
