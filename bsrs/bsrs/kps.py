import numpy as np

from typing import List


COCO_PERSON_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


OPENPOSE_PERSON_KEYPOINT_NAMES = [
    "nose",
    "chest",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
]

FRONT_KEYPOINT_FEATURE_DICT = {
    "head_width": [16, 17],
    "shoulders": [2, 5],
    "y_neck": [0, 1],
}
""" Feature map for subject facing straight on.
"""

LEFT_KEYPOINT_FEATURE_DICT = {
    # "left_arm": [5, 6],
    "left_leg": [11, 12],
    "angle_shoulder_nose": [5, 0],
    "y_neck": [1, 0],
    "eye_ear": [15, 17],
    "nose_ear": [0, 17],
}
""" Feature map for subject facing left.
"""

RIGHT_KEYPOINT_FEATURE_DICT = {
    # "right_arm": [2, 3],
    "right_leg": [8, 9],
    "angle_shoulder_nose": [2, 0],
    "y_neck": [1, 0],
    "neck": [0, 1],
    "eye_ear": [15, 17],
    "nose_ear": [0, 17],
}
""" Feature map for subject facing right.
"""

KEYPOINT_FEATURES = {
    "front": FRONT_KEYPOINT_FEATURE_DICT,
    "left": LEFT_KEYPOINT_FEATURE_DICT,
    "right": RIGHT_KEYPOINT_FEATURE_DICT,
}
""" View with corresponding feature distance map.
"""


def mean_keypoint(keypoints: List, indexes: List) -> List:
    averaging = []
    for index in indexes:
        (x, y, p) = keypoints[index]
        if p:
            averaging += [[x, y]]
    if len(averaging) > 0:
        averaged = np.mean(np.array(averaging), axis=0, dtype=int)
        return list(averaged) + [1]
    else:
        return [0, 0, 0]


def coco_to_openpose(keypoints: List):
    from_order = COCO_PERSON_KEYPOINT_NAMES
    to_order = OPENPOSE_PERSON_KEYPOINT_NAMES
    mapped = [[0, 0, 0]] * len(to_order)
    for i, key in enumerate(to_order):
        if key in from_order:
            mapped[i] = keypoints[from_order.index(key)]
    mapped[to_order.index("chest")] = mean_keypoint(
        keypoints,
        [from_order.index("left_shoulder"), from_order.index("right_shoulder")],
    )
    return mapped
