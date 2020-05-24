import os
import errno
import cv2 as cv
import numpy as np
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
import torchvision.models.detection as detection

from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import OrderedDict
from torchvision import transforms


def save_fig(fig: plt.Figure, name: str):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    figure_dir = os.path.join(cur_dir, "report", "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, name + ".pgf"))
    fig.savefig(os.path.join(figure_dir, name + ".pdf"))


def draw_keypoints(image, keypoints, radius=1, alpha=1.0):
    overlay = image.copy()
    for kp in keypoints:
        for x, y, v in kp:
            if int(v):
                cv.circle(overlay, (int(x), int(y)), radius, (0, 255, 0), -1, cv.LINE_AA)
    return cv.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_body_connections(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    b_conn = [(0, 1), (1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]
    h_conn = [(0, 14), (0, 15), (14, 16), (15, 17)]
    l_conn = [(5, 6), (6, 7), (11, 12), (12, 13)]
    r_conn = [(2, 3), (3, 4), (8, 9), (9, 10)]
    for kp in keypoints:
        for i, j in b_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in h_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in l_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 255, 0), thickness)
        for i, j in r_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 0, 255), thickness)
    return cv.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

def _draw_connection(image, point1, point2, color, thickness=1):
    x1, y1, v1 = point1
    x2, y2, v2 = point2
    if v1 and v2:
        cv.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv.LINE_AA)
    return image


class StateError(RuntimeError):
    """ Error raised if something is not in a valid state.
    """


def resize_image(
    image, width: int = None, height: int = None, inter: int = cv.INTER_AREA,
):
    """Utility method to resize an image where not all dimensions must be provided.
    If only one of width or height is provided the one not provided will be calculated
    as to keep the same aspect ratio. If both are provided the image will be resized
    as normal.

    Args:
        image (Array[Number]): Input image to resize, this can have any
            number of channels, i.e. it can be mono or bgr.
        width (int, optional): The fixed width to resize to. Defaults to None.
        height (int, optional): The fixed height to resize to. Defaults to None.
        inter (int, optional): The resizing interpolation algorithm to use.
            Defaults to cv.INTER_AREA.

    Returns:
        Array[Number]: Resized image in respect to the input arguments.
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


from typing import List, Dict, Tuple


def mean_keys(dic: Dict, keys: List[str]):
    values = []
    for key in keys:
        value = dic.get(key, None)
        if value is not None:
            values += [value]
    mean_value = np.mean(values)
    return mean_value


FRONT_KEYPOINT_FEATURE_DICT = {
    "head_width": [16, 17],
    "shoulders": [2, 5],
    "neck": [0, 1],
}

LEFT_KEYPOINT_FEATURE_DICT = {
    # "left_arm": [5, 6],
    "left_leg": [11, 12],
    "angle_shoulder_nose": [5, 0],
    "neck": [0, 1],
    "eye_ear": [15, 17],
    "nose_ear": [0, 17],
}

RIGHT_KEYPOINT_FEATURE_DICT = {
    "right_arm": [2, 3],
    "right_leg": [8, 9],
    "right_body": [2, 8],
    "y_neck": [0, 1],
}

COMPONENTS = len(LEFT_KEYPOINT_FEATURE_DICT) + 1


KEYPOINT_FEATURES = {
    "front": FRONT_KEYPOINT_FEATURE_DICT,
    "left": LEFT_KEYPOINT_FEATURE_DICT,
    "right": RIGHT_KEYPOINT_FEATURE_DICT,
}

COCO_SEGMENT_NAMES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

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


# https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
# https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
img_mapping = {
    "DSC00165.JPG": "021z001ps.jpg",
    "DSC00166.JPG": "021z001pf.jpg",
    "DSC00167.JPG": "021z002ps.jpg",
    "DSC00168.JPG": "021z002pf.jpg",
    "DSC00169.JPG": "021z003ps.jpg",
    "DSC00170.JPG": "021z003pf.jpg",
    "DSC00171.JPG": "021z004ps.jpg",
    "DSC00172.JPG": "021z004pf.jpg",
    "DSC00173.JPG": "021z005ps.jpg",
    "DSC00174.JPG": "021z005pf.jpg",
    "DSC00175.JPG": "021z006ps.jpg",
    "DSC00176.JPG": "021z006pf.jpg",
    "DSC00177.JPG": "021z007ps.jpg",
    "DSC00178.JPG": "021z007pf.jpg",
    "DSC00179.JPG": "021z008ps.jpg",
    "DSC00180.JPG": "021z008pf.jpg",
    "DSC00181.JPG": "021z009ps.jpg",
    "DSC00182.JPG": "021z009pf.jpg",
    "DSC00183.JPG": "021z010ps.jpg",
    "DSC00184.JPG": "021z010pf.jpg",
    "DSC00185.JPG": "024z011pf.jpg",
    "DSC00186.JPG": "024z011ps.jpg",
}


def find_side(data, threshold: int, required: int, from_right: bool = False) -> int:
    """Find the point where data exceeds the threshold for the required number of
    values in a row. By default will search from the left.

    Args:
        data (Array[int]): Summed pixel values across an axis of an image.
        threshold (int, optional): The summed pixel value along an axis before
            it is deemed to be part of the view.
        required (int, optional): The number of columns/rows that exceed the
            threshold in a row before a view side is accepted. Defaults to 20.
        from_right (bool, optional): Whether to search from right to left.
            Defaults to False.

    Returns:
        int: Point where threshold was first exceeded. None if no side found.
    """
    ongoing = 0
    start = 0
    for idx in range(len(data)):
        if from_right:
            point = len(data) - idx - 1
        else:
            point = idx
        value = data[point]
        if value > threshold:
            if ongoing == 0:
                start = point
            ongoing += 1
        else:
            ongoing = 0
        if ongoing >= required:
            return start
    return None


# DEEPLAB_RESNET101 = segmentation.deeplabv3_resnet101(pretrained=True)


class BodyShapeFE:
    def __init__(
        self,
        segmenter: nn.Module = segmentation.deeplabv3_resnet101(pretrained=True),
        keypoint_estimator: nn.Module = detection.keypointrcnn_resnet50_fpn(
            pretrained=True
        ),
        input_height: int = 600,
    ):
        self.segmenter = segmenter
        self.keypoint_estimator = keypoint_estimator
        self.input_height = input_height

        self.cache = {}
        self.cache["keypoints"] = {}
        self.cache["masks"] = {}
        self.cache["images"] = {}
        # Move to GPUs if available
        # BodyPoseEstimator handles this for itself
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.segmenter:
            segmenter.to(self.device)
        self.segmenter.eval()
        if self.keypoint_estimator:
            keypoint_estimator.to(self.device)
        self.keypoint_estimator.eval()

    def height(self, path: str):
        mask = self.mask(path)
        width_calc = np.sum(mask != 0, axis=1)
        y0 = find_side(width_calc, 1, 20)
        y1 = find_side(width_calc, 1, 20, from_right=True)
        return y1 - y0

    def head_mask(self, path: str):
        mask = np.zeros_like(self.mask(path))
        contour = self.contour(path)
        neck_left, neck_right = self.neck_keypoints(path)
        left_index = np.linalg.norm(contour - neck_left, axis=1).argmin()
        right_index = np.linalg.norm(contour - neck_right, axis=1).argmin()
        contour_mask = np.ones(contour.shape[0], dtype=bool)
        contour_mask[left_index + 1 : right_index] = 0
        contour = contour[contour_mask, :]
        cv.drawContours(mask, [contour], -1, (255), -1)
        return mask

    def process(self, path: str, plot: bool = False):
        keypoints = self.keypoints(path)
        direction = self.direction(path)
        # keypoints = np.concatenate(
        #     (keypoints, [(x, y, 1) for (x, y) in self.neck_keypoints(path)])
        # )
        dist_dict = self.dist_dict(path)
        self.height(path)
        self.head_mask(path)
        # print(keypoints.shape)
        # scaled_dist_dict = BodyPoseClassifier.scale_dist_dict(dist_dict)

        # Collate into data dictionary for outputs
        data = {}
        data["path"] = path
        data["keypoints"] = keypoints
        data["direction"] = direction
        data["dist_dict"] = dist_dict
        data["feature"] = self.feature(path)

        self.cache["images"].pop(path, None)
        return data

    def feature(self, path: str):
        feature = []
        feature += list(self.dist_dict(path).values())
        feature += [self.height(path)]
        # Add hu moments (only for side views)
        if self.direction(path) != "front":
            moments = cv.moments(self.mask(path))
            hu_moments = cv.HuMoments(moments).flatten()
            feature += list(hu_moments)
        # Add head hu moments
        moments = cv.moments(self.head_mask(path))
        hu_moments = cv.HuMoments(moments).flatten()

        feature += list(hu_moments)

        return np.array(feature, dtype=np.float64)

    def keypoints(self, path: str):
        if path not in self.cache["keypoints"]:
            image = self.image(path)
            image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            preprocess = transforms.ToTensor()
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            input_batch.to(self.device)
            with torch.no_grad():
                output = self.keypoint_estimator(input_batch)
                keypoints = output[0]["keypoints"]
            if keypoints.shape[0] == 0:
                RuntimeWarning("No person detected.")
                return None
            elif keypoints.shape[0] > 1:
                RuntimeWarning("Multiple people detected, using first person.")
            self.cache["keypoints"][path] = keypoints[0].cpu().numpy()

        keypoints = self.cache["keypoints"][path]
        keypoints = coco_to_openpose(keypoints)

        return keypoints

    def mask(self, path: str):
        if path not in self.cache["masks"]:
            # Semantic segmentation mask
            predictions = self.segment(path)
            mask = predictions == 15  # person
            mask = np.array(mask, dtype=np.uint8)

            # Green screen mask
            image = self.image(path)
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
            g_mask = np.logical_and(h > 53, h < 130)
            g_mask = np.logical_and(g_mask, s >= 102)
            g_mask = np.logical_and(g_mask, v >= 120)
            mask[g_mask] = 0
            self.cache["masks"][path] = mask * 255
        mask = self.cache["masks"][path].copy()
        mask[:, list(range(250))] = 0
        return mask

    def image(self, path: str):
        if path not in self.cache["images"]:
            image = cv.imread(path)
            if image is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
            self.cache["images"][path] = resize_image(image, height=self.input_height)
        return self.cache["images"][path]

    def segment(self, path: str):
        image = self.image(path)
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch.to(self.device)
        with torch.no_grad():
            output = self.segmenter(input_batch)["out"][0]
        predictions = output.argmax(0).byte().cpu().numpy()
        return predictions

    def dist_dict(self, path: str) -> Dict:
        dist_dict = OrderedDict()

        keypoints = self.keypoints(path)
        direction = self.direction(path)

        for item, keys in KEYPOINT_FEATURES[direction].items():
            length = 0
            for i1, i2 in zip(keys, keys[1:]):
                (x1, y1, p1), (x2, y2, p2) = keypoints[i1], keypoints[i2]
                if p1 == 0 or p2 == 0:
                    dist_dict[item] = None
                    break
                if item.startswith("angle_"):
                    length += np.arctan2(y2 - y1, x2 - x1) * 180 / math.pi
                elif item.startswith("y_"):
                    length += abs(y1 - y2)
                elif item.startswith("x_"):
                    length += abs(x1 - x2)
                else:
                    length += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            dist_dict[item] = length
        return dist_dict

    def contour(self, path: str):
        mask = self.mask(path)
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]
        contour = np.squeeze(contour)
        return contour

    def neck_keypoints(self, path: str, plot: bool = False):
        keypoints = self.keypoints(path)
        contour = self.contour(path)

        # if any(map(lambda x: x in path, ["018z061ps"])):
        #     plot = True
        direction = self.direction(path)
        if direction in ["left", "right"]:
            left_mask = contour[:, 0] <= keypoints[1][0]
            right_mask = contour[:, 0] > keypoints[1][0]
            kpl, kpr = 1, 1
            ax = 1
        elif direction in ["front"]:
            left_mask = contour[:, 1] <= keypoints[2][1]
            right_mask = contour[:, 1] <= keypoints[5][1]
            kpl, kpr = 2, 5
            ax = 0
        else:
            raise NotImplementedError("Direction not supported")
        # Start at the closest Y position on each side
        # Use flipped array for left points so can move up by increasing index
        index_left = np.argmin(np.abs(contour[left_mask, ax] - keypoints[kpl][ax])) - 1
        index_left = np.flip(np.arange(contour.shape[0]))[left_mask][index_left]
        points_left = np.flip(contour, axis=0)
        # ---
        index_right = np.argmin(np.abs(contour[right_mask, ax] - keypoints[kpr][ax]))
        index_right = np.arange(contour.shape[0])[right_mask][index_right]
        points_right = contour
        # Nudge index back in case on turning point
        index_left -= 10
        index_right -= 10
        # Find curving point of neck
        peaks_left = scipy.signal.find_peaks(points_left[:, 0])[0]
        # Negate X dimension of right points so can find peak instead of trough
        peaks_right = scipy.signal.find_peaks(-points_right[:, 0])[0]
        finish_index_left, finish_index_right = -1, -1
        # Sorting function to handle wrapping of peaks
        if len(peaks_left) > 0:
            key = lambda x: x if x > index_left else x + len(points_left)  # noqa
            finish_index_left = sorted(peaks_left, key=key)[0]
        if len(peaks_right) > 0:
            key = lambda x: x if x > index_right else x + len(points_right)  # noqa
            finish_index_right = sorted(peaks_right, key=key)[0]

        if plot:
            plt.figure("Neck Left")
            plt.plot(points_left[:, 0])
            plt.scatter(peaks_left, points_left[peaks_left, 0], color="r")
            plt.scatter(index_left, points_left[index_left, 0], color="b")
            plt.scatter(finish_index_left, points_left[finish_index_left, 0], color="g")
            plt.figure("Neck Right")
            plt.plot(points_right[:, 0])
            plt.scatter(index_right, points_right[index_right, 0], color="b")
            plt.scatter(peaks_right, points_right[peaks_right, 0], color="r")
            plt.scatter(
                finish_index_right, points_right[finish_index_right, 0], color="g"
            )

        # Got finish point
        finish_point_left = points_left[finish_index_left]
        finish_point_right = points_right[finish_index_right]

        # If front-on use the max y coordinate (bottom)
        # if self.direction(path) == "front":
        #     finish_point_left[1] = max(finish_point_left[1], finish_point_right[1])
        #     finish_point_right[1] = max(finish_point_left[1], finish_point_right[1])

        if plot:
            image = self.image(path).copy()
            max_x, max_y = image.shape[1], image.shape[0]
            contour = np.array(
                [
                    finish_point_left,
                    finish_point_right,
                    (max_x, finish_point_right[1]),
                    (max_x, max_y),
                    (0, max_y),
                    (0, finish_point_left[1]),
                ]
            )
            cv.drawContours(image, [contour], -1, (0, 0, 255), 1)
            cv.circle(
                image, tuple(points_left[index_left]), 3, (0, 0, 255), -1, cv.LINE_AA
            )
            cv.circle(
                image, tuple(points_right[index_right]), 3, (0, 0, 255), -1, cv.LINE_AA
            )
            cv.circle(image, tuple(finish_point_left), 3, (0, 255, 0), -1, cv.LINE_AA)
            cv.circle(image, tuple(finish_point_right), 3, (0, 255, 0), -1, cv.LINE_AA)
            plt.figure()
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            plt.show()

        return (finish_point_left, finish_point_right)

    @staticmethod
    def scale_dist_dict(dist_dict: Dict, keys: List):
        body_length_keys = ["left_body", "right_body"]
        body_length_keys = ["shoulders"]
        body_length = mean_keys(dist_dict, body_length_keys)
        scaled_dist_dict = OrderedDict()
        for item in dist_dict:
            scaled_dist_dict[item] = dist_dict[item] / body_length
        return scaled_dist_dict

    def filter_dist_dict(self):
        pass

    def direction(self, path: str) -> str:
        keypoints = self.keypoints(path)
        # If left and right shoulder points found and far apart then front on
        # otherwise direction between neck and nose dictates side
        if (not keypoints[2][2] or not keypoints[5][2]) or (
            np.linalg.norm(keypoints[2][:2] - keypoints[5][:2]) < 50
        ):
            if keypoints[1][0] - keypoints[0][0] > 0:
                return "left"
            else:
                return "right"
        else:
            return "front"


class BodyShapeClassifier:
    def __init__(self, feature_extractor: BodyShapeFE):
        """Classifier that uses body shape features. Handles training and evaluation.
        Note that front and side poses will train using separate features.

        Args:
            feature_extractor (BodyShapeFE): Feature extractor to use for body shape information.
        """
        self.fe = feature_extractor
        self.data = {}
        # Classifier dictionary holding lists of features prior to preprocessing
        self.feature_lists: Dict[str, List[np.array]] = {}
        # Classifier dictionary holding Numpy array of preprocessed features
        self.features: Dict[str, np.array] = {}
        # Classifier dictionary holding class identifiers
        self.classes: Dict[str, List[str]] = {}
        # Classifier dictionary holding array of preprocessed features w
        self.pipelines: Dict[str, Pipeline] = {}
        self.reset()

    def reset(self):
        for classifiers in self.classifiers:
            self.feature_lists[classifiers] = []
            self.classes[classifiers] = []
            self.features[classifiers] = None
            self.pipelines[classifiers] = None

    @property
    def classifiers(self) -> List[str]:
        return ["left", "right", "front"]

    def add_features(self, paths: List[str], prepare: bool = True):
        for i, path in enumerate(paths):
            self.add_feature(path, prepare=False)
        if prepare:
            self.prepare_features()

    def add_feature(self, path: str, prepare: bool = True):
        data = self.fe.process(path)
        classifier = data["direction"]
        data["class"] = self.get_class(path)
        self.data[path] = data
        self.feature_lists[classifier] += [data["feature"]]
        self.classes[classifier] += [data["class"]]
        if prepare:
            self.prepare_features()

    def get_class(self, path: str) -> str:
        # name without direction/extension
        return os.path.basename(path)[:-5]

    def infer(self, path: str):
        data = self.fe.process(path)
        self.data[path] = data
        classifier = data["direction"]
        feature = self.pipelines[classifier].transform(
            data["feature"][np.newaxis, ...]
        )[0]
        data["feature_dists"] = np.array(
            [
                np.linalg.norm(feature - ref_feature) / len(ref_feature)
                for ref_feature in self.features[classifier]
            ]
        )
        data["est_feature_index"] = np.argmin(data["feature_dists"])
        data["est_class"] = self.classes[classifier][data["est_feature_index"]]
        return data

    def prepare_features(self):
        for classifier in self.classifiers:
            if len(self.feature_lists[classifier]) == 0:
                continue
            features = np.stack(self.feature_lists[classifier])
            # Normalise each feature
            pipeline = Pipeline([("norm", StandardScaler())])
            features = pipeline.fit_transform(features)
            self.features[classifier] = features
            self.pipelines[classifier] = pipeline


def training_images(training_dir: str) -> Tuple[List[str], List[str]]:
    side_images = glob.glob(os.path.join(training_dir, "*s.jpg"))
    front_images = glob.glob(os.path.join(training_dir, "*f.jpg"))
    return side_images, front_images


def test_images(test_dir: str) -> List[str]:
    return glob.glob(os.path.join(test_dir, "*.jpg"))


def split_test_images(paths: List[str], reference: Dict) -> Tuple[List[str], List[str]]:
    side_images, front_images = [], []
    for path in paths:
        ref = reference[os.path.basename(path)]
        if ref.endswith("s.jpg"):
            side_images += [path]
        elif ref.endswith("f.jpg"):
            front_images += [path]
        else:
            raise RuntimeError("Unknown file ending")
    return side_images, front_images


def calc_eer(
    valid_classes: List[str], attempt_classes: List[str], distances: np.array, plot: bool = True
) -> Tuple[float, float]:
    # Equal Error Rate (EER) Graph
    trials = 10000
    fac = np.zeros((trials,))
    frc = np.zeros((trials,))
    thresholds = np.linspace(0, 1, trials)
    # valid_indexes = list(range(len(valid_classes)))
    for ti, threshold in enumerate(thresholds):
        # correct = est_classes[i] == matched[i]
        for ai, attempt_class in enumerate(attempt_classes):
            # Process for should be allowed
            # FRC : Distance Actual < Threshold
            if attempt_class in valid_classes:
                vi = valid_classes.index(attempt_class)
                frc[ti] += distances[vi, ai] > threshold
            # Process for should not be allowed
            # FAC : Distance Not In < Threshold
            else:
                fac[ti] += any(distances[:, ai] <= threshold)
    far = fac / (len(attempt_classes) - len(valid_classes))
    frr = frc / len(valid_classes)
    eer_idx = np.argwhere(np.diff(np.sign(far - frr))).flatten()[0]
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(thresholds, far, label="FAR", color="C3")
        ax.plot(thresholds, frr, label="FRR", color="C0")
        plt.axvline(x=eer_threshold, color="C2", linestyle="--")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("FAR / FRR ($\%$)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(linestyle=":")
        ax.legend()
        save_fig(fig, "eer_plot")

    return eer, eer_threshold





if __name__ == "__main__":
    latex_plots = True
    if latex_plots:
        # matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    # Load existing feature extractor for caching if available
    fe_pickle_path = "feature_extractor_dl_600.pickle"
    if True and os.path.exists(fe_pickle_path):
        fe = torch.load(fe_pickle_path)
    else:
        fe = BodyShapeFE(input_height=300)
    # Create a new classifier that uses the feature extractor (training is quick)
    classifier = BodyShapeClassifier(fe)

    # ## TRAINING SET ## #
    def map_class(path: str):
        return os.path.basename(path)[:-5]

    train_dir = r"C:\dsj\biometrics_cw\image_set\training"
    train_side_images, train_front_images = training_images(train_dir)
    train_images = train_side_images + train_front_images
    # Assume that each side image has a front image at the same index
    assert len(train_side_images) == len(train_front_images)
    train_classes = [map_class(p) for p in train_front_images]

    # ## TESTING SET ## #
    def map_test_image(path: str):
        return os.path.join(train_dir, img_mapping[os.path.basename(path)])

    test_dir = r"C:\dsj\biometrics_cw\image_set\test"
    test_images = test_images(test_dir)
    test_side_images, test_front_images = split_test_images(test_images, img_mapping)
    # Assume that each side image has a front image at the same index
    assert len(test_side_images) == len(test_front_images)
    test_classes = [map_class(map_test_image(p)) for p in test_front_images]

    # --- CACHE MAJOR FEATURES IMMEDIATELY ---
    for i, path in enumerate(tqdm(train_images + test_images)):
        fe.mask(path)
        fe.keypoints(path)
    torch.save(fe, fe_pickle_path)

    # Extract features from training set
    classifier.add_features(train_images)

    # Run test set classification
    correct_count = 0
    for path in test_front_images:
        data = classifier.infer(path)
        data["actual_class"] = classifier.get_class(map_test_image(path))
        correct = data["actual_class"] == data["est_class"]
        correct_count += correct
        print(
            "{:1} - {:10} - {:10} - {}".format(
                correct,
                data["actual_class"],
                data["est_class"],
                min(data["feature_dists"]),
            )
        )
    print(correct_count, "/", len(test_front_images))
    print(correct_count / len(test_front_images) * 100)
    front_correct = correct_count

    # Run test set classification
    correct_count = 0
    for path in test_side_images:
        data = classifier.infer(path)
        data["actual_class"] = classifier.get_class(map_test_image(path))
        correct = data["actual_class"] == data["est_class"]
        correct_count += data["actual_class"] == data["est_class"]
        print(
            "{:1} - {:10} - {:10} - {}".format(
                correct,
                data["actual_class"],
                data["est_class"],
                min(data["feature_dists"]),
            )
        )
    print(correct_count, "/", len(test_side_images))
    print(correct_count / len(test_side_images) * 100)
    side_correct = correct_count

    print(
        "\nCORRECT: {} / {} | {}".format(
            front_correct, len(test_front_images), len(test_side_images)
        )
    )

    plt.figure("Side")
    feature_distances = [
        classifier.data[path]["feature_dists"] for path in test_side_images
    ]
    feature_distances = np.stack(feature_distances)
    side = feature_distances
    # feature_distances = np.delete(feature_distances, 38, axis=1)
    # feature_distances = np.delete(feature_distances, 41, axis=1)
    plt.imshow(feature_distances, cmap="plasma_r")
    plt.figure("Front")
    feature_distances = [
        classifier.data[path]["feature_dists"] for path in test_front_images
    ]
    feature_distances = np.stack(feature_distances)
    front = feature_distances
    # feature_distances = np.delete(feature_distances, 41, axis=1)
    # print(train_side_images[38])
    # print(train_side_images[41])
    # print(train_front_images[41])
    plt.imshow(feature_distances, cmap="plasma_r")
    fig, ax = plt.subplots(2, 1)
    combined = np.stack((front, side), axis=0)
    combined = np.mean(combined, axis=0)

    eer, eer_threshold = calc_eer(test_classes, train_classes, combined)

    classes = classifier.classes["left"]
    est_indexes = [est for est in np.argmin(combined, axis=1)]
    est_classes = [classes[est] for est in est_indexes]
    matched = [classifier.data[path]["actual_class"] for path in test_side_images]
    correct_total = 0
    correct_rejected = 0
    correct_thresholded = 0
    for i in range(len(matched)):
        correct = est_classes[i] == matched[i]
        print(est_classes[i] == matched[i],  est_classes[i], matched[i])
        correct_total += correct
        correct_thresholded += correct and combined[i, train_classes.index(est_classes[i])] < eer_threshold

    print("CCR: {:.2f} ({})".format(correct_total / len(test_classes) * 100, correct_total))
    print("TCCR: {:.2f} ({})".format(correct_thresholded / len(test_classes) * 100, correct_thresholded))
    print("EER: {:.2f}".format(eer * 100))
    print("EER Threshold: {:.2f}".format(eer_threshold))

    # for i in range(len(matched)):
    # print(correct_total, correct_rejected / total)
    # print(correct)

    # classes = classifier.classes["left"]
    matches = np.argmin(combined, axis=1)
    # combined = np.zeros_like(combined)
    # for i, match in enumerate(matches):
    #     combined[i, match] = 1
    rank_combined = np.array(list([scipy.stats.rankdata(row) for row in combined]))
    rank_combined[rank_combined > 3] = 4
    combined[rank_combined > 3] = np.max(combined)
    ax[0].imshow(combined, cmap="plasma_r")
    ax[1].imshow(rank_combined, cmap="plasma_r")
    plt.show()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/masks", exist_ok=True)
    os.makedirs("outputs/masks_head", exist_ok=True)
    os.makedirs("outputs/keypoints", exist_ok=True)
    for path in train_images + test_images:
        cv.imwrite(os.path.join("outputs/masks", os.path.basename(path)), fe.mask(path))
        cv.imwrite(
            os.path.join("outputs/masks_head", os.path.basename(path)),
            fe.head_mask(path),
        )
        image = fe.image(path).copy()
        keypoints = fe.keypoints(path)
        image = draw_body_connections(image, [keypoints], thickness=2, alpha=0.7)
        image = draw_keypoints(image, [keypoints], radius=5, alpha=0.8)
        cv.imwrite(os.path.join("outputs/keypoints", os.path.basename(path)), image)

    # fig, ax = plt.subplots(1, 1)
    # for i, feature_dist in enumerate(data["feature_dists"]):
    #     if i == data["actual_index"]:
    #         color = "g"
    #     elif i == data["estimated_index"]:
    #         color = "r"
    #     else:
    #         color = "b"
    #     ax.bar(i, feature_dist, color=color)
    # Bar chart
    # test_index = 5
    # data = classifier.data[test_images[test_index]]

    # path = train_images[data["actual_index"]]
    # image = cv.imread(path)
    # image = resize_image(image, height=classifier.image_height)
    # image_dst = draw_body_connections(
    #     image, [classifier.data[path]["keypoints"]], thickness=2, alpha=0.7
    # )
    # image_dst = draw_keypoints(
    #     image_dst, [classifier.data[path]["keypoints"]], radius=5, alpha=0.8
    # )
    # for n, (i, j, p) in enumerate(classifier.data[path]["keypoints"]):
    #     if p:
    #         cv.putText(
    #             image_dst,
    #             str(n),
    #             (i, j),
    #             cv.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (255, 255, 255),
    #             1,
    #             cv.LINE_AA,
    #         )
    # cv.imshow("Correct", resize_image(image_dst, height=600))

    # path = train_images[data["estimated_index"]]
    # image = cv.imread(path)
    # image = resize_image(image, height=classifier.image_height)
    # image_dst = draw_body_connections(
    #     image, [classifier.data[path]["keypoints"]], thickness=2, alpha=0.7
    # )
    # image_dst = draw_keypoints(
    #     image_dst, [classifier.data[path]["keypoints"]], radius=5, alpha=0.8
    # )
    # cv.imshow("Estimated", resize_image(image_dst, height=600))

    # path = test_images[test_index]
    # image = cv.imread(path)
    # image = resize_image(image, height=classifier.image_height)
    # image_dst = draw_body_connections(
    #     image, [classifier.data[path]["keypoints"]], thickness=2, alpha=0.7
    # )
    # image_dst = draw_keypoints(
    #     image_dst, [classifier.data[path]["keypoints"]], radius=5, alpha=0.8
    # )
    # cv.imshow("Input", resize_image(image_dst, height=600))

    # fig, ax = plt.subplots(1, 1)
    # xs, ys = [], []
    # for path in train_images:
    #     xs += [data_dict[path]["feature"][0]]
    #     ys += [data_dict[path]["feature"][1]]
    # ax.scatter(torch.stack(xs), torch.stack(ys), color="b")
    # # ax.scatter(xs[correct_index], ys[correct_index], color='g')
    # xs, ys = [], []
    # for path in test_images:
    #     x0 = data_dict[path]["feature"][0]
    #     y0 = data_dict[path]["feature"][1]
    #     x1 = data_dict[map_test_image(path)]["feature"][0]
    #     y1 = data_dict[map_test_image(path)]["feature"][1]
    #     ax.plot((x0, x1), (y0, y1), color="b")
    #     ax.scatter(x0, y0, color="r")
    #     ax.scatter(x1, y1, color="g")

    plt.show()
    while True:
        if cv.waitKey(1) & 0xFF == 27:  # exit if pressed `ESC`
            break

    # for i, (front_path, side_path) in enumerate(zip(train_front_images, train_side_images)):
    #     if abs(train_front_heights[i] - train_side_heights[i]) > 30:
    #         plt.figure()
    #         image = classifier.image(front_path)
    #         plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    #         plt.figure()
    #         image = classifier.image(side_path)
    #         plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    #         plt.show()
    #         exit()

    # fig, ax = plt.subplots(1, 1)
    # plt.scatter(train_side_heights, train_front_heights, color='b')
    # plt.scatter(test_side_heights, test_front_heights, color='r')
    # plt.show()

# feature_dict = {}
#     for feature, value in data[dist_dict_key].items():
#         if feature not in feature_dict:
#             feature_dict[feature] = {}
#             feature_dict[feature]["class"] = []
#             feature_dict[feature]["value"] = []
#         feature_dict[feature]["class"] += [data["class"]]
#         feature_dict[feature]["value"] += [value]

# for feature in feature_dict.values():
#     feature["value"] = np.array(feature["value"])
