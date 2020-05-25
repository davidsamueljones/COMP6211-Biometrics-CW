import os
import errno
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
import torchvision.models.detection as detection
import bsrs.utils as utils

from PIL import Image
from collections import OrderedDict
from torchvision import transforms
from typing import List, Dict

from bsrs.kps import coco_to_openpose, KEYPOINT_FEATURES


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

    def image(self, path: str) -> np.array:
        """Get the image from the provided path. The image will be placed in the cache
        once loaded and cleared whenever the next image is requested (only one image
        cached at a time).
        """
        if path not in self.cache["images"]:
            self.cache["images"].clear()
            image = cv.imread(path)
            if image is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
            self.cache["images"][path] = utils.resize_image(
                image, height=self.input_height
            )
        return self.cache["images"][path].copy()

    def process(self, path: str) -> Dict:
        """Create dictionary holding all image features.
        """
        keypoints = self.keypoints(path)
        direction = self.direction(path)
        dist_dict = self.dist_dict(path)
        self.height(path)
        self.head_mask(path)

        # Collate into data dictionary for outputs
        data = {}
        data["path"] = path
        data["keypoints"] = keypoints
        data["direction"] = direction
        data["dist_dict"] = dist_dict
        data["feature"] = self.feature(path)

        return data

    def feature(self, path: str) -> np.array:
        """Create a feature-vector, this will vary in length and content depending on
        the current view, i.e. different measurements and moments.
        """
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

    def keypoints(self, path: str) -> List[np.array]:
        """Use a CNN keypoint estimator to extract keypoints. Return keypoints using then
        COCO 18 point format where each keypoint is (x, y, p) where p is a flag indicating
        if the keypoint is valid. Only the first match is supported (no multi-person). This
        uses Pytorch and is GPU accelerated. The result of this will be cached.
        """
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

    def segment(self, path: str) -> np.array:
        """Perform semantic segmentation using a CNN. At each pixel location the value will
        be the closest match segment (as a COCO segment ID). This uses Pytorch and is GPU
        accelerated.
        """
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

    def direction(self, path: str) -> str:
        """ Determine direction of view using keypoints.  If left and right shoulder
        points found and far apart then front on otherwise direction between neck
        and nose dictates side.
        """
        keypoints = self.keypoints(path)
        if (not keypoints[2][2] or not keypoints[5][2]) or (
            np.linalg.norm(keypoints[2][:2] - keypoints[5][:2]) < 50
        ):
            if keypoints[1][0] - keypoints[0][0] > 0:
                return "left"
            else:
                return "right"
        else:
            return "front"

    def mask(self, path: str) -> np.array:
        """Create a mask of the human subject using the semantic segmentation result before
        applying a green screen removal in the HSV colour space. The result of this will be
        cached.
        """
        save_mask = any(map(lambda x: x in path, []))

        if path not in self.cache["masks"] or save_mask:
            # Semantic segmentation mask
            predictions = self.segment(path)
            mask = predictions == 15  # person
            mask = np.array(mask, dtype=np.uint8) * 255
            semantic_mask = mask.copy()
            # Remove LHS artefacts in images where argmax was a bit generous (FIXME)
            mask[:, :250] = 0

            # Green screen mask
            image = self.image(path)
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
            g_mask = np.logical_and(h > 53, h < 130)
            g_mask = np.logical_and(g_mask, s >= 102)
            g_mask = np.logical_and(g_mask, v >= 120)
            mask[g_mask] = 0

            if save_mask:
                contours, _ = cv.findContours(
                    mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
                )
                bx, by, bw, bh = cv.boundingRect(contours[0])
                image = self.image(path).copy()
                utils.save_img(image[by : by + bh, bx : bx + bw], "mask_input")
                indices = np.where(semantic_mask == 0)
                image[indices[0], indices[1], :] = [0, 0, 0]
                utils.save_img(image[by : by + bh, bx : bx + bw], "mask_semantic")
                indices = np.where(mask == 0)
                image[indices[0], indices[1], :] = [0, 0, 0]
                utils.save_img(image[by : by + bh, bx : bx + bw], "mask_final")
                contours, _ = cv.findContours(
                    mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
                )
                contour_img = np.zeros(image.shape)
                cv.drawContours(contour_img, contours, -1, (255, 255, 255), -1)
                cv.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
                utils.save_img(contour_img[by : by + bh, bx : bx + bw], "mask_contour")

            self.cache["masks"][path] = mask * 255

        return self.cache["masks"][path]

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

    def contour(self, path: str, head: bool = False):
        mask = self.head_mask(path) if head else self.mask(path)
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]
        contour = np.squeeze(contour)
        return contour

    def height(self, path: str):
        mask = self.mask(path)
        width_calc = np.sum(mask != 0, axis=1)
        y0 = utils.find_side(width_calc, 1, 20)
        y1 = utils.find_side(width_calc, 1, 20, from_right=True)
        return y1 - y0

    def neck_keypoints(self, path: str, plot: bool = False):
        keypoints = self.keypoints(path)
        contour = self.contour(path)
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
        nudge = 10
        index_left -= nudge
        index_right -= nudge
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

    def measurements_img(self, path: str) -> np.array:
        image = self.image(path)
        direction = self.direction(path)
        keypoints = self.keypoints(path)
        for item, keys in KEYPOINT_FEATURES[direction].items():
            for i1, i2 in zip(keys, keys[1:]):
                (x1, y1, p1), (x2, y2, p2) = keypoints[i1], keypoints[i2]
                if item.startswith("angle_"):
                    utils._draw_connection(
                        image, keypoints[i1], keypoints[i2], (255, 0, 0), 2
                    )
                elif item.startswith("y_"):
                    utils._draw_connection(
                        image, (x1, y1, p1), (x2, y1, p2), (255, 255, 255), 1
                    )
                    utils._draw_connection(
                        image, (x2, y1, p1), (x2, y2, p2), (255, 0, 255), 2
                    )
                # elif item.startswith("x_"):
                #     length += abs(x1 - x2)
                else:
                    utils._draw_connection(
                        image, keypoints[i1], keypoints[i2], (0, 0, 255), 2
                    )
        return image
