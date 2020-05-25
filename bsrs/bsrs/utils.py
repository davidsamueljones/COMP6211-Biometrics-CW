import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict


def figure_dir() -> str:
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    figure_dir = os.path.join(cur_dir, "..", "report", "figures")
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def save_fig(fig: plt.Figure, name: str, size: Tuple[int, int] = [3.5, 2]):
    fig.tight_layout()
    fig.set_size_inches(w=size[0], h=size[1])
    fig.savefig(os.path.join(figure_dir(), name + ".pgf"))
    fig.savefig(os.path.join(figure_dir(), name + ".pdf"))


def save_img(img: np.array, name: str):
    path = os.path.join(figure_dir(), name + ".png")
    cv.imwrite(path, img)


def draw_keypoints(image: np.array, keypoints, radius=1, alpha=1.0):
    overlay = image.copy()
    kp = keypoints
    for x, y, v in kp:
        if int(v):
            cv.circle(overlay, (int(x), int(y)), radius, (0, 0, 255), -1, cv.LINE_AA)
    return cv.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_keypoint_numbers(image: np.array, keypoints: List, size: float = 0.5, alpha: float = 1.0) -> np.array:
    overlay = image.copy()
    for n, (i, j, p) in enumerate(keypoints):
        if p:
            cv.putText(
                overlay,
                str(n),
                (i, j),
                cv.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return cv.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_connections(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    b_conn = [(0, 1), (1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]
    h_conn = [(0, 14), (0, 15), (14, 16), (15, 17)]
    l_conn = [(5, 6), (6, 7), (11, 12), (12, 13)]
    r_conn = [(2, 3), (3, 4), (8, 9), (9, 10)]

    kp = keypoints
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
