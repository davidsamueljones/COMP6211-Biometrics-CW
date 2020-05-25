import os
import cv2 as cv
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

import torch

from tqdm import tqdm
from typing import Tuple, List, Dict

import bsrs.utils as utils
from bsrs import BodyShapeFE, BodyShapeClassifier


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
""" Matching image for test class.
"""


def get_training_images(training_dir: str) -> Tuple[List[str], List[str]]:
    side_images = glob.glob(os.path.join(training_dir, "*s.jpg"))
    front_images = glob.glob(os.path.join(training_dir, "*f.jpg"))
    return side_images, front_images


def get_test_images(test_dir: str) -> List[str]:
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
    valid_classes: List[str],
    attempt_classes: List[str],
    distances: np.array,
    plot: bool = True,
) -> Tuple[float, float]:
    # Equal Error Rate (EER) Graph
    trials = 10000
    fac = np.zeros((trials,))
    frc = np.zeros((trials,))
    thresholds = np.linspace(0, 1, trials)
    # valid_indexes = list(range(len(valid_classes)))
    for ti, threshold in enumerate(thresholds):
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
        utils.save_fig(fig, "eer_plot")

    return eer, eer_threshold


def train_and_classify(fe_pickle_path: str):
    # --- INITIALISE FEATURE EXTRACTOR / CLASSIFIER --- #
    if True and os.path.exists(fe_pickle_path):
        fe = torch.load(fe_pickle_path)
    else:
        fe = BodyShapeFE(input_height=600)
    # Create a new classifier that uses the feature extractor (training is quick)
    classifier = BodyShapeClassifier(fe)

    # --- GET DATASETS --- #
    # ## TRAINING SET ## #
    def map_class(path: str):
        return os.path.basename(path)[:-5]

    train_dir = r"C:\dsj\biometrics_cw\image_set\training"
    train_side_images, train_front_images = get_training_images(train_dir)
    train_images = train_side_images + train_front_images
    # Assume that each side image has a front image at the same index
    assert len(train_side_images) == len(train_front_images)
    train_classes = [map_class(p) for p in train_front_images]

    # ## TESTING SET ## #
    def map_test_image(path: str):
        return os.path.join(train_dir, img_mapping[os.path.basename(path)])

    test_dir = r"C:\dsj\biometrics_cw\image_set\test"
    test_images = get_test_images(test_dir)
    test_side_images, test_front_images = split_test_images(test_images, img_mapping)
    # Assume that each side image has a front image at the same index
    assert len(test_side_images) == len(test_front_images)
    test_classes = [map_class(map_test_image(p)) for p in test_front_images]
    test_indexes = [train_classes.index(test_class) for test_class in test_classes]

    # --- CACHE MAJOR FEATURES IMMEDIATELY ---
    for i, path in enumerate(tqdm(train_images + test_images)):
        fe.mask(path)
        fe.keypoints(path)
    torch.save(fe, fe_pickle_path)

    # Extract features from training set
    classifier.add_features(train_images)
    # Create features for test set (this will cache results)
    for path in test_images:
        classifier.infer(path)

    # --- ASSESS PERFORMANCE --- #

    def feature_distances_vector(paths: List[str]) -> np.array:
        feature_distances = [classifier.data[path]["feature_dists"] for path in paths]
        return np.stack(feature_distances)

    def feature_distances_plot(distances: np.array, name: str, max_rank: int = 5):
        distances = np.log(distances)
        ranked = np.array(list([scipy.stats.rankdata(row) for row in distances]))
        if max_rank is not None:
            ranked[ranked > max_rank] = max_rank + 1
            # distances[ranked > max_rank] = np.max(distances)

        fig = plt.figure()
        plt.imshow(distances, cmap="plasma_r")
        utils.save_fig(fig, name + "_distances", size=(3.3, 1.2))
        fig = plt.figure()
        plt.imshow(ranked, cmap="plasma_r")
        utils.save_fig(fig, name + "_ranked", size=(3.3, 1.2))

    def cr_metrics(
        distances: np.array, test_paths: List[str], threshold: int, verbose: bool = True
    ) -> Tuple[int, int, Dict]:
        results = {}
        est_indexes = [est for est in np.argmin(distances, axis=1)]
        est_classes = [train_classes[est] for est in est_indexes]
        actual_classes = [map_class(map_test_image(p)) for p in test_paths]
        actual_indexes = [train_classes.index(ac) for ac in actual_classes]
        results["correct"] = np.zeros(len(test_paths))
        results["correct_thresholded"] = np.zeros(len(test_paths))
        results["wrong_thresholded"] = np.zeros(len(test_paths))
        results["est_indexes"] = est_indexes
        results["actual_indexes"] = actual_indexes
        for i in range(len(test_paths)):
            correct = est_classes[i] == actual_classes[i]
            within_threshold = (
                distances[i, train_classes.index(est_classes[i])] < threshold
            )
            if verbose:
                print(
                    "{} | E: {} - A: {} | E: {:.2f} - A: {:.2f}".format(
                        "✓" if est_classes[i] == actual_classes[i] else "✗",
                        est_classes[i],
                        actual_classes[i],
                        distances[i, est_indexes[i]],
                        distances[i, actual_indexes[i]],
                    )
                )
            results["correct"][i] = correct
            results["correct_thresholded"][i] = correct and within_threshold
            results["wrong_thresholded"][i] = not correct and not within_threshold

        correct_total = np.sum(results["correct"])
        correct_thresholded = np.sum(results["correct_thresholded"])
        if verbose:
            print("-----------------------------------------------------")
            print("Correct Total: {}".format(correct_total))
            print("Thresholded Total: {}".format(correct_thresholded))
        return correct_total, correct_thresholded, results

    def correct_match_plot(classes: int, cr_results: Dict):
        correct_indexes = cr_results["actual_indexes"]
        image = np.full((len(correct_indexes), classes, 3), 220)
        for y, x in enumerate(correct_indexes):
            image[y, x, :] = (0, 153, 51)
        for y, x in enumerate(cr_results["est_indexes"]):
            if cr_results["wrong_thresholded"][y]:
                image[y, x, :] = (0x87, 0xCE, 0xEB)
            elif cr_results["correct"][y]:
                if not cr_results["correct_thresholded"][y]:
                    image[y, x, :] = (255, 165, 0)
            else:
                image[y, x, :] = (194, 59, 34)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image)
        utils.save_fig(fig, "ccr_plot", size=(3.3, 1.2))

    print("\nSIDE")
    side_distances = feature_distances_vector(test_side_images)
    eer, threshold = calc_eer(test_classes, train_classes, side_distances, False)
    side_cr, side_tcr, _ = cr_metrics(side_distances, test_side_images, threshold)
    feature_distances_plot(side_distances, "side")
    print("EER: {:.2f} ({:.2f})".format(eer, threshold))

    print("\nFRONT")
    front_distances = feature_distances_vector(test_front_images)
    eer, threshold = calc_eer(test_classes, train_classes, front_distances, False)
    front_cr, front_tcr, _ = cr_metrics(front_distances, test_side_images, threshold)
    feature_distances_plot(front_distances, "front")
    print("EER: {:.2f} ({:.2f})".format(eer, threshold))

    print("\nCOMBINED")
    combined_distances = np.stack((side_distances, front_distances), axis=0)
    combined_distances = np.mean(combined_distances, axis=0)
    eer, threshold = calc_eer(test_classes, train_classes, combined_distances, True)
    combined_cr, combined_tcr, combined_cr_res = cr_metrics(
        combined_distances, test_side_images, threshold
    )
    feature_distances_plot(combined_distances, "combined")
    print("EER: {:.2f} ({:.2f})".format(eer, threshold))

    correct_match_plot(len(train_classes), combined_cr_res)

    # --- Output Images --- #
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
        image = utils.draw_connections(image, keypoints, thickness=2, alpha=0.7)
        image = utils.draw_keypoints(image, keypoints, radius=5, alpha=0.8)
        cv.imwrite(os.path.join("outputs/keypoints", os.path.basename(path)), image)

    def keypoint_img(path: str, save_name: str):
        image = fe.image(path)
        keypoints = fe.keypoints(path)
        image = utils.draw_connections(image, keypoints, thickness=2, alpha=0.7)
        image = utils.draw_keypoints(image, keypoints, radius=5, alpha=0.8)
        bx, by, bw, bh = cv.boundingRect(fe.contour(path))
        utils.save_img(image[by : by + bh, bx : bx + bw], save_name)

    def measurement_img(path: str, save_name: str):
        image = fe.measurements_img(path)
        bx, by, bw, bh = cv.boundingRect(fe.contour(path))
        utils.save_img(image[by : by + bh, bx : bx + bw], save_name)

    def head_mask_img(path: str, save_name: str):
        image = fe.head_mask(path)
        bx, by, bw, bh = cv.boundingRect(fe.contour(path, head=True))
        utils.save_img(image[by - 1 : by + bh + 1, bx - 1 : bx + bw + 1], save_name)

    keypoint_img(train_front_images[0], "front_kp_a")
    keypoint_img(train_side_images[0], "side_kp_a")
    keypoint_img(test_front_images[0], "front_kp_b")
    keypoint_img(test_side_images[0], "side_kp_b")

    measurement_img(train_front_images[0], "front_measurements_a")
    measurement_img(train_side_images[0], "side_measurements_a")
    measurement_img(test_front_images[0], "front_measurements_b")
    measurement_img(test_side_images[0], "side_measurements_b")

    head_mask_img(train_front_images[0], "front_head_a")
    head_mask_img(train_side_images[0], "side_head_a")
    head_mask_img(test_front_images[0], "front_head_b")
    head_mask_img(test_side_images[0], "side_head_b")
    head_mask_img(train_front_images[6], "front_head_c")
    head_mask_img(train_side_images[6], "side_head_c")
    head_mask_img(test_front_images[9], "front_head_d")
    head_mask_img(test_side_images[9], "side_head_d")


if __name__ == "__main__":
    latex_plots = True
    if latex_plots:
        matplotlib.use("pgf")
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
    train_and_classify(fe_pickle_path=fe_pickle_path)
    plt.show()
