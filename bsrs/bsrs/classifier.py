
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Dict

from bsrs.features import BodyShapeFE


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
