import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, features, labels):
        self.root = self._build_tree(features, labels)

    def _build_tree(self, features, labels, current_depth=0):
        number_of_samples, number_of_features = features.shape
        number_of_classes = len(np.unique(labels))

        if (
            current_depth >= self.max_depth
            or number_of_classes == 1
            or number_of_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(labels)
            return Node(value=leaf_value)

        best_feature_index, best_threshold = self._best_split(features, labels, number_of_features)
        if best_feature_index is None:
            leaf_value = self._most_common_label(labels)
            return Node(value=leaf_value)

        left_indices = features[:, best_feature_index] <= best_threshold
        right_indices = features[:, best_feature_index] > best_threshold

        left_subtree = self._build_tree(features[left_indices], labels[left_indices], current_depth + 1)
        right_subtree = self._build_tree(features[right_indices], labels[right_indices], current_depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, features, labels, number_of_features):
        best_gini_score = 1.0
        best_feature_index = None
        best_threshold = None

        for feature_index in range(number_of_features):
            unique_thresholds = np.unique(features[:, feature_index])
            for threshold in unique_thresholds:
                left_indices = features[:, feature_index] <= threshold
                right_indices = features[:, feature_index] > threshold
                if len(labels[left_indices]) == 0 or len(labels[right_indices]) == 0:
                    continue

                gini_score = self._gini_index(labels[left_indices], labels[right_indices])
                if gini_score < best_gini_score:
                    best_gini_score = gini_score
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _gini_index(self, left_labels, right_labels):
        total_count = len(left_labels) + len(right_labels)

        def gini(labels):
            class_probabilities = [np.sum(labels == class_label) / len(labels) for class_label in np.unique(labels)]
            return 1.0 - sum(p ** 2 for p in class_probabilities)

        weighted_gini = (
            len(left_labels) / total_count * gini(left_labels)
            + len(right_labels) / total_count * gini(right_labels)
        )
        return weighted_gini

    def _most_common_label(self, labels):
        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    def predict(self, features):
        return np.array([self._traverse_tree(sample, self.root) for sample in features])

    def _traverse_tree(self, sample, node):
        if node.is_leaf_node():
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._traverse_tree(sample, node.left)
        else:
            return self._traverse_tree(sample, node.right)
