import numpy as np
from scipy import stats
from decision_tree_node import Node

class DecisionTree:

    def predict(self, example):
        """
        Predict the label for a given example.
        """
        idx = 0

        while True:
            current_node = self.nodes_list[idx]

            if current_node.leaf_node:
                return current_node.label_hat

            split_dim = current_node.dim_to_split
            threshold = current_node.thresh_to_split

            if example[split_dim] <= threshold:
                idx = current_node.childs[0]
            else:
                idx = current_node.childs[1]

    def test(self, test_features, true_labels):
        """
        Predict the labels for the test set and calculate accuracy.
        """

        predicted_labels = np.zeros_like(true_labels)

        for idx, true_label in enumerate(true_labels):
            example = test_features[idx]
            predicted_labels[idx] = self.predict(example)

        accuracy = np.mean(true_labels == predicted_labels)
        self.test_accuracy = round(accuracy * 100, 2)
        print(f"Test Accuracy: {self.test_accuracy} %")

    def split_node(self, node_collection, node_idx):
        """
        Splits a node and updates the node collection.
        """

        current_node = node_collection[node_idx]
        current_node.leaf_node = False

        feature_idx = current_node.dim_to_split
        split_point = current_node.thresh_to_split

        left_node_data = current_node.data[current_node.data[:, feature_idx] <= split_point]
        right_node_data = current_node.data[current_node.data[:, feature_idx] > split_point]

        left_labels = current_node.labels[current_node.data[:, feature_idx] <= split_point]
        right_labels = current_node.labels[current_node.data[:, feature_idx] > split_point]

        left_label_hat = self._find_mode(left_labels)
        right_label_hat = self._find_mode(right_labels)

        new_left_node = Node(len(node_collection), node_idx, left_node_data, left_labels, left_label_hat)
        new_right_node = Node(len(node_collection) + 1, node_idx, right_node_data, right_labels, right_label_hat)

        node_collection.append(new_left_node)
        node_collection.append(new_right_node)

        current_node.childs = [new_left_node.node_id, new_right_node.node_id]
        print(f"Node {node_idx} was split into {new_left_node.node_id} and {new_right_node.node_id}")

        return node_collection

    def train(self, features, labels, max_leaves, max_thresholds_to_check):
        """
        Train the decision tree.
        """
        self.nodes_list = [Node(0, None, features, labels, 0)]

        should_stop = False
        while not should_stop:

            pure_leaves = 0
            best_loss_improvement = -np.inf
            best_node_idx = None

            for idx, current_node in enumerate(self.nodes_list):
                if not current_node.leaf_node or current_node.is_pure_node():
                    if current_node.is_pure_node():
                        pure_leaves += 1
                    continue

                if not current_node.loss_improv_calc:
                    optimal_split_dim, optimal_threshold, optimal_loss = self._find_best_split(current_node.data, current_node.labels, max_thresholds_to_check)
                    current_node.loss_improv = current_node.loss_current - optimal_loss
                    current_node.dim_to_split = optimal_split_dim
                    current_node.thresh_to_split = optimal_threshold
                    current_node.loss_improv_calc = True

                if current_node.loss_improv > best_loss_improvement:
                    best_loss_improvement = current_node.loss_improv
                    best_node_idx = idx

            if pure_leaves == self._count_leaves(self.nodes_list):
                should_stop = True
                print("All leaves are pure.")

            if best_node_idx is not None:
                self.nodes_list = self.split_node(self.nodes_list, best_node_idx)
            if len(self.nodes_list) >= max_leaves:
                print("Reached the maximum number of leaves.")
                should_stop = True

        train_loss = sum(node.loss_current for node in self.nodes_list if node.leaf_node)
        self.train_accuracy = round(100 * (1 - train_loss / len(labels)), 2)
        print(f"Training Accuracy: {self.train_accuracy} %")

    def _find_best_split(self, data, labels, max_thresholds):
        num_features = data.shape[1]
        best_loss = np.inf

        for feature_idx in range(num_features):
            best_threshold, current_loss = self._evaluate_feature_split(data, labels, feature_idx, max_thresholds)
            if current_loss < best_loss:
                best_loss = current_loss
                optimal_threshold = best_threshold
                optimal_feature_idx = feature_idx

        return optimal_feature_idx, optimal_threshold, best_loss

    def _evaluate_feature_split(self, data, labels, feature_idx, max_thresholds):
        threshold_candidates = self._find_threshold_candidates(data[:, feature_idx], max_thresholds)
        best_loss = np.inf
        for threshold in threshold_candidates:
            current_loss = self._compute_loss_if_split(data, labels, feature_idx, threshold)
            if current_loss < best_loss:
                best_loss = current_loss
                best_threshold = threshold
        return best_threshold, best_loss

    def _find_threshold_candidates(self, feature_values, max_thresholds):
        unique_values = np.unique(feature_values)
        if len(unique_values) <= max_thresholds:
            return unique_values
        else:
            return np.linspace(np.min(feature_values), np.max(feature_values), num=max_thresholds)

    def _compute_loss_if_split(self, data, labels, feature_idx, threshold):
        left_labels = labels[data[:, feature_idx] <= threshold]
        right_labels = labels[data[:, feature_idx] > threshold]

        total_loss = self._compute_loss(left_labels) + self._compute_loss(right_labels)
        return total_loss

    def _compute_loss(self, labels):
        most_common_label = self._find_mode(labels)
        return np.sum(labels != most_common_label)

    def _find_mode(self, labels):
        return stats.mode(labels, axis=None)[0][0]

    def _count_leaves(self, nodes):
        return sum(node.leaf_node for node in nodes)
