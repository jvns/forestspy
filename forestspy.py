from collections import namedtuple
import numba
import numpy as np
Node = namedtuple('Node', ['feature_id', 'threshold', 'left', 'right', 'class_distribution', 'feature_name'])

class ForestSpy(object):
    def __init__(self, forest_classifier, feature_names):
        self.forest = forest_classifier
        self.feature_names = feature_names
        self.populate_trees()

    def populate_trees(self):
        self.trees = []
        for est in self.forest.estimators_:
            self.trees.append(Tree(est, self.feature_names))

    def predict_nodes(self, features):
        return [t.predicted_node(features) for t in self.trees]


@numba.jit(nopython=True)
def _predicted_node(features, feature_ids, thresholds, children_right, children_left):
    node_id = 0
    while node_id != -1:
        prev_node_id = node_id
        feature_id = feature_ids[node_id]
        feature_value = features[feature_id]
        threshold = thresholds[node_id]
        if feature_value > threshold:
            node_id = children_right[node_id]
        else:
            node_id = children_left[node_id]
    return prev_node_id

class Tree(object):
    def __init__(self, decision_tree, feature_names):
        self.tree = decision_tree.tree_
        self.feature_names = feature_names
        self.threshold = self.tree.threshold
        self.children_right = self.tree.children_right
        self.children_left = self.tree.children_left
        self.feature_ids = self.tree.feature
        self.value = self.tree.value

    def condition(self, node_id):
        node = self.node(node_id)
        return node

    def parent(self, node_id):
        left_parent = np.nonzero(self.children_left == node_id)[0]
        right_parent = np.nonzero(self.children_right == node_id)[0]
        if len(left_parent) > 0:
            return left_parent[0], '<'
        else:
            return right_parent[0], '>'

    def print_path(self, features):
        self.print_path_from_node(0, features)

    def node(self, node_id):
        feature_id = self.feature_ids[node_id]
        feature_name=self.feature_names[feature_id]
        class_distribution=self.value[node_id]
        threshold=self.threshold[node_id]
        left=self.children_left[node_id]
        right=self.children_right[node_id]
        return Node(feature_id=feature_id, feature_name=feature_name, class_distribution=class_distribution, threshold=threshold, left=left, right=right )

    def predicates(self, features):
        node_id = 0
        predicates = []
        while node_id != -1:
            node = self.node(node_id)
            value = features[node.feature_id]
            if value > node.threshold:
                format_string = "{feature} > {threshold}"
                node_id = node.right
            else:
                format_string = "{feature} <= {threshold}"
                node_id = node.left
            predicates.append(format_string.format(
                feature=node.feature_name,
                threshold=node.threshold))
        print node.class_distribution
        return '\n AND '.join(sorted(predicates))

    def predicted_node(self, features):
        return _predicted_node(features, self.feature_ids, self.threshold, self.children_right, self.children_left)
