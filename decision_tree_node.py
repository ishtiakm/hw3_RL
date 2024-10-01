import numpy as np

class Node:

    def __init__(self, id, parent, data, labels, label_hat):
        self.id = id                       # Identifier for the node's position in the list
        self.parent = parent               # Reference to the parent node's ID
        self.childs = None                 # To store child nodes if it gets split
        
        self.data = data                   # Data points that reach this node
        self.labels = labels               # Labels corresponding to the data

        self.loss_current = np.sum(labels != label_hat)  # Misclassification count for this node
        self.loss_improv = 0
        self.loss_improv_calc = False      # Whether improvement calculation has been done

        self.leaf_node = True              # Initially, node is a leaf
        self.dim_to_split = None           # Dimension chosen for splitting
        self.thresh_to_split = None        # Threshold value for the chosen split dimension
        self.label_hat = label_hat         # Predicted label for this node

        self.data_test = None              # Placeholder for test data, if used later
        self.labels_test = None            # Placeholder for test labels, if used later

    def change_leaf_node_status(self, new_status):
        """
        Update the node's leaf status.
        """
        self.leaf_node = new_status        # Changes the node's status to leaf or non-leaf

    def is_pure_node(self):
        """
        Checks if all labels in the node are the same.
        """
        return len(set(self.labels)) == 1  # If all labels are identical, node is pure

    def print_info(self):
        """
        Prints the node's data along with labels and current prediction.
        """
        combined_data = np.hstack((self.data, self.labels.reshape(-1, 1)))
        print("Data and Labels:\n{}".format(combined_data))
        print(f"Current Prediction: {self.label_hat}")
