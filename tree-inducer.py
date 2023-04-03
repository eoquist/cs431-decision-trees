"""
Description of the program up here 
"""


import argparse
import math
import numpy as np
import os.path

__author__ = "Emilee Oquist"    #
__license__ = "MIT"
__date__ = "March 2023"



# The example tree I have shown on the HW was never meant to be taken as the real answer.
# But I will say that the pruned tree that I got has 19 nodes altogether, and that the root node is Issue F.
# This tree has an estimated accuracy of ~95%. Hope that helps.

class Representative():
    """ An object stand-in for a representative that contains their ID, affiliation, and voting history. """
    def __init__(self, rep_ID, party, voting_record):
        # Representative information
        self.representative_ID = rep_ID
        self.party = party
        self.voting_record = voting_record

class Node():
    """ Decision Tree Classifier Node. """
    def __init__(self, issue=None, yea_voters=None, abstained_voters=None, children=None, parent_majority=None, is_leaf=None):
        """ Constructor for Node. """ 
        self.issue = ""
        self.num_issues = 0
        
        # Decision tree information
        self.children = children
        self.information_gain = 0
        self.is_leaf = is_leaf


class DecisionTreeClassifier:
    def __init__(self, depth=None):  # None --> optional
        """ Constructor for DecisionTreeClassifier. """
        self.max_depth = depth
        self.root - None

    def predict(self, data):
        """ Predicts the labels of the test data using the trained decision tree. """
        predictions = []
        for sample in X:
            node = self.tree
            while node.is_leaf() is False:
                if sample[node.feature] <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            predictions.append(node.label)
        return predictions

    def build_tree(self, training_data, labels_list, depth):
        base_case = self.check_base_case()
        if base_case is not None:
            return base_case
        # Implement decision tree algorithm here
        # First, you must create a decision tree and print it. To do this, there will be no test set.

        # Calculate the entropy of the entire training set.
        #     This represents the amount of information we need to classify a new, unknown datum of the same sort as was found in the training set.
        # Calculate what the information gain would be, were we to split based on each feature, in turn.
        # Choose the single best feature, and divide the data set into two or more discrete groups.
        #     e.g. split based on medium, dividing the set into oil paintings, acrylic paintings, and watercolors.
        # If a subgroup is not uniformly labeled, recurse.
        pass

    def print_tree(self, decision_tree):
        """ Prints the decision tree. """
        pass

    def check_base_case(self, data, labels_list, depth):
        # Base case: if all samples belong to one class, return a leaf node
        symbol = None
        for label in data:
            if 
        if len(data) == 1:
            return Node(data[0])
        # feature=None, threshold=None, left=None, right=None, info_gain=None, label=None

        # Base case: if maximum depth is reached, return a leaf node with the majority class label
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(max(set(y), key=y.count))
        pass


    def build_tuning_set(self, dataset):
        """
        You should do this as if the testing datum never existed. Print out the accuracy on the left-out testing data as the estimate of your tree's 
        data. Please do not print out the hundreds of trees you create for this step
        """
        return dataset[::4]  # slicing


    def entropy(self, data, labels_list):
        """ Computes the entropy of the given labels for the decision tree which is used in information_gain. """
        entropy = 0
        labels, counts = np.unique(data[labels_list], return_counts=True)
        # entropy summation
        for label in range(len(labels)):
            probability = counts[label] / len(data)
            entropy += (probability * math.log2(probability))
        return -(entropy)


    def information_gain(self, data, labels_list, feature):
        """ Measure the quality (information gain) of a feature split. """
        # Entropy of parent node
        parent_entropy = self.entropy(data, labels_list)

        for issue in voting_record

        # Compute the entropy of the child nodes
        labels, counts = np.unique(data[:, feature], return_counts=True)
        child_entropy = 0
        for value in labels:
            subset_labels = labels_list[data[:, feature] == value]
            child_entropy += (np.sum(counts[data[:, feature] == value]) /
                              np.sum(counts)) * self.entropy(subset_labels)

        # Compute the information gain
        return parent_entropy - child_entropy
    
    
    """
    If two issues have the same information gain, use the issue with the earlier letter.
        
    If you reach a point where you still have both Democrats and Republicans, but they all have the 
    exact same voting record, create a leaf that classifies them as the majority left.

    If you reach a point where there are still Democrats and Republicans, but they have different 
    voting records even though the best information gain is 0, keep adding to the tree. Another 
    issue may now be able to split them.

    If a branch has 0 reps, or a tied number of indistinguishable reps, classify based on the
    majority at its parent node. If this is tied too, keep going back up the tree until you
    have a majority.
    """


"""
Second, you must estimate your decision tree's accuracy. To do this, you will use
leave-one-out-cross-validation. Loop through every datum. For that datum, exclude it from
the calculation and create a new decision tree as was done above. Then after it has been
trained and tuned, test it on the left-out datum. Do this for all your data.

"""

# Pruning the tree is done by assessing your entire tree's current accuracy on the tuning set, and its accuracy
# when each internal node is pruned. (A pruned node is one that is replaced by a leaf that
# classifies a rep based on the majority of representatives in the training set who would have
# reached that node.) If the best pruned tree is at least as good as the overall tree, making
# the prune permanent and repeat the process. If two possible prunings are tied for best, use
# the pruning that eliminates more nodes. Keep pruning until every possible chop would make
# your tree perform worse on the tuning set.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python program for creating a decision tree.')
    parser.add_argument('filename', type=str, help='Path to the data file to use')
    args = parser.parse_args()
    filename = args.filename

    if not os.path.isfile(filename):
        print(f"Error: '{filename}' is not a valid file path.") # Python3.6+ f-strings
        exit(1)
    else:
        with open(filename, "r") as file:
            lines = file.readlines()

        # Parse lines and store data in NumPy array
        data = []
        labels_list = np.unique()
        for one_line in lines:
            info = one_line.strip().split("\t")
            representative_ID = info[0]
            label = info[1]
            voting_record = info[2]
            # Node
            node = Node(ID=representative_ID, label=label, voting_record=voting_record)
            data.append(node)

        # Convert list to NumPy array
        data = np.array(data)

        DT_classifier = DecisionTreeClassifier()
        DT_classifier.build_tree(data, labels_list, depth=0)
        pass