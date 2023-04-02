"""
You should use the decision tree algorithm that we discussed in class, using the “information gain” metric in order 
to determine what is the best question to ask at each stage. Some other implementation details to consider are:

  If two issues have the same information gain, use the issue with the earlier letter.
    
  If you reach a point where you still have both Democrats and Republicans, but they all have the 
    exact same voting record, create a leaf that classifies them as the majority left.

  If you reach a point where there are still Democrats and Republicans, but they have different 
    voting records even though the best information gain is 0, keep adding to the tree. Another 
    issue may now be able to split them.

  If a branch has 0 reps, or a tied number of indistinguishable reps, classify based on the
    majority at its parent node. If this is tied too, keep going back up the tree until you
    have a majority.

============================================

In this tree, to classify a new representative we would first ask how s/he voted on Issue C.
If the rep voted “yea”, ask about Issue A. Here a vote of “yea” or an abstention means a
Democrat, while a vote of “nay” means a Republican. But if the rep voted “nay” on Issue C,
ask about Issue F. Here a “yea” mean Republican, while on a “nay” you need to ask about
Issue L...and so on.

Second, you must estimate your decision tree's accuracy. To do this, you will use
leave-one-out-cross-validation. Loop through every datum. For that datum, exclude it from
the calculation and create a new decision tree as was done above. Then after it has been
trained and tuned, test it on the left-out datum. Do this for all your data. (Note that when
taking out every fourth datum to be the tuning set, you should do this as if the testing
datum never existed.) Print out the accuracy on the left-out testing data as the estimate of
your tree's data. Please do not print out the hundreds of trees you create for this step.

"""
#
#
#
#  COMMENTS TO SELF ABIOVE TO BE DELETED
#
#
#

"""
Description of the program up here 

This program implements uses hard-coding because it's tailored for a decision tree with 3 classes.
Examples of this being:
  - class labels are hard coded 
  - child_entropy in information_gain gets the weighted_child entropy --> only 2 labels means the other is implied
"""


import argparse
import math
import numpy as np
__author__ = "Emilee Oquist"    #
__license__ = "MIT"
__date__ = "March 2023"


# ArgumentParser
parser = argparse.ArgumentParser(
    description='Python program for creating a decision tree.')
parser.add_argument('filename', type=str, help='Path to the data file to use')
args = parser.parse_args()
filename = args.filename

# Hard-coding class labels
vote_negative = '-'
vote_positive = '+'
vote_neutral = '.'

# The example tree I have shown on the HW was never meant to be taken as the real answer.
# But I will say that the pruned tree that I got has 19 nodes altogether, and that the root node is Issue F.
# This tree has an estimated accuracy of ~95%. Hope that helps.


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):  # = None means optional
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict(self, X):
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

    def build_tree(self, X, y, depth):
        # Implement decision tree algorithm here
        # First, you must create a decision tree and print it. To do this, there will be no test set.
        pass

    def build_tuning_set(self, dataset):
        return dataset[::4]  # slicing

    # Calculate the entropy of the entire training set.
    #     This represents the amount of information we need to classify a new, unknown datum of the same sort as was found in the training set.
    # Calculate what the information gain would be, were we to split based on each feature, in turn.
    # Choose the single best feature, and divide the data set into two or more discrete groups.
    #     e.g. split based on medium, dividing the set into oil paintings, acrylic paintings, and watercolors.
    # If a subgroup is not uniformly labeled, recurse.

    def entropy(self, data, labels_list):
        """
        Computes the entropy of the given data for the decision tree.
        """
        entropy = 0
        labels, counts = np.unique(data[labels_list], return_counts=True)
        # entropy summation
        for label in range(len(labels)):
            probability = float(counts[label]) / float(len(data))
            entropy += (probability * math.log2(probability))
        # ensure that the entropy values are non-negative --> logs doing normal log things
        return -(entropy)

    def information_gain(self, data, labels, feature):
        """
        Measure the quality (information gain) of a feature split.
        """
        # Compute the entropy of the parent node
        parent_entropy = self.entropy(data, labels)

        # Compute the entropy of the child nodes 
        unique_values = np.unique(data[:, feature]) # "give me all rows in the data array, but only the column indexed by 'feature'."
        child_entropies = []
        for value in unique_values:
            child_indices = data[:, feature] == value
            child_labels = labels[child_indices]
            if len(child_labels) == 0:
                child_entropy = 0
            else:
                child_entropy = self.entropy(data[child_indices], child_labels)
            child_entropies.append(child_entropy)

        # Compute the information gain
        counts = [sum(labels == value) for value in np.unique(labels)]
        child_weights

        # Compute the information gain
        return parent_entropy - child_entropy

# Pruning the tree is done by assessing your entire tree's current accuracy on the tuning set, and its accuracy
# when each internal node is pruned. (A pruned node is one that is replaced by a leaf that
# classifies a rep based on the majority of representatives in the training set who would have
# reached that node.) If the best pruned tree is at least as good as the overall tree, making
# the prune permanent and repeat the process. If two possible prunings are tied for best, use
# the pruning that eliminates more nodes. Keep pruning until every possible chop would make
# your tree perform worse on the tuning set.
