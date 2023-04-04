"""
Political Decision Tree Classifier

If you would like to modify the code to allow for a maximum_depth specification,
the second to last line of the parse() function where DT_classifier is being 
initialized is where you can hard code a maximum depth. If you would like a user
to modify that - more must be changed.

Hard coded:
--> different vote type values: + - .
--> output labels: D and R
"""

import argparse
import math
from string import ascii_uppercase
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
    def __init__(self, issue=None, representatives=None, children=None, parent_majority=None, is_leaf=None):
        """ Constructor for Node. """ 
        # feature/inputs, labels/outputs, data
        self.issue = issue 
        self.representatives = representatives
        
        # Decision tree information
        self.children = children
        self.information_gain = 0
        self.is_leaf = is_leaf

        # Edge case
        self.parent_majority = parent_majority



class DecisionTreeClassifier:
    def __init__(self, root, issues_list=None, labels_list=None, vote_types=None, max_depth=None):  # None --> optional
        """ Constructor for DecisionTreeClassifier. """
        self.root = root
        self.children = np.array([])
        self.max_depth = max_depth
        self.issues_list = issues_list
        self.labels_list= labels_list
        self.vote_types = vote_types

    def print_tree(self, node, tab_indent_string):
        """ Prints the decision tree. """
        if node is None:
            return
        
        tab_indent_string = tab_indent_string + "\t"

        # Print the node's issue and political party
        if(node == self.root): # bad obj comparison :()
            print("Issue " + str(node.issue) + ":")
        else:
            print(tab_indent_string + "Issue " + str(node.issue) + ":")
            print()

        for child in node.children:
            self.print_tree(child,)

    def build_tree(self, node, depth):
        """ Builds the decision tree. """
        base_case = self.check_base_case()
        if base_case is not None:
            return base_case

        # Calculate the entropy of the entire training set.
        entropy = self.entropy(node)

        # Calculate what the information gain would be, were we to split based on each feature, in turn.
        best_info_gain, best_split_issue = self.information_gain(data=node.representatives, possible_splits=self.issues_list)

        # divide the data set into two or more discrete groups.

        # If a subgroup is not uniformly labeled, recurse.


    def check_base_case(self, node, max_depth=None):
        """ Checks the terminating / base cases for the build_tree function. """
        # Base case: if all samples belong to one class, return a leaf node
        if len(data) == 1:
            return Node(parent_label=labels[0], is_leaf=True)

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
        
        symbol = None
        for label in data:
            pass # !!!
        if len(data) == 1:
            return Node(data[0])
        # feature=None, threshold=None, left=None, right=None, info_gain=None, label=None

        # Base case: if maximum depth is reached, return a leaf node with the majority class label
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(max(set(y), key=y.count))
        pass


    def build_tuning_set(self, dataset):
        """
        You should do this as if the testing datum never existed. 
        
        Print out the accuracy on the left-out testing data as the estimate of your tree's 
        data. Please do not print out the hundreds of trees you create for this step
        """
        return dataset[::4]  # slicing


    def entropy(self, data):
        """ Computes the entropy of the given labels. """
        entropy = 0
        count = self.count_party_instances(data)
        # entropy summation
        for party_index in range(len(count)):
            probability = count[party_index] / len(data)
            entropy += (probability * math.log2(probability))
        return -(entropy)
    
    def count_party_instances(representatives):
        """ Count instances of democrats and republicans. """
        dem_count = 0
        rep_count = 0
        for rep in representatives:
            if rep.party == 'D':
                dem_count += 1
            elif rep.party == 'R':
                rep_count += 1
        return [dem_count, rep_count]


    def information_gain(self, data, possible_splits):
        """ Measure the quality (information gain) of a feature split. """
        parent_entropy = self.entropy(data)

        best_split_issue = ""
        best_info_gain = 0

        for issue in range(len(possible_splits)):
            for idx, symbol in enumerate(self.vote_types):
                representative_split = np.array([rep for rep in data if rep.voting_record[idx] == symbol])
                weighted_sum_split_entropy += (len(representative_split) / len(data)) * self.entropy(representative_split)

                #Calculate the information gained from the split
                information_gain = parent_entropy - weighted_sum_split_entropy

                if information_gain >= best_info_gain:
                    best_info_gain = information_gain
                    self.best_split_issue = issue
        return best_info_gain, best_split_issue
    
    def count_representative_vote_on_issue(self, data, issue):
        return list(filter(lambda rep: rep.voting_record == 'D',self.representatives))
    

    def leave_one_out_cross_validation(self):
        """ Estimate the decision tree's accuracy. """

        """
        Loop through every datum. For that datum, exclude it from
        the calculation and create a new decision tree as was done above. Then after it has been
        trained and tuned, test it on the left-out datum. Do this for all your data.
        """

        pass
    


# Pruning the tree is done by assessing your entire tree's current accuracy on the tuning set, and its accuracy
# when each internal node is pruned. (A pruned node is one that is replaced by a leaf that
# classifies a rep based on the majority of representatives in the training set who would have
# reached that node.) If the best pruned tree is at least as good as the overall tree, making
# the prune permanent and repeat the process. If two possible prunings are tied for best, use
# the pruning that eliminates more nodes. Keep pruning until every possible chop would make
# your tree perform worse on the tuning set.

def parse(filename):
    """ Parse the voting data file and create the decision tree. """
    with open(filename, "r") as file:
        lines = file.readlines()
        num_representatives = len(lines)

    list_of_representatives = np.array([])
    party_minidict = {"D": 0, "R": 0}
    num_issues = -1
    issues_list = []

    # Parse lines
    for one_line in lines:
        info = one_line.strip().split("\t")
        representative_ID = info[0]
        party = info[1]
        voting_record = info[2]

        if party in party_minidict:
            party_minidict[party] += 1
        else:
            party_minidict[party] += 1
            
        representative = Representative(representative_ID, party, voting_record)
        np.append(list_of_representatives,representative)

        if(num_issues is -1 or issues_list == [] ):
            num_issues = len(voting_record)
            issues_list = ascii_uppercase[:num_issues]
    # ----- ----- ----- ----- ----- #
    majority_party = max(party_minidict, key=party_minidict.get)
    parent_node = Node(issue=None, representatives=list_of_representatives, parent_majority=majority_party, is_leaf=False) 
    # children & info_gain missing
    labels = ['R',"D"]
    vote_types = ["+","-","."]
    DT_classifier = DecisionTreeClassifier(root=parent_node, issues_list=issues_list, labels_list=labels, vote_types=vote_types, max_depth=None)
    DT_classifier.build_tree(node=DT_classifier.root, depth=DT_classifier.max_depth)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python program for creating a decision tree.')
    parser.add_argument('filename', type=str, help='Path to the data file to use')
    args = parser.parse_args()
    filename = args.filename

    if not os.path.isfile(filename):
        print(f"Error: '{filename}' is not a valid file path.") # Python3.6+ f-strings
        exit(1)
    else:
        parse(filename)
