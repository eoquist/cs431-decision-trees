"""
Political Decision Tree Classifier

The space complexity is *kinda* bad - but is better suited for easy-to-understand
lookups (in my opinion)

If you would like to modify the code to allow for a maximum_depth specification,
the second to last line of the parse() function where DT_classifier is being 
initialized is where you can hard code a maximum depth. If you would like a user
to modify that - more must be changed.

Hard coded:
--> different vote type values: + - .
--> output labels: D and R (in parse() as party_minidict and labels)
"""

import argparse
import math
from string import ascii_uppercase
import numpy as np
import os.path

__author__ = "Emilee Oquist"    # Received help from Lucas
__license__ = "MIT"
__date__ = "March 2023"



class Representative():
    """ An object stand-in for a representative that contains their ID, affiliation, and voting history. """
    def __init__(self, rep_ID, party, voting_record):
        # Representative information
        self.representative_ID = rep_ID
        self.party = party
        self.voting_record = voting_record



class Node():
    """ Decision Tree Classifier Node. """
    def __init__(self, issue=None, representatives=None, party_count=None, children=None, majority=None, is_leaf=None):
        """ Constructor for Node. """ 
        # feature/inputs, labels/outputs, data
        self.issue = issue 
        self.threshold_vote = None
        self.representatives = representatives
        self.party_count = party_count
        
        # Decision tree information
        self.children = children
        self.information_gain = 0
        self.is_leaf = is_leaf

        # Edge case
        self.majority = majority



class DecisionTreeClassifier:
    """  
    If you reach a point where you still have both Democrats and Republicans, but they all have the 
    exact same voting record, create a leaf that classifies them as the majority left.

    If you reach a point where there are still Democrats and Republicans, but they have different 
    voting records even though the best information gain is 0, keep adding to the tree. Another 
    issue may now be able to split them.

    If a branch has 0 reps, or a tied number of indistinguishable reps, classify based on the
    majority at its parent node. If this is tied too, keep going back up the tree until you
    have a majority.
     """
    
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
            self.print_tree(child,tab_indent_string)


    def fit(self, data):
        """ Tries to predict the label of the test datum using the trained decision tree. """
        predictions = []
        # prediction alg here

        return predictions


    def build_tree(self, node, possible_splits, depth=None):
        """ Builds the decision tree. """
        # If samples belong to one class, return a leaf node --> (entropy 0)
        if self.entropy(node) == 0:
            return Node(representatives=node.representatives, is_leaf=True)
        #   feature=None, threshold=None, left=None, right=None, info_gain=None, label=None
        
        
        # else recurse
        
        # Calculate what the information gain would be, were we to split based on each feature, in turn.
        best_split_issue, representative_split = self.get_best_split(node=node, data=node.representatives, possible_splits=self.issues_list)
        counts_dict, majority_party = self.count_party_instances_and_majority(representative_split)

        # divide the data set into two or more discrete groups.
        decision_node = Node(issue=best_split_issue, representatives=representative_split, party_count=counts_dict, majority=majority_party, is_leaf=False) 
        


    def entropy(self, data):
        """ Computes the entropy of the given labels. """
        entropy = 0
        counts_dict = self.count_party_instances_and_majority(data)
        # entropy summation
        for party in self.labels_list:
            probability = counts_dict[party] / len(data)
            entropy += (probability * math.log2(probability))
        return -(entropy)
    

    def count_party_instances_and_majority(self, data):
        """ Count instances of parties. """
        party_minidict = {}
        for label in self.labels_list:
            party_minidict[label] = len(list(filter(lambda rep: rep.party == label, data)))
        majority_party = max(party_minidict, key=party_minidict.get)
        return party_minidict, majority_party


    def get_best_split(self, data, possible_splits):
        """ Get the issue that gives the best information gain. """
        best_split_issue = ""
        best_info_gain = 0

        for issue in possible_splits:
            info_gain, representative_split = self.information_gain(data, issue)

            # If two issues have the same information gain, use the issue with the earlier letter.
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_issue = issue

        return best_split_issue, representative_split


    def information_gain(self, data, issue):
        """ Measure the quality (information gain) of a feature split. """
        parent_entropy = self.entropy(data)

        weighted_sum_entropy_of_split = 0
        representative_split = {}

        for symbol in self.vote_types:
            representative_split[symbol] = np.array([rep for rep in data if rep.voting_record[issue] == symbol])
            weighted_sum_entropy_of_split += (len(representative_split) / len(data)) * self.entropy(representative_split)

        # Calculate the information gained from the split
        information_gain = parent_entropy - weighted_sum_entropy_of_split
        return information_gain, representative_split


    def count_representative_vote_on_issue(self, data, issue):
        """ Returns a dict with a key-value pair where the key is the party and the value is the number of
        representatives who voted each symbol on that issue. """
        counts = {}
        for label in self.labels_list:
            vote_counts = np.zeros(len(self.labels_list))
            for idx, symbol in enumerate(self.vote_types):
                vote_counts[idx] = len(list(filter(lambda rep: rep.voting_record[issue] == symbol, data)))
            counts[label] = vote_counts
        return counts
    

    def build_tuning_set(self, dataset):
        """ Build the tuning set. """
        return dataset[::4]  # slicing
    

    def leave_one_out_cross_validation(self, data):
        """ Estimate the decision tree's accuracy. """
        accuracy = 0
        for index in range(len(data)):
            data_with_left_out = data[:index] + data[index+1:]
            """
            For that datum, exclude it from the calculation and create a new decision tree as was done above. 
            Then after it has been trained and tuned, test it on the left-out datum. Do this for all your data.

            Print out the accuracy on the left-out testing data as the estimate of your tree's 
            data.
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

    list_of_representatives = np.array([])
    num_issues = -1
    issues_list = []
    party_minidict = {"D": 0,"R": 0}

    # Parse lines
    for one_line in lines:
        info = one_line.strip().split("\t")
        representative_ID = info[0]
        party = info[1]
        voting_record_string = info[2]

        if party == "D":
            party_minidict["D"] += 1
        elif party == "R":
            party_minidict["R"] += 1

        num_issues = len(voting_record_string)
        if issues_list == []:
            issues_list = ascii_uppercase[:num_issues]

        voting_record = {}
        for i in range(num_issues):
            voting_record[ascii_uppercase[i]] = voting_record_string[i]
            
        representative = Representative(representative_ID, party, voting_record)
        np.append(list_of_representatives,representative)

    # ----- ----- ----- ----- ----- #
    majority_party = max(party_minidict, key=party_minidict.get)
    parent_node = Node(issue=None, representatives=list_of_representatives, party_count=None, majority=majority_party, is_leaf=False) 
    # children & info_gain missing
    labels = ["D","R"]
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
