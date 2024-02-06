import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations


class AssociationRules:
    """
    A class that provides methods for association rule mining and analysis.

    Methods:
    - equal_frequency: Perform equal-frequency discretization on a given column of data.
    - equal_width: Perform equal-width discretization on a given column of data.
    - support: Calculate the support of an itemset in a given dataset.
    - create_candidats: Generate candidate itemsets of a given size.
    - l: Generate frequent itemsets using the Apriori algorithm.
    - appriori: Generate frequent itemsets using the Apriori algorithm.
    - get_best_rules: Find the best association rules based on confidence.
    - association_rules: Generate all possible association rules from a set of items.
    - confidence: Calculate the confidence of an association rule.
    - lift: Calculate the lift of an association rule.
    - get_best_rules_lift: Find the best association rules based on lift.
    - cosine: Calculate the cosine similarity of an association rule.
    - get_best_rules_cosine: Find the best association rules based on cosine similarity.
    """

    def __init__(self, data):
        self.data = data

    def setDataset(self, data):
        self.data = data

    @staticmethod
    def equal_frequency(data, data_col, label, num_classes=0):
        """
        Perform equal-frequency discretization on a given column of data.

        Parameters:
        - data_col: str - The name of the column to be discretized.
        - label: str - The label prefix for the custom labels.
        - num_classes: int - The number of classes for discretization. If not provided, it will be calculated based on the data.

        Returns:
        - discretized_with_labels: Series - The discretized column with custom labels.
        """
        if num_classes == 0:
            num_classes = int((1 + 10 / 3) * math.log(len(data[data_col]), 10))

        # Perform equal-frequency discretization
        discretized = pd.qcut(
            data[data_col], q=num_classes, labels=False, duplicates="drop"
        )

        # Get the bin edges
        _, bin_edges = pd.qcut(
            data[data_col], q=num_classes, retbins=True, duplicates="drop"
        )

        # Create custom labels based on the bin edges
        labels = [f"{label}class{i}" for i in range(len(bin_edges) - 1)]

        # Assign the custom labels to the discretized column
        discretized_with_labels = discretized.map(dict(enumerate(labels)))

        return discretized_with_labels

    @staticmethod
    def equal_width(data, data_col, label, num_classes=0):
        """
        Perform equal-width discretization on a given column of data.

        Parameters:
        - data_col: str - The name of the column to be discretized.
        - label: str - The label prefix for the custom class labels.
        - num_classes: int - The number of classes to be created. If not provided, it is calculated based on the data size.

        Returns:
        - discretized_with_labels: Series - The discretized column with custom class labels.
        """
        if num_classes == 0:
            num_classes = int((1 + 10 / 3) * math.log(len(data[data_col]), 10))
        # Perform equal-width discretization
        discretized = pd.cut(
            data[data_col], bins=num_classes, labels=False, duplicates="drop"
        )

        # Get the bin edges
        _, bin_edges = pd.cut(
            data[data_col], bins=num_classes, retbins=True, duplicates="drop"
        )

        # Create custom labels based on the bin edges
        labels = [f"{label}class{i}" for i in range(len(bin_edges) - 1)]
        # labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]

        # Assign the custom labels to the discretized column
        discretized_with_labels = discretized.map(dict(enumerate(labels)))

        return discretized_with_labels

    def support(self, itemset):
        """
        Calculate the support of an itemset in a given dataset.

        Parameters:
        - itemset: list - The itemset for which to calculate support.

        Returns:
        - int - The support count of the itemset in the dataset.
        """
        data = self.data
        return len(
            data[
                data.apply(
                    lambda row: all(item in row.values for item in itemset), axis=1
                )
            ]
        )

    def l(self, candidatas, supp_min):
        """
        Returns a list of candidate items that meet the minimum support threshold.

        Parameters:
        - candidatas (list): The list of candidate items.
        - supp_min (float): The minimum support threshold.

        Returns:
        - l (list): The list of candidate items that meet the minimum support threshold.
        """
        l = []
        if self.support(candidatas) >= supp_min:
            l.append(candidatas)
        return l

    def appriori(self, suppmin):
        """
        Applies the Apriori algorithm to generate frequent itemsets.

        Args:
            suppmin (float): The minimum support threshold.

        Returns:
            list: A list of frequent itemsets.
        """
        itemset = self.get_itemsets()
        l = []
        suppmin = int(len(self.data) * suppmin)
        stop = False
        n = 1
        candidats = self.create_candidates(itemset, n)
        while not stop:
            supps = 0
            new_items = []
            for c in candidats:
                if self.support(c) >= suppmin:
                    supps += 1
                    l.append(c)
                    new_items.append(c)
            items = np.unique(np.array([item for tpl in new_items for item in tpl]))
            candidats = self.create_candidates(items, n + 1)
            if supps == 0:
                stop = True
            else:
                n += 1
        return l

    def get_best_rules(self, l, confmin):
        """
        Returns a dictionary of the best association rules for each item in the given list.

        Parameters:
        - l (list): The list of items.
        - confmin (float): The minimum confidence threshold.

        Returns:
        - dict: A dictionary where the keys are the items from the input list and the values are lists of the best association rules for each item.
        """
        res = {}
        for x in l:
            res[x] = []
            rules = self.association_rules(x)
            for r in rules:
                if self.confidence(r) >= confmin:
                    res[x].append(r)
            if len(res[x]) == 0:
                # remove it
                res.pop(x)
        return res

    def association_rules(self, combins):
        """
        Generate association rules from a list of combinations.

        Args:
            combinations (list): A list of combinations.

        Returns:
            list: A list of association rules, where each rule is represented as a tuple of two lists.
                  The first list contains the items in the antecedent of the rule, and the second list
                  contains the items in the consequent of the rule.
        """
        length = len(combins)
        rules = []
        for i in range(1, length):
            combi = list(combinations(combins, i))
            for c in combi:
                c = list(c)
                c2 = [x for x in combins if x not in c]
                rules.append((c, c2))
        return rules

    def confidence(self, rule):
        """
        Calculates the confidence of a given rule.

        Parameters:
        rule (tuple): A tuple representing the rule, where rule[0] is the antecedent and rule[1] is the consequent.

        Returns:
        float: The confidence of the rule.
        """
        return (self.support((rule[0] + rule[1]))) / self.support(rule[0])

    def lift(self, rule):
        """
        Calculate the lift of an association rule.

        Parameters:
        - rule: tuple - The association rule.

        Returns:
        - float - The lift of the association rule.
        """
        return self.confidence(rule) / self.support(rule[1])

    def get_best_rules_lift(self, l, confmin):
        """
        Returns a dictionary of the best association rules with lift greater than or equal to `confmin`.

        Parameters:
        - l: A list of items.
        - confmin: The minimum lift value for an association rule to be considered.

        Returns:
        - res: A dictionary where the keys are items from the input list `l`, and the values are lists of association rules that meet the lift criteria.

        """
        res = {}
        for x in l:
            res[x] = []
            rules = self.association_rules(x)
            for r in rules:
                if self.lift(r) >= confmin:
                    res[x].append(r)
            if len(res[x]) == 0:
                # remove it
                res.pop(x)
        return res

    def cosine(self, rule):
        """
        Calculates the cosine similarity between two items in a rule.

        Parameters:
        rule (tuple): A tuple containing two items in the rule.

        Returns:
        float: The cosine similarity between the two items.
        """
        return self.support((rule[0] + rule[1])) / math.sqrt(
            self.support(rule[0]) * self.support(rule[1])
        )

    def get_best_rules_cosine(self, l, confmin):
        """
        Returns a dictionary of the best association rules based on cosine similarity.

        Args:
            l (list): The list of items.
            confmin (float): The minimum confidence threshold.

        Returns:
            dict: A dictionary where the keys are the items from the input list `l` and the values are lists of the best association rules for each item.
        """
        res = {}
        for x in l:
            res[x] = []
            rules = self.association_rules(x)
            for r in rules:
                if self.cosine(r) >= confmin:
                    res[x].append(r)
            if len(res[x]) == 0:
                # remove it
                res.pop(x)
        return res

    def get_itemsets(self):
        """
        Returns a list of all unique items in the dataset.

        Returns:
        - items: list - A list of all unique items in the dataset.
        """
        data = self.data
        # get all the unique items from all the attributes
        items = np.unique(np.array([item for tpl in data.values for item in tpl]))
        return items

    def create_candidates(self, items, n):
        """
        Generate candidate itemsets of a given size.

        Args:
            items (list): A list of items.
            n (int): The size of the itemsets to generate.

        Returns:
            list: A list of candidate itemsets of size `n`.
        """
        uples = list(combinations(items, n))
        return uples
