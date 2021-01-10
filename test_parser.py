"""
    This code builds a small decision tree
    trains over iris dataset

    We test using the built decision tree
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)

clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
clf.fit(X_train, y_train)

"""
    Now the testing part
"""
from tree_parser import parse_tree
import torch

model = parse_tree(clf)

print(model.forward(torch.Tensor(X_train)).shape)