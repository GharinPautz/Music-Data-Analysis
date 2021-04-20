"""
Programmer: Kellie Colson
Class: CptS 322-02, Spring 2021
Programming Assignment #5
3/20/21
Did not attempt bonus

Description: This file contains methods for classifier evaluation
"""

import mysklearn.myutils as myutils
import random
import math
import copy
import numpy as np
import mysklearn.mypytable as mypytable

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       # seed random number generator
       random.seed(random_state)
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
       pass
    
    if shuffle: 
        # shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        myutils.randomize_in_place(X, parallel_list=y)
        pass
    
    num_instances = len(X)
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) 
    split_index = num_instances - test_size 

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    folds_index = []
    for _ in range(n_splits):
        folds_index.append([])

    # group samples into folds
    loop_num = 0
    for index, sample in enumerate(X):
        folds_index[loop_num % n_splits].append(index)
        loop_num += 1

    for fold in folds_index:
        # test on fold
        X_test_folds.append(fold)
        # train on remaining folds (folds - fold)
        remaining_fold_indices = []
        for new_fold in folds_index:
            if new_fold != fold:
                for val in new_fold:
                    remaining_fold_indices.append(val)
        X_train_folds.append(remaining_fold_indices)
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_train_folds = []
    X_test_folds = []
    header = ["index", "class label"]

    # append class label (y val) to each instance in X
    X_copy = copy.deepcopy(X)
    X_indices = []
    for index, instance in enumerate(X_copy):
        X_indices.append([index, y[index]])

    # create mypytable obj
    table = mypytable.MyPyTable(header, X_indices)

    # partition samples by class label
    group_names, group_subtables = myutils.group_by(table.data, table.column_names, "class label")

    # remove class labels from group subtables
    for group in group_subtables:
        for instance in group:
            instance.pop()

    folds = []
    for _ in range(n_splits):
        folds.append([])

    # for each group, distribute the instances one at a time to a fold
    loop_num = 0
    for group in group_subtables:
        for value in group:
            folds[loop_num % n_splits].append(value[0])
            loop_num += 1

    for fold in folds:
        # test on fold
        X_test_folds.append(fold)
        # train on remaining folds (folds - fold)
        remaining_fold_indices = []
        for new_fold in folds:
            if new_fold != fold:
                for val in new_fold:
                    remaining_fold_indices.append(val)
        X_train_folds.append(remaining_fold_indices)

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    # initialize matrix to all zeros
    for _ in range(len(labels)):
        row = []
        for _ in range(len(labels)):
            row.append(0)
        matrix.append(row)

    for i in range(len(y_true)):
        # handles diagonal of confusion matrix
        if y_true[i] == y_pred[i]:
            index = labels.index(y_true[i])
            matrix[index][index] += 1
        else:
            matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1
    return matrix