from mysklearn import myutils, mypytable, myevaluation
import random
import operator
import math

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        mean_x = sum(X_train) / len(X_train)
        mean_y = sum(y_train) / len(y_train)
        
        m = sum([(X_train[i] - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train))]) / sum([(X_train[i] - mean_x) ** 2 for i in range(len(X_train))])
        b = mean_y - m * mean_x

        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        # for each test (x_val) in X_test, calculate y_predicted (y_val) using self.slope and self.intercept
        for x in X_test:
            # y = mx + b
            y_predicted.append(self.slope * x[0] + self.intercept) 
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        n_features = len(X_test[0])
        for test in X_test:
            for i, instance in enumerate(self.X_train):
                # append the class label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                # append the distance to test
                if(type(X_test[0][0]) == int or type(X_test[0][0]) == float):
                    dist = myutils.compute_euclidean_distance(instance[:n_features], test)
                else:
                    dist = myutils.compute_categorical_distance(instance[:n_features], test)
                instance.append(dist)
            
            # sort X_train 
            train_sorted = sorted(self.X_train, key=operator.itemgetter(-1))

            # grab the top k
            top_k = train_sorted[:self.n_neighbors]
            for instance in top_k:
                distances.append(instance[-1])
                neighbor_indices.append(instance[-2])

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        header = []
        n_features = len(X_test[0])
        for val_num in range(n_features):
            col_name = "col_" + str(val_num)
            header.append(col_name)
        header.append("class label")
        header.append("index")
        header.append("distance")

        for test in X_test:
            for i, instance in enumerate(self.X_train):
                # append the class label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                # append the distance to test
                if(type(X_test[0][0]) == int or type(X_test[0][0]) == float):
                    dist = myutils.compute_euclidean_distance(instance[:n_features], test)
                else:
                    dist = myutils.compute_categorical_distance(instance[:n_features], test)
                instance.append(dist)
            
            # sort X_train 
            train_sorted = sorted(self.X_train, key=operator.itemgetter(-1))

            # grab the top k
            top_k = train_sorted[:self.n_neighbors]
            top_k_table = mypytable.MyPyTable(header, top_k)

            values, counts = myutils.get_frequencies(top_k_table.data, top_k_table.column_names, "class label")
            highest_val_count = max(counts)
            highest_val_index = counts.index(highest_val_count)
            y_predicted.append(values[highest_val_index])

        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        priors_dict = {}
        posteriors_dict = {}

        num_instances = len(y_train)
        
        # calculate priors
        group_names, group_subtables = myutils.group_by_col(y_train)
        for index, subtable in enumerate(group_subtables):
            dict_key = group_names[index]
            priors_dict[dict_key] = len(subtable) / num_instances
            # add classifiers to posteriors_dict to make nested dictionary
            posteriors_dict[dict_key] = {}

        # calculate posteriors
        classifier_group_names, classifier_group_subtables = myutils.group_by_index(X_train, len(X_train[0]) - 1)
        for index, subtable in enumerate(classifier_group_subtables):
            num_classifier_instances = len(subtable)
            classifier_key = classifier_group_names[index]
            
            # for each attribute (minus classifier at last index)
            for i in range(len(subtable[0]) - 1):
                # group by attribute at index i and calculate posteriors
                attribute_group_names, attribute_group_subtables = myutils.group_subtable_by_index(subtable, X_train, i)
                for index, attribute_subtable in enumerate(attribute_group_subtables):
                    key = "att" + str(i) + "=" + str(attribute_group_names[index])
                    val = len(attribute_subtable) / num_classifier_instances
                    posteriors_dict[classifier_key][key] = val
        self.X_train = X_train
        self.y_train = y_train
        self.priors = priors_dict
        self.posteriors = posteriors_dict

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        # find classifiers
        classifiers, group_subtables = myutils.group_by_col(self.y_train)

        for test in X_test:
            # predict classifier for each test instance
            probabilities_dict = {}

            # calculate probability for each classifier
            for classifier in classifiers:
                calc_probability = self.priors[classifier]
                for index, key in enumerate(test):
                    key_str = "att" + str(index) + "=" + str(key)
                    calc_probability = calc_probability * self.posteriors[classifier][key_str]
                probabilities_dict[classifier] = calc_probability

            # find largest probability 
            largest_probability = max(probabilities_dict, key=probabilities_dict.get)

            # append predicted classifer to y_predicted
            y_predicted.append(largest_probability)
        return y_predicted

class MyZeroRClassifier:

    def __init__(self):
        """Initializer for MyZeroRClassifier.

        """
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self):
        return max(set(self.y_train), key = self.y_train.count)

class MyRandomClassifier:
    
    def __init__(self):
        """Initializer for MyRandomClassifier.

        """
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self):
        rand_index = random.randint(0, len(self.y_train) - 1)
        return self.y_train[rand_index]

'''
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        # calculate a header (e.g. ["att0", "att1", ...])
        header = []
        for i in range(len(X_train[0]) - 1):
            header.append("att" + str(i))

        # calculate the attribute domains dictionary
        attribute_domains = {}
        for index, attribute in enumerate(header):
            col = myutils.get_column_by_index(X_train, index)
            col_values, col_counts = myutils.get_col_frequencies(col)
            attribute_domains[attribute] = col_values

        available_attributes = header.copy() 
        tree = myutils.tdidt(X_train, available_attributes, header, attribute_domains)
        # print("tree:", tree)
        self.tree = tree
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predicted = []
        for test in X_test:
            # generate header for tree
            header = []
            for i in range(len(test)):
                header.append("att" + str(i))
            prediction = myutils.predict_classifier(header, self.tree, test)
            predicted.append(prediction)

        return predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        decision_rules = []
        decision_rules = myutils.get_decision_rules(self.tree, attribute_names, class_name, decision_rules)
    
        for rule in decision_rules:
            print(rule)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
'''
def select_attribute(instances, available_attributes, class_vals):
    """Selects the attribute to split on based on the calculated entropy for all the current attributes.

    Args:
        instances (list of list of obj): The current partition of instances being analyzed
        available_attributes (list of str): The list of the current available attributes to split on
        class_vals (list of str): The domain of class labels

    Returns:
        split_attribute (str): The attribute to split on (example: att0)
    """

    # Get Estart
    e_start = 0
    for labels, counts in label_counts(instances).items():
        frac = counts / len(instances)
        e_start += (-frac * math.log(frac, 2))


    entropies = []
    for attribute in available_attributes:
        # att1 means train_index is 1
        train_index = int(attribute[-1])
        # Senior, Mid, Junior
        domain_vals = partition_instances(instances, train_index)
        # domain_vals = myutils.find_unique_values(instances, train_index)

        weighted_entropy = 0
        for _, val_instances in domain_vals.items():
            frac_list = []
            for _, lab_counts in label_counts(val_instances).items():
                frac_list.append(lab_counts/len(val_instances)) # 4/6, 2/6

            e_dom_val = 0
            for frac in frac_list:
                e_dom_val += -(frac) * math.log(frac, 2)

            weighted_entropy += (len(val_instances) / len(instances)) * e_dom_val

        entropies.append(e_start - weighted_entropy)

    max_value = max(entropies)
    max_index = entropies.index(max_value)
    split_attribute = available_attributes[max_index]  
                
    return split_attribute

def partition_instances(instances, split_attribute):
    """Partitions the instances based on the domain of the split attribute

    Args:
        instances (list of list of obj): The current partition of instances being analyzed
        split_attribute (str): The attribute that we are going to split on

    Returns:
        partitions (dict): The partitions of the instances grouped by the domain of the split attribute
    """
    if isinstance(split_attribute, str):
        attribute_index = int(split_attribute[-1])
    else:
        attribute_index = split_attribute
    attribute_domain = myutils.find_unique_values(instances, attribute_index)
    partitions = {}
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    
    return partitions

def all_same_class(instances):
    """Determines whether the instances are all part of the same class label

    Args:
        instances (list of list of obj): The current partition of instances being analyzed

    Returns:
        True if instances all have the same class label
        False otherwise
    """
    class_labels = []
    for instance in instances:
        if instance[-1] not in class_labels:
            class_labels.append(instance[-1])
    
    if len(class_labels) == 1:
        return True
    else:
        return False

def label_counts(instances):
    """Counts the occurences of each label in the current instances

    Args:
        instances (list of list of obj): The current partition of instances being analyzed

    Returns:
        labels (dict): Key is class labels in instances, value is count of each in instances
    """
    labels = {}
    for instance in instances:
        if instance[-1] not in labels:
            labels[instance[-1]] = 1
        else:
            labels[instance[-1]] += 1
    return labels

def get_majority_vote(instances):
    """Determines the class label with the majority of instances having that label

    Args:
        instances (list of list of obj): The current partition of instances being analyzed
    
    Returns:
        majority_label (str): The class label that has the most occurences in the instances
    """
    labels = {}
    for instance in instances:
        if instance[-1] not in labels:
            labels[instance[-1]] = 1
        else:
            labels[instance[-1]] += 1
    
    majority_label = max(labels.items(), key=operator.itemgetter(1))[0]
    return majority_label

def tdidt(current_instances, available_attributes, class_vals):
    """The algorithm that recursively creates the decision tree

    Args:
        current_instances (list of list of obj): The current partition of instances being analyzed
        available_attributes (list of str): The list of the current available attributes to split on
        class_vals (list of str): The domain of class labels
    
    Returns:
        tree (list of list of obj): The nested list that represents the tree
    """
    # Find split attribute based on minimizing entropy
    split_attribute = select_attribute(current_instances, available_attributes, class_vals)

    available_attributes.remove(split_attribute)

    tree = ["Attribute", split_attribute]
    partitions = partition_instances(current_instances, split_attribute)

    for attribute_value, partition in partitions.items():
        value_subtree = ["Value", attribute_value]

        # CASE 1
        if (len(partition) > 0 and all_same_class(partition)):
            leaf_node = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            value_subtree.append(leaf_node)
        # CASE 2
        elif (len(partition) > 0 and len(available_attributes) == 0):
            label = get_majority_vote(partition)
            leaf_node = ["Leaf", label, len(partition), len(current_instances)]
            value_subtree.append(leaf_node)
        # CASE 3
        elif len(partition) == 0:
            label = get_majority_vote(current_instances)
            leaf_node = ["Leaf", label, len(current_instances), len(current_instances)]
            tree = leaf_node
            return tree
        else:
            subtree = tdidt(partition, available_attributes.copy(), class_vals)
            value_subtree.append(subtree)

        tree.append(value_subtree)

    return tree

def predict_helper(instance, tree):
    """Helper recursive function for traversing the tree and predicting the class label

    Args:
        instance (list of obj): The instance we are predicting the label for
        tree (list of list of obj): Nested list representation of the decision tree

    Returns:
        If at a leaf node, returns the class label
        else, returns itself to traverse through the next node
    """
    info_type = tree[0]
    if info_type == "Attribute":
        attribbute_index = int(tree[1][-1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[attribbute_index]:
                return predict_helper(instance, value_list[2])
    else:
        return tree[1]

def tdidt_print_rules(tree, rule, class_name, default_header, attribute_names):
    """Recursive helper for printing decision rules

    Args:
        tree (list of list of obj): the decision tree
        rule (str): the string representation of the decision rule
        class_name (str): the class label
        default_header (list of str): the tree represented names of attributes (i.e. att0, att1)
        attribute_names (list of str): the actual names of the attributes
    """
    info_type = tree[0]

    if info_type == "Attribute":
        if rule != "IF ":
            rule += " AND "

        if attribute_names is None: 
            rule += tree[1]
        else:
            index = default_header.index(tree[1])
            rule += attribute_names[index]
            
        for i in range(2, len(tree)):
            value_list = tree[i]
            rule2 = str(rule) + " = " + str(value_list[1])
            tdidt_print_rules(value_list[2], rule2, class_name, default_header, attribute_names)

    else: # "Leaf"
        print(rule, "THEN", class_name, "=", tree[1])

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        # Original header [att0, att1, att2, ..., attn]
        header = []
        for i in range(0, len(self.X_train[0])):
            header.append("att" + str(i))

        class_vals = myutils.find_unique_values(y_train)

        # Complete table with stiched together X_train and y_train
        train = [self.X_train[i] + [self.y_train[i]] for i in range(0, len(self.X_train))]

        # Original header [att0, att1, att2, ..., attn]
        available_attributes = header.copy()
        tree = tdidt(train, available_attributes, class_vals)

        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            tree_copy = self.tree.copy()
            label = predict_helper(instance, tree_copy)
            if label is not None:
                y_predicted.append(label)
            else:
                y_predicted.append(self.y_train[0])

        
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        default_header = ["att" + str(i) for i in range(0, len(self.X_train))]
        tdidt_print_rules(self.tree, "IF ", class_name, default_header, attribute_names)

        pass


class MyRandomForestClassifier:
    """
        N = number of decision trees
        M = number of accurate trees
        F = number of attributes to use
    """
    
    def __init__(self, N, M, F):
        """Initializer for MyRandomForestClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.best_trees = None
        self.N = N
        self.M = M
        self.F = F
        self.best_m_trees = None
        self.M_attr_sets = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # stitch together X_train and y_train so y_train is in right most column
        train = [self.X_train[i] + [self.y_train[i]] for i in range(0, len(self.X_train))]

        trees = []
        attr_sets = []
        tree_accuracies = []
        for i in range(self.N):
            # Call bootstrap method to get different random data from data set (each instance has every column)
            bootstrapped_table = myutils.compute_bootstrapped_sample(train)
            bootstrapped_y = myutils.get_column_by_index(bootstrapped_table, -1)
            bootstrapped_X = myutils.remove_column(bootstrapped_table, -1)

            # call train test split to get X_train, y_train, X_test, y_test
            tree_X_train, tree_X_validation, tree_y_train, tree_y_validation = myutils.train_test_split(bootstrapped_X, bootstrapped_y, 1/3)
            # TODO: Randomly select F indices and make X_train those F columns
            # print(tree_X_train[:10])
            # print(tree_X_validation[:10])
            # print(tree_y_train[:10])
            # print(tree_y_validation[:10])

            num_attributes = len(tree_X_train[0])
            attr_indices = myutils.generate_F_indices(num_attributes, self.F)
            attr_indices = sorted(attr_indices)
            attr_sets.append(attr_indices)

            subsetted_tree_X_train = myutils.attribute_subset_table(tree_X_train, attr_indices)

            # create decision tree
            decision_tree = MyDecisionTreeClassifier()
            # print(subsetted_tree_X_train[:10])
            decision_tree.fit(subsetted_tree_X_train, tree_y_train)
            trees.append(decision_tree)

            # calculate tree accuracy using validation set
            predicted = decision_tree.predict(tree_X_validation)
            match_count = 0
            for index, prediction in enumerate(predicted):
                if prediction == tree_y_validation[index]:
                    match_count += 1
            accuracy = match_count / len(predicted)
            tree_accuracies.append(accuracy)
        
        # select M best trees based on accuracies
        # sort accuracies and cooresponding trees
        zipped_lists = zip(trees, tree_accuracies, attr_sets)
        sorted_zipped = sorted(zipped_lists, reverse=True, key=lambda x:x[1])
        tuples = zip(*sorted_zipped)
        sorted_trees, sorted_accuracies, sorted_attr_sets = [list(tuple) for tuple in tuples]
        best_m_trees = sorted_trees[:self.M]
        best_sorted_attr_sets = sorted_attr_sets[:self.M]

        self.best_m_trees = best_m_trees
        self.M_attr_sets = best_sorted_attr_sets

    def predict(self, X_test):
        classifications = []
        # run X_test instances over selected M trees to make predictions
        for instance in X_test:
            predictions = []
            num_trees = len(self.best_m_trees)
            for i in range(0, num_trees - 1):
                test_instance = []
                for j in self.M_attr_sets[i]:
                    test_instance.append(instance[j])
                predicted = self.best_m_trees[i].predict([test_instance])
                for index, prediction in enumerate(predicted):
                    predictions.append(prediction)
            
            # Get majority label from predictions
            classifications.append(max(set(predictions), key=predictions.count))

        return classifications