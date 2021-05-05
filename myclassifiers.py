import myutils
import mypytable
import random
import operator
import math
import myevaluation

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

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        trees = []
        tree_accuracies = []
        for i in range(self.N):
            # bagging
            bootstrap_table, bootstrap_y = myutils.bootstrap(X_train, y_train)
            # divide bootstrap_table into test (1/3) and remainder (2/3) sets
            test_set_end_index = int(len(bootstrap_table) / 3)
            test_set = bootstrap_table[:test_set_end_index]
            test_set_y = bootstrap_y[:test_set_end_index]
            remainder_set = bootstrap_table[test_set_end_index:]
            remainder_set_y = bootstrap_y[test_set_end_index:]

            # split remainder set into training (2/3) and validation (1/3) sets
            validation_set_end_index = int(len(remainder_set) / 3)
            validation_set = remainder_set[:validation_set_end_index]
            validation_set_y = remainder_set_y[:validation_set_end_index]
            training_set = remainder_set[validation_set_end_index:]
            training_set_y = remainder_set_y[validation_set_end_index:]

            # select F attributes from training set
            # attribute_indexes = list(range(len(training_set[0])))
            # attribute_subset = myutils.random_attribute_subset(attribute_indexes, self.F)

            # training_set = myutils.attribute_subset_table(training_set, attribute_subset)

            # create decision tree
            decision_tree = MyDecisionTreeClassifier()
            decision_tree.fit(training_set, training_set_y)
            trees.append(decision_tree)

            # calculate tree accuracy using validation set
            predicted = decision_tree.predict(validation_set)
            match_count = 0
            for index, prediction in enumerate(predicted):
                if prediction == validation_set_y[index]:
                    match_count += 1
            accuracy = match_count / len(predicted)
            tree_accuracies.append(accuracy)
        
        # select M best trees based on accuracies
        # sort accuracies and cooresponding trees
        zipped_lists = zip(trees, tree_accuracies)
        sorted_zipped = sorted(zipped_lists, reverse=True, key=lambda x:x[1])
        tuples = zip(*sorted_zipped)
        sorted_trees, sorted_accuracies = [list(tuple) for tuple in tuples]
        best_m_trees = sorted_trees[:self.M]
        self.best_trees = best_m_trees

        # run test_set instances over selected M trees to make predictions
        predictions = [[] for i in range(len(test_set))]
        for tree in self.best_trees:
            predicted = tree.predict(test_set)
            for index, prediction in enumerate(predicted):
                predictions[index].append(prediction)
        
        classifications = []
        for test_predictions in predictions:
            classifications.append(max(set(test_predictions), key=test_predictions.count))
        return classifications

    def predict(test_set):
        # run test_set instances over selected M trees to make predictions
        predictions = [[] for i in range(len(test_set))]
        for tree in self.best_trees:
            predicted = tree.predict(test_set)
            for index, prediction in enumerate(predicted):
                predictions[index].append(prediction)
        
        classifications = []
        for test_predictions in predictions:
            classifications.append(max(set(test_predictions), key=test_predictions.count))
        return classifications


