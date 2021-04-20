import numpy as np
import math
import random
import mysklearn.mypytable as mpt


def get_column_by_index(table, col_index):
    """get column at col_index

    Args:
        table(list of list): table containing data
        col_index(int): index of column to retrieve

    Returns:
        column(list): return column of data at col_index from table
    """
    column = []

    for row in table:
        if (row[col_index] != ""):
            column.append(row[col_index])
    return column

def get_column(table, header, col_name):
    """get column with col_name from table

    Args:
        table(MyPyTable): table containing data
        header(list): names of table columns
        col_name(string): name of column to retrieve from table

    Returns:
        column(list): return column of data with col_name from table
    """
    col_index = header.index(col_name)
    column = []

    for row in table:
        if (row[col_index] != ""):
            column.append(row[col_index])
    return column

def group_by(table, header, group_by_col_name):
    """group table rows into various subtables by common values in given column

    Args:
        table(MyPyTable): table containing data
        header(list): names of table columns
        group_by_col_name(string): name of column to group data by

    Returns:
        group_names(list): list of values for which data was grouped
        group_subtables(list of list): list of tables cooresponding to a group value
    """
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)
    
    # we need the unique values for our group by column
    group_names = list(set(col)) # e.g. 74 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate subtable based on its group_by_col_name value
    for row in table:
        group_by_value = row[col_index]
        # which subtable to put this row in
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row)
    
    return group_names, group_subtables

def group_by_col(col):
    """group table rows into various subtables by common values in given column

    Args:
        col(list): column to group data in

    Returns:
        group_names(list): list of values for which data was grouped
        group_subtables(list of list): list of tables cooresponding to a group value
    """
    
    # we need the unique values for our group by column
    group_names = list(set(col)) # e.g. 74 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate subtable based on its group_by_col_name value
    for val in col:
        group_by_value = val
        # which subtable to put this row in
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(val)
    
    return group_names, group_subtables

def group_by_index(table, col_index):
    """group table rows into various subtables by common values in given column

    Args:
        table(list of list): table containing data
        col_index(int): index of column to group data by

    Returns:
        group_names(list): list of values for which data was grouped
        group_subtables(list of list): list of tables cooresponding to a group value
    """
    col = get_column_by_index(table, col_index)
    
    # we need the unique values for our group by column
    group_names = list(set(col)) # e.g. 74 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate subtable based on its group_by_col_name value
    for row in table:
        group_by_value = row[col_index]
        # which subtable to put this row in
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row)
    
    return group_names, group_subtables

def group_subtable_by_index(subtable, table, col_index):
    """group subtable rows into various subtables by common values in given column

    Args:
        subtable(list of list): table containing data
        table(list of list): parent table from which subtable was derived
        col_index(int): index of column to group data by

    Returns:
        group_names(list): list of values for which data was grouped
        group_subtables(list of list): list of tables cooresponding to a group value
    """
    col = get_column_by_index(subtable, col_index)
    parent_col = get_column_by_index(table, col_index)
    
    # we need the unique values for our group by column
    group_names = list(set(col)) # e.g. 74 75, 76, 77
    parent_group_names = list(set(parent_col))
    group_subtables = [[] for _ in range(len(parent_group_names))] # [[], [], [], []]
    # algorithm: walk through each row and assign it to the appropriate subtable based on its group_by_col_name value
    for row in subtable:
        group_by_value = row[col_index]
        # which subtable to put this row in
        group_index = parent_group_names.index(group_by_value)
        group_subtables[group_index].append(row)
    
    return parent_group_names, group_subtables

def get_frequencies(table, header, col_name):
    """get the frequencies of values in a given column

    Args:
        table(MyPyTable): table containing data
        header(list): names of table columns
        col_name(string): name of column to count frequencies for

    Returns:
        values(list): discrete values in column
        counts(list): number of instances of values in column
    """
    values = []
    counts = []
    col = get_column(table, header, col_name)
    for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        else:
            # have seen this value 
            val_index = values.index(value)
            counts[val_index] += 1

    return values, counts

def get_col_frequencies(col):
    """get the frequencies of values in a given column

    Args:
        col(list): column of data

    Returns:
        values(list): discrete values in column
        counts(list): number of instances of values in column
    """
    values = []
    counts = []
    for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        else:
            # have seen this value 
            val_index = values.index(value)
            counts[val_index] += 1

    return values, counts

def compute_euclidean_distance(v1, v2):
    """compute distance between two values

    Args:
        v1(list): first list of number type 
        v2(list): second list of number type

    Returns:
       dist(float): computed distance between v1 and v2
    """
    assert len(v1) == len(v2)

    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist 

def compute_categorical_distance(v1, v2):
    """ compute distance between two categorical values

    Args:   
        v1(list of string): first list of categorical values
        v2(list of string): second list of categorical values

    Returns:
        dist(int): computed distance between v1 and v2
    """
    if v1 == v2:
        return 0
    else: 
        return 1


def randomize_in_place(alist, parallel_list=None):
    """randomize list in place (no return val)

    Args:
        alist(list): list to be randomized
        parallel_list (list): list parallel to alist to be randomized in parallel
    """
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None: 
            parallel_list[i], parallel_list[rand_index] = \
                parallel_list[rand_index], parallel_list[i]

def classify_mpgs(vals):
    """classify list of mpg values as a fuel economy rating

    Args:
        vals(list): list of mpg values to be classified

    Returns:
        ratings(list): classified fuel economy ratings parallel to vals
    """
    ratings = []
    for val in vals:
        if val <= 13:
            ratings.append(1)
        elif val >= 14 and val < 15:
            ratings.append(2)
        elif val >= 15 and val < 17:
            ratings.append(3)
        elif val >= 17 and val < 20:
            ratings.append(4)
        elif val >= 20 and val < 24:
            ratings.append(5)
        elif val >= 24 and val < 27:
            ratings.append(6)
        elif val >= 27 and val < 31:
            ratings.append(7)
        elif val >= 31 and val < 37:
            ratings.append(8)
        elif val >= 37 and val < 45:
            ratings.append(9)
        else:
            # mpg >= 45
            ratings.append(10)
    return ratings

def classify_mpg_val(mpg):
    """classify list of mpg values as a fuel economy rating

    Args:
        vals(list): list of mpg values to be classified

    Returns:
        ratings(list): classified fuel economy ratings parallel to vals
    """
    if mpg <= 13:
        return 1
    elif mpg >= 14 and mpg < 15:
        return 2
    elif mpg >= 15 and mpg < 17:
        return 3
    elif mpg >= 17 and mpg < 20:
        return 4
    elif mpg >= 20 and mpg < 24:
        return 5
    elif mpg >= 24 and mpg < 27:
        return 6
    elif mpg >= 27 and mpg < 31:
        return 7
    elif mpg >= 31 and mpg < 37:
        return 8
    elif mpg >= 37 and mpg < 45:
        return 9
    else:
        # mpg >= 45
        return 10

def categorize_weight(vals):
    """ categorize vehicle weights based on NHTSA vehicle sizes

    Args:
        vals(list): list of weight values to be categorized

    Returns:
        rankings(list): categorized weight ranking 
    """
    rankings = []
    for val in vals:
        if val <= 1999:
            rankings.append(1)
        elif val >= 2000 and val < 2500:
            rankings.append(2)
        elif val >=2500 and val < 3000:
            rankings.append(3)
        elif val >=3000 and val < 3500:
            rankings.append(4)
        else:
            # weight >= 3500
            rankings.append(5)

    return rankings

def numerical_to_categorical(vals):
    """ Convert numerical values to categorical values

    Args: 
        vals(list): list of numerical values to be converted to strings (categorical)

    Returns:
        categorical_vals(list): converted string (categorical) values
    """
    categorical_vals = []
    for val in vals:
        categorical_vals.append(str(val))
    return categorical_vals

def normalize(data):
    """normalize the given data using min max normalization

    Args:
        data(list): list of values to be normalized

    Returns:
        normalized_data(list): normalized values (between 0 and 1)
    """
    normalized_data = []
    for val in data:
        normalized_val = (val - min(data)) / (max(data) - min(data))
        normalized_data.append(normalized_val)
    return normalized_data

def transpose(table):
    """ helper function to transpose a table (swap rows and columns)

    Args:
        table(MyPyTable): table to transpose

    Returns:
        transposed_table(list of lists): result of transpose operations on table arg

    Notes:
        Referenced the following Geeks for Geeks article: https://www.geeksforgeeks.org/python-transpose-elements-of-two-dimensional-list/
    """
    transposed_table = []
    transposed_table = list(map(list, zip(*table)))
    return transposed_table

def average(list_vals):
    """calculate the average value in a liss

    Args:
        list_vals(list): list of numerical values 

    Returns:
        calculated average of list_vals
    """
    sum_vals = 0
    for val in list_vals:
        sum_vals += val
    return sum_vals / len(list_vals)

def select_attribute(instances, available_attributes):
    entropies = []
    for attribute in available_attributes:
        # get attribute labels for current attribute
        group_names, group_subtables = group_by_index(instances, int(attribute[-1]))
       
        attribute_label_entropies = []
        num_instances = []
        for index, label in enumerate(group_names): # ['senior', 'mid', 'junior']
            num_label_instances = len(group_subtables[index])
            num_instances.append(num_label_instances)
            classifier_group_names, classifier_group_subtables = group_by_index(group_subtables[index], -1)

            if(len(classifier_group_names) > 1):
                # calculate entropy
                p = []
                for index in range(len(classifier_group_names)):
                    p.append(len(classifier_group_subtables[index]) / num_label_instances)
                entropy = -(p[0] * math.log(p[0], 2)) 
                for i in range(len(classifier_group_names) - 1):
                    entropy = entropy - (p[i+1] * math.log(p[i+1], 2))
                attribute_label_entropies.append(entropy)
                p.clear()
            else:
                # entropy is 0 since all instances of class label are same classification
                attribute_label_entropies.append(0)
        
        # calculate weighted entropy
        weighted_entropy = attribute_label_entropies[0] * (num_instances[0] / len(instances))
        for index in range(len(attribute_label_entropies) - 1):
            weighted_entropy = weighted_entropy + attribute_label_entropies[index + 1] * (num_instances[index + 1] / len(instances))
        entropies.append(weighted_entropy)
        
        num_instances.clear()
        attribute_label_entropies.clear()
    
    min_entropy_index = entropies.index(min(entropies))
    return available_attributes[min_entropy_index]

def partition_instances(instances, split_attribute, header, attribute_domains):
    # comments refer to split_attribute "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0

    partitions = {} # key (attribute value): value (partition)
    # task: try to finish this
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions 

def all_same_class(instances):
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def tdidt(current_instances, available_attributes, header, attribute_domains):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            values_subtree.append(["Leaf", partition[0][-1], len(partition), len(current_instances)])
            tree.append(values_subtree)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            col = get_column_by_index(partition, -1)
            col_values, col_counts = get_col_frequencies(col)
            max_index = col_counts.index(max(col_counts))
            values_subtree.append(["Leaf", col_values[max_index], len(partition), len(current_instances)])
            tree.append(values_subtree)
       
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            # use current instances to find majority class label
            class_groups, class_subtables = group_by_index(current_instances, -1)
            class_subtable_lengths = []
            for subtable in class_subtables:
                class_subtable_lengths.append(len(subtable))
            max_index = class_subtable_lengths.index(max(class_subtable_lengths))
            # add leaf node to tree
            tree = ["Leaf", class_groups[max_index], class_subtable_lengths[max_index], sum(class_subtable_lengths)]

        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(), header, attribute_domains)
            values_subtree.append(subtree)
            tree.append(values_subtree)

    return tree

def predict_classifier(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        # now I need to find which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return predict_classifier(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label

def get_decision_rules(tree, attribute_names, class_name, decision_rules, decision_str=""):    
    decision_str = decision_str + "IF "
    info_type = tree[0]
    if info_type == "Attribute":
        tree_attribute = tree[1]
        if attribute_names is not None:
            header_index = int(tree_attribute[-1])
            decision_str = decision_str + str(attribute_names[header_index]) + " == "
        else:
            decision_str = decision_str + str(tree_attribute) + " == "
        
        loop_str_additions = ""
        for i in range(2, len(tree)):
            value_list = tree[i]
            decision_str = decision_str + str(value_list[1]) + " "
            loop_str_additions = str(value_list[1]) + " "

            if value_list[2][0] == "Attribute":
                decision_str = decision_str + "AND "
                loop_str_additions = loop_str_additions + "AND "
                get_decision_rules(value_list[2], attribute_names, class_name, decision_rules, decision_str)
            else:
                # leaf
                decision_str = decision_str + "THEN " + str(class_name) + " = " + str(value_list[2][1])
                loop_str_additions = loop_str_additions + "THEN " + str(class_name) + " = " + str(value_list[2][1])

                decision_rules.append(decision_str)
            decision_str = decision_str[: - len(loop_str_additions)]
    else: # "Leaf"
        decision_str = decision_str + " THEN " + str(class_name) + " = " + str(tree[1])
        decision_rules.append(decision_str)
        decision_str = ""
    
    return decision_rules