from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable

interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

interview_tree = \
        ['Attribute', 'att0', 
            ['Value', 'Senior', 
                ['Attribute', 'att2', 
                    ['Value', 'no', 
                        ['Leaf', 'False', 3, 5]], 
                    ['Value', 'yes', 
                        ['Leaf', 'True', 2, 5]]]], 
            ['Value', 'Mid', 
                ['Leaf', 'True', 4, 14]], 
            ['Value', 'Junior', 
                ['Attribute', 'att3', 
                    ['Value', 'no', 
                        ['Leaf', 'True', 3, 5]], 
                    ['Value', 'yes', 
                        ['Leaf', 'False', 2, 5]]]]]

def test_random_forest_classifier_fit():
    mp_table = MyPyTable(interview_header, interview_table)
    # Formulate X_train and y_train
    y_train = mp_table.get_column('interviewed_well')
    X_train_col_names = ["level", "lang", "tweets", "phd"]
    X_train = mp_table.get_rows(X_train_col_names)

    myRF = MyRandomForestClassifier(N=4, M=2, F=4)
    myRF.fit(X_train, y_train)

    assert len(myRF.M_attr_sets) == myRF.M

def test_random_forest_classifier_predict():
    X_test = [["Mid", "Python", "no", "no", "True"],
              ["Mid", "R", "yes", "yes", "True"],
              ["Mid", "Python", "no", "yes", "True"]]
    
    y_test = ["True", "True", "True"]

    mp_table = MyPyTable(interview_header, interview_table)
    # Formulate X_train and y_train
    y_train = mp_table.get_column('interviewed_well')
    X_train_col_names = ["level", "lang", "tweets", "phd"]
    X_train = mp_table.get_rows(X_train_col_names)

    myRF = MyRandomForestClassifier(N=4, M=2, F=4)
    myRF.fit(X_train, y_train)
    predictions = myRF.predict(X_test)

    for i in range(0, len(predictions)):
        assert predictions[i] == y_test[i]