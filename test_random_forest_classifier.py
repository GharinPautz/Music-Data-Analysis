from mysklearn.myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier
import mysklearn.mypytable as mpt

def test_random_forest_fit():
    # interview dataset
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
    table = mpt.MyPyTable(interview_header, interview_table)
    y_train = table.get_column("interviewed_well")
    X_train = []
    # for row in interview_table:
    #     X_train.append(row[:-1])
    random_forest_classifier =  MyRandomForestClassifier(N=5, M=3, F=3)
    classification = random_forest_classifier.fit(interview_table, y_train)
    print("classification: ", classification)

    