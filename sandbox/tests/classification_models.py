import unittest


class TestClassifiers(unittest.TestCase):

    def test_decision_tree(self):
        from sklearn import tree
        #[height, weight, shoe_size]
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, Y)

        prediction = clf.predict([[190, 70, 43]])
        print prediction

    def test_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        #[height, weight, shoe_size]
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

        clf = RandomForestClassifier(n_estimators=2)
        clf = clf.fit(X, Y)

        prediction = clf.predict([[190, 70, 43]])
        print prediction

    def test_k_nearest_neighbour(self):
        from sklearn.neighbors import KNeighborsClassifier
        #[height, weight, shoe_size]
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, Y)

        prediction = neigh.predict([[190, 70, 43]])
        print prediction

    def test_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        #[height, weight, shoe_size]
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

        neigh = LogisticRegression()
        neigh.fit(X, Y)

        prediction = neigh.predict([[190, 70, 43]])
        print prediction

    def test_naive_bayes(self):
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
        Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

        gnb = gnb.fit(X, Y)

        prediction = gnb.predict([[190, 70, 43]])
        print prediction

    def test_artificial_neural_network(self):
        # To do the following you need to run command: pip install scikit-neuralnetwork
        from sknn.mlp import Classifier, Layer
        import numpy as np
        X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])
        Y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male'])
        nn = Classifier(
            layers=[
                Layer("Maxout", units=100, pieces=2),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=25)
        nn.fit(X, Y)

        prediction = nn.predict([[190, 70, 43]])
        print prediction

    def test_ann(self):
        from pybrain.datasets.classification import ClassificationDataSet
        # below line can be replaced with the algorithm of choice e.g.
        # from pybrain.optimization.hillclimber import HillClimber
        from pybrain.optimization.populationbased.ga import GA
        from pybrain.tools.shortcuts import buildNetwork

        # create XOR dataset
        d = ClassificationDataSet(2)
        d.addSample([181, 80], [1])
        d.addSample([177, 70], [1])
        d.addSample([160, 60], [0])
        d.addSample([154, 54], [0])
        d.setField('class', [ [0.],[1.],[1.],[0.]])

        nn = buildNetwork(2, 3, 1)
        # d.evaluateModuleMSE takes nn as its first and only argument
        ga = GA(d.evaluateModuleMSE, nn, minimize=True)
        for i in range(100):
            nn = ga.learn(0)[0]

        print nn.activate([181, 80])


    def test_decision_tree_load_data(self):
        # Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
        import pandas as pd
        from sklearn import tree
        from sklearn.preprocessing import LabelEncoder

        # Import the adult.txt file into Python
        data = pd.read_csv('adults.txt', sep=',')

        # DO NOT WORRY ABOUT THE FOLLOWING 2 LINES OF CODE
        # Convert the string labels to numeric labels
        for label in ['race', 'occupation']:
            data[label] = LabelEncoder().fit_transform(data[label])

        # Take the fields of interest and plug them into variable X
        X = data[['race', 'hours_per_week', 'occupation']]

        # Make sure to provide the corresponding truth value
        Y = data['sex'].values.tolist()

        # Instantiate the classifier
        clf = tree.DecisionTreeClassifier()

        # Train the classifier using the data
        clf = clf.fit(X, Y)

        # White (race), 40 (Hours per week), Adm-clerical (occupation)
        test_sample = [[4, 39, 1]]

        # Predict the output
        prediction = clf.predict(test_sample)
        print prediction

    def test_decision_tree_load_data_train_test(self):
        import pandas as pd
        from sklearn import tree
        from sklearn.preprocessing import LabelEncoder
        from sklearn.cross_validation import train_test_split

        # Import the adult.txt file into Python
        data = pd.read_csv('adults.txt', sep=',')

        # DO NOT WORRY ABOUT THE FOLLOWING 2 LINES OF CODE
        # Convert the string labels to numeric labels
        for label in ['race', 'occupation']:
            data[label] = LabelEncoder().fit_transform(data[label])

        # Take the fields of interest and plug them into variable X
        X = data[['race', 'hours_per_week', 'occupation']]
        # Make sure to provide the corresponding truth value
        Y = data['sex'].values.tolist()

        # Split the data into test and training (30% for test)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        # Instantiate the classifier
        clf = tree.DecisionTreeClassifier()

        # Train the classifier using the train data
        clf = clf.fit(X_train, Y_train)

        # Validate the classifier
        accuracy = clf.score(X_test, Y_test)
        print 'Accuracy: ' + str(accuracy)


    def test_random_forest_load_data_cv(self):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.cross_validation import train_test_split

        data = pd.read_csv('adults.txt', sep=',')
        lb = LabelEncoder()
        data['race'] = lb.fit_transform(data['race'])
        data['hours_per_week'] = lb.fit_transform(data['hours_per_week'])
        data['occupation'] = lb.fit_transform(data['occupation'])
        #data['sex'] = lb.fit_transform(data['sex'])

        X = data[['race', 'hours_per_week', 'occupation']]
        Y = data['sex'].values.tolist()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        clf = RandomForestClassifier(n_estimators=1000)
        clf = clf.fit(X_train, Y_train)
        print clf.predict(X_test)[:80]

        print clf.score(X_test, Y_test)