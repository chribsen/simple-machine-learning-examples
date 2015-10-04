import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


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
clf = LogisticRegression()

# Train the classifier using the train data
clf = clf.fit(X_train, Y_train)

# Validate the classifier
accuracy = clf.score(X_test, Y_test)
print 'Accuracy: ' + str(accuracy)

# Make a confusion matrix
prediction = clf.predict(X_test)

cm = confusion_matrix(prediction, Y_test)
print cm