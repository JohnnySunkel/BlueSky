# Class imbalance

# import the packages and functions we'll need 
# to deal with class imbalance
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# load the input data
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter = ',')
X, y = data[:, :-1], data[:, -1]

# separate the input data into 2 classes based on their labels
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

# visualize the input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s = 75, facecolors = 'black',
    edgecolors = 'black', linewidth = 1, marker = 'x')
plt.scatter(class_1[:, 0], class_1[:, 1], s = 75, facecolors = 'white',
    edgecolors = 'black', linewidth = 1, marker = 'o')
plt.title('Input data')

# split the data into training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
    y, test_size = 0.25, random_state = 5)

# define the parameters for an Extremely Random Forest classifier
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0,
          'class_weight': 'balanced'}
        
# build the classifier
classifier = ExtraTreesClassifier(** params)

# train the classifier
classifier.fit(X_train, y_train)

# visualize the classifier
visualize_classifier(classifier, X_train, y_train)

# predict and visualize output for the test dataset
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test)

# evaluate classifier performance
class_names = ['Class_0', 'Class_1']
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train),
                            target_names = class_names))
print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred,
                            target_names = class_names))
print("#" * 40 + "\n")
plt.show()
