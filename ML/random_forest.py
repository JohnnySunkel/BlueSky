# Random Forest and Extremely Random Forest classifiers

# import the packages and functions we'll need to build the
# random forest and extremely random forest classifiers
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier

# define an argument parser that can take the classifier type
# as an input parameter
def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Classify data using \
                                     Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest = 'classifier_type',
                        required = True, choices = ['rf', 'erf'],
                        help = "Type of classifier to use; can be \
                        either 'rf' or 'erf'")
    return parser
    
# define the main function and parse the input arguments
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    
    # load input data
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter = ',')
    X, y = data[:, :-1], data[:, -1]

    # separate the data into 3 classes based on their labels
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])
    
    # visualize the input data
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s = 75, facecolors = 'white',
                edgecolors = 'black', linewidth = 1, marker = 's')
    plt.scatter(class_1[:, 0], class_1[:, 1], s = 75, facecolors = 'white',
                edgecolors = 'black', linewidth = 1, marker = 'o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s = 75, facecolors = 'white',
                edgecolors = 'black', linewidth = 1, marker = '^')
    plt.title('Input data')
    
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
        y, test_size = 0.25, random_state = 5)
    
    # define the parameters for the model
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    # depending on the input parameter we either construct a 
    # random forest classifier or an extremely randon forest classifier
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(** params)
    else:
        classifier = ExtraTreesClassifier(** params)
        
    # train and visualize the classifier
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train)
    
    # compute and visualize output for the test dataset
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test)
    
    # evaluate classifier performance
    class_names = ['Class_0', 'Class_1', 'Class_2']
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train),
                                target_names = class_names))
    print("#" * 40 + "\n")
    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred,
                                target_names = class_names))
    print("#" * 40 + "\n")
