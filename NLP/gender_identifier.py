# Gender identifier

# import packages
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

# define a function to extract the last N letters from the input word
# this will act as our "feature"
def extract_features(word, N = 2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}
            
# define the main function
if __name__ == "__main__":
    # create training data using labeled names available in NLTK
    male_list = [(name, 'male') for name in names.words('male.txt')]
    female_list = [(name, 'female') for name in names.words('female.txt')]
    data = (male_list + female_list)
    
    # seed the random number generator
    random.seed(5)
    
    # shuffle the data
    random.shuffle(data)
    
    # create test data
    input_names = ['Alexander', 'Danielle', 'David', 'Cheryl']

    # define the percentage of data that will be used 
    # for training and test datasets
    num_train = int(0.8 * len(data))
    
    # iterate through different values of N to compare accuracy
    for i in range(1, 6):
        print('\nNumber of end letters:', i)
        features = [(extract_features(n, i), gender) for (n, gender) in data]
                    
        # separate the data into training and test datasets
        train_data, test_data = features[:num_train], features[num_train:]

        # build a Naive Bayes classifier using the training data
        classifier = NaiveBayesClassifier.train(train_data)
        
        # compute the accuracy of the classifier
        accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)
        print('Accuracy = ' + str(accuracy) + '%')
        
        # predict outputs for the test names
        for name in input_names:
            print(name, '==>', classifier.classify(extract_features(name, i)))
