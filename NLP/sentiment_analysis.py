# Sentiment analyzer

# import packages
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])
    
# define the main function
if __name__ == "__main__":
    # load the movie reviews from the corpus
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')
    
    # extract the features from the movie reviews
    features_pos = [(extract_features(movie_reviews.words(
        fileids = [f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(
        fileids = [f])), 'Negative') for f in fileids_neg]
                                                          
    # define the split for training (80%) and testing (20%)
    threshold = 0.8
    num_pos = int(threshold * len(features_pos))
    num_neg = int(threshold * len(features_neg))
    
    # create training and test datasets
    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]

    # print the number of datapoints used for training and test
    print('\nNumber of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))
    
    # train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print('\nAccuracy of the classifier:', nltk_accuracy(
        classifier, features_test))
    
    # print the top N most informative words
    N = 15
    print('\nTop ' + str(N) + ' most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i + 1) + '. ' + item[0])
        if i == N - 1:
            break
        
    # define sample reviews to be used for testing
    input_reviews = [
        'The costumes in this movie were great',
        'I think the story was terrible and the characters were very weak',
        'People say that the director of the movie is amazing',
        'This is such an idiotic movie. I will not recommend it to anyone.'
        ]
        
    # iterate through the sample reviews and predict the outputs
    print('\nMovie review predictions:')
    for review in input_reviews:
        print('\nReview:', review)
        
        # compute the probabilities for each class
        probabilities = classifier.prob_classify(
            extract_features(review.split()))
        
        # pick the maximum value among the probabilities
        predicted_sentiment = probabilities.max()
        
        # print outputs
        print('Predicted sentiment:', predicted_sentiment)
        print('Probability:', 
              round(probabilities.prob(predicted_sentiment), 2))
