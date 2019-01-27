# Lemmatization

# import packages
from nltk.stem import WordNetLemmatizer

# define input words
input_words = ['writing', 'calves', 'be', 'branded', 'horse', 
               'randomize', 'possibly', 'provision', 'hospital',
               'kept', 'scratchy', 'code']
               
# create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# create a list of lemmatizer names for table display
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *lemmatizer_names),
      '\n', '=' * 75)
# lemmatize each word and display the output
for word in input_words:
    output = [word,
              lemmatizer.lemmatize(word, pos = 'n'),
              lemmatizer.lemmatize(word, pos = 'v')]
    print(formatted_text.format(*output))
