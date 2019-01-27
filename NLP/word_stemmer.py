# Word stemming

# import packages
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# define input words
input_words = ['writing', 'calves', 'be', 'branded', 'horse',
               'randomize', 'possibly', 'provision', 'hospital',
               'kept', 'scratchy', 'code']
               
# create stemmer objects
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

# create a list of stemmer names for table display
stemmer_names = ['PORTER', 'LANCASTER', 'SNOWBALL']
formatted_text = '{:>16}' * (len(stemmer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names),
      '\n', '=' * 68)
# stem each word and display the output
for word in input_words:
    output = [word, 
              porter.stem(word),
              lancaster.stem(word),
              snowball.stem(word)]
    print(formatted_text.format(*output))
