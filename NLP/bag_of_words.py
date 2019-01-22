# Bag of words model

# import packages and functions
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from text_chunker import chunker

# read the data from the Brown corpus
input_data = ' '.join(brown.words()[:5400])

# define the number of words in each chunk
chunk_size = 800

# divide the input text into chunks
text_chunks = chunker(input_data, chunk_size)

# convert the chunks into dictionary items
chunks = []
for count, chunk in enumerate(text_chunks):
    d = {'index': count, 'text': chunk}
    chunks.append(d)
    
# extract the document term matrix
count_vectorizer = CountVectorizer(min_df = 7, max_df = 20)
document_term_matrix = count_vectorizer.fit_transform([chunk['text']
                                                       for chunk in chunks])

# extract the vocabulary and display it
vocabulary = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary:\n", vocabulary)

# generate names for the chunks
chunk_names = []
for i in range(len(text_chunks)):
    chunk_names.append('Chunk-' + str(i + 1))
    
# print the document term matrix
print("\nDocument term matrix:")
formatted_text = '{:>12}' * (len(chunk_names) + 1)
print('\n', formatted_text.format('Word', *chunk_names), '\n')
for word, item in zip(vocabulary, document_term_matrix.T):
    # 'item' is a 'csr_matrix' data structure
    output = [word] + [str(freq) for freq in item.data]
    print(formatted_text.format(*output))
