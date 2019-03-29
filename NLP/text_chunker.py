# Text chunking

# import packages
import numpy as np
from nltk.corpus import brown

# define a function to split the input text into chunks, where
# each chunk contains N words
def chunker(input_data, N):
    input_words = input_data.split(' ')
    output = []

    cur_chunk = []
    count = 0
    for word in input_words:
        cur_chunk.append(word)
        count += 1
        if count == N:
            output.append(' '.join(cur_chunk))
            count, cur_chunk = 0, []

    output.append(' '.join(cur_chunk))
    
    return output
    
# define the main function
if __name__ == "__main__":
    # read the first 12000 words from the Brown corpus
    input_data = ' '.join(brown.words()[:12000])
    
    # define the number of words in each chunk
    chunk_size = 700
    
    # divide the input text into chunks and display the output
    chunks = chunker(input_data, chunk_size)
    print('\nNumber of text chunks =', len(chunks), '\n')
    for i, chunk in enumerate(chunks):
        print('Chunk', i + 1, '==>', chunk[:50])
