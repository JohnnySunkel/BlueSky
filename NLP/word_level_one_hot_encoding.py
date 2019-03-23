import numpy as np

# Initial data. One entry per sample (in this example, 
# a sample is a sentence, but it could be an entire document).
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Build an index of all tokens in the data
token_index = {}
for sample in samples:
    # Tokenize the samples via the split method. In real life,
    # you would also strip punctuation and special characters
    # from the samples.
    for word in sample.split():
        if word not in token_index:
            # Assign a unique index to each unique word. 
            # Note that you do not attribute index 0 to anything.
            token_index[word] = len(token_index) + 1
    
# Vectorize the samples. You'll only consider the first
# max_length words in each sample.         
max_length = 10
    
# Store the results
results = np.zeros(shape = (len(samples),
                            max_length,
                            max(token_index.values()) + 1))
    
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
