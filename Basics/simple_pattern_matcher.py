def patternMatcher(pattern, text):
    # track the start of each attempt
    starti = 0
    # index into text
    i = 0
    # index into pattern
    j = 0
    match = False
    stop = False
    while not match and not stop:
        if text[i] == pattern[j]:
            i = i + 1
            j = j + 1
        else:
            # shift to right
            starti = starti + 1
            i = starti
            j = 0
            
        if j == len(pattern):
            match = True
        else:
            if i == len(text):
                stop = True
                
    if match:
        return i - j
    else:
        return -1
