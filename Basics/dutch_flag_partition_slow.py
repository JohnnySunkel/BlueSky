def dutch_flag_partition(pivot_index, A):
    pivot = A[pivot_index]
    # First pass: group the elements smaller than the pivot.
    for i in range(len(A)):
        # Look for smaller elements.
        for j in range(i + 1, len(A)):
            if A[j] < pivot:
                A[i], A[j] = A[j], A[i]
                break
    # Second pass: group the elements larger than the pivot.
    for i in reversed(range(len(A))):
        # Look for larger elements. Stop when we reach an element
        # less than the pivot, since the first pass has already
        # moved them to the start of A.
        for j in reversed(range(i)):
            if A[j] > pivot:
                A[i], A[j] = A[j], A[i]
                break
