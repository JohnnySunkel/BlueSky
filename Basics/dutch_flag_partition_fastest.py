def dutch_flag_partition(pivot_index, A):
    pivot = A[pivot_index]
    # Keep the following invariants during partitioning:
    # bottom_group: A[:smaller].
    # middle_group: A[smaller:equal].
    # unclassified_group: A[equal:larger].
    # top_group: A[larger:].
    smaller, equal, larger = 0, 0, len(A)
    # Keep iterating as long as there is an unclassified element.
    while equal < larger:
        # A[equal] is the incoming unclassified element.
        if A[equal] < pivot:
            A[smaller], A[equal] = A[equal], A[smaller]
            smaller, equal = smaller + 1, equal + 1
        elif A[equal] == pivot:
            equal += 1
        else:
            # A[equal] > pivot.
            larger -= 1
            A[equal], A[larger] = A[larger], A[equal]
