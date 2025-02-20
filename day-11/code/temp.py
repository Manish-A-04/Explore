
def quicksort(arr):
    """
    Sorts an array using the quicksort algorithm.

    Args:
        arr: The array to sort.

    Returns:
        The sorted array.
    """
    if len(arr) < 2:
        return arr  # Base case: already sorted

    pivot = arr[0]  # Choose the first element as the pivot
    less = [i for i in arr[1:] if i <= pivot]
    greater = [i for i in arr[1:] if i > pivot]

    return quicksort(less) + [pivot] + quicksort(greater)
