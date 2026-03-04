import torch


def hamming_distance(l1, l2):
    """
    Compute Hamming distance between two lists of equal length.
    """
    if len(l1) != len(l2):
        raise ValueError("Hamming distance is only defined for lists of equal length.")
    return sum(a != b for a, b in zip(l1, l2))
 