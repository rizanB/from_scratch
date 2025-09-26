import numpy as np
from utils.printv import printv


def pad_input(x, filter, verbose=False) -> np.ndarray:
    """
    pads input so conv output is same size as input
    for odd filter of size k, symm_padding = (k-1)/2 when stride=1
    symm_padding is applied to both sides

    for even filter of size k, total padding = k-1
    left receives k//2 and right receives the rest

    TODO: at present, it assumes all filters are the same size, need to add padding separately to all filter arrays if arrays are of different sizes

    Args:
    x (np.ndarray): input
    filter (np.ndarray): filter array
    verbose: flag

    Returns:
    padded_x (np.ndarray): padded input
    """

    padded_x = None
    k = len(filter)

    # odd filter
    if (k % 2) != 0:
        # symmetric padding, applied to both sides of input
        symm_padding = int((k - 1) / 2)
        printv(
            f"size of filter is: {k}, total padding to add, on both sides is: {symm_padding}",
            verbose,
        )
        padded_x = np.pad(x, symm_padding, "constant")
        printv(f"padded x: {padded_x}", verbose)

    # for even filter sizes, pad asymmetrically
    else:
        # asymmetric padding,
        total_padding = int(k - 1)
        printv(f"size of filter is: {k}, padding is: {total_padding}", verbose)
        left_padding = total_padding // 2

        printv(
            f"left padding: {left_padding}, right padding: {total_padding - left_padding}",
            verbose,
        )
        padded_x = np.pad(x, (left_padding, total_padding - left_padding), "constant")
        print(f"padded x: {padded_x}", verbose)
    return padded_x
