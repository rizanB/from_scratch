import numpy as np

from utils.logging_helper import get_logger

logger = get_logger(__name__)


def pad_input(x, filter, verbose=False) -> np.ndarray:
    """
    pads input so conv output is same size as input
    for odd filter of size k, symm_padding = (k-1)/2 when stride=1
    symm_padding is applied to both sides

    for even filter of size k, total padding = k-1
    left receives k//2 and right receives the rest

    TODO: at present, it assumes all filters are the same size,
    need to add padding separately to all filter arrays
    if arrays are of different sizes

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
        logger.debug(
            f"size of filter: {k}, total padding to add, (both sides): {symm_padding}"
        )
        padded_x = np.pad(x, symm_padding, "constant")
        logger.debug(f"padded x: {padded_x}")

    # for even filter sizes, pad asymmetrically
    else:
        # asymmetric padding,
        total_padding = int(k - 1)
        logger.debug(f"size of filter is: {k}, padding is: {total_padding}")
        left_padding = total_padding // 2

        logger.debug(f"leftp: {left_padding}, rightp: {total_padding - left_padding}")
        padded_x = np.pad(x, (left_padding, total_padding - left_padding), "constant")
        logger.debug(f"padded x: {padded_x}")
    return padded_x
