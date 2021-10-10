import numpy as np


# copied from essentia/src/python/essentia/pytools/spectral.py
def nsgcq_overlap_add(cq):
    """Frame-wise invertible Constant-Q synthesis.
    This function performs the overlap-add process of the CQ frames obtained by nsgcq_gram.
    The output of this algorithm may be used for visualization purposes.
    Note: It is not possible to perform a perfect reconstruction from the overlapped version of the CQ data. 

    Args:
        (list of 2D complex arrays): Time / frequency complex matrices representing the NSGCQ `constantq` coefficients for each `frameSize // 2` samples jump.
    Returns:
        (2D complex array): The overlapped version of the Constant-Q. 
    """
    frameNum = cq.shape[0]
    cqChannels = cq.shape[1]
    cqSize = cq.shape[2]

    hopSize = cqSize // 2
    timeStamps = (frameNum + 1) * hopSize

    index = np.arange(cqSize)

    cqOverlap = np.zeros(
        [cqChannels, timeStamps], dtype=cq.dtype)

    # Overlap-add.
    for jj in range(frameNum):
        cqOverlap[:, jj * hopSize + index] += cq[jj, :, :]

    return cqOverlap[:, hopSize:]
