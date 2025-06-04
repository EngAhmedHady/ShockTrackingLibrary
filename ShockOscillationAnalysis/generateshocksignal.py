# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 02:41:58 2023

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from .shocktracking import ShockTraking
from .decorators import calculate_running_time

def GradientGenerator(img:np.ndarray[int], KernalDim: int = 3)-> np.ndarray[int]:
    """
    Generate the gradient magnitude of an image using Sobel operators.
    This function applies Sobel operators to compute the gradient magnitude of the input image.
    The `KernalDim` parameter specifies the dimension of the Sobel kernel used for gradient calculation.

    Parameters:
        - **img (numpy.ndarray)**: Input image (grayscale).
        - **KernalDim (int)**: Dimension of the Sobel kernel. Default is 3.

    Returns:
        numpy.ndarray: Gradient magnitude of the input image.

    Example:
        >>> gradient = GradientGenerator(image, KernalDim=3)

    .. note::
        - The input image should be in grayscale.
        - The function returns the gradient magnitude of the input image.
    """
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=KernalDim, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=KernalDim, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def IntegralShocktracking(SnapshotSlice: list[int], Plot: bool, count: int,
                          ShockLocation: float, uncertain: bool) -> tuple[float, bool]:
    """
    Perform shock tracking based on integral method discriped in https://dx.doi.org/10.2139/ssrn.4797840.

    Parameters:
        - **SnapshotSlice (list)**: snapshot slice in grayscale where shock is tracked.
        - **Plot (bool)**: Whether to plot the slice tracking process info.
        - **count (int)**: Current snapshot/image number.
        - **ShockLocation (float)**: Location of the shock from the previous iteration.
        - **uncertain (bool)**: Flag indicating uncertainty.

    Returns:
        tuple: A tuple containing:
            - float: Updated shock location.
            - bool: Flag indicating uncertainty.

    Example:
        >>> shock_loc, is_uncertain = IntegralShocktracking(slice_values, Plot=True, count=10, ShockLocation=0, uncertain=False)

        It updates the shock location and determines if there's uncertainty in the tracking process.
    """

    LastShockLocation = ShockLocation[-1] if ShockLocation else -1

    minLoc, certain, reason = ShockTraking(SnapshotSlice,
                                           LastShockLoc=LastShockLocation,
                                           Plot=Plot,
                                           count=count)
    ShockLocation.append(minLoc)
    if not certain: uncertain.append([count, minLoc, reason])
    return ShockLocation, uncertain

def GradShocktracking(GradSlice,Plot,count,ShockLocation, uncertain):
    """
    Perform shock tracking based on gradient values.

    Parameters:
        - **GradSlice (numpy.ndarray)**: Array containing gradient values for shock tracking.
        - **Plot (bool)**: Flag indicating whether to generate plots of the results.
        - **count (int)**: Current iteration count.
        - **ShockLocation (list)**: List containing the shock location from previous iterations.
        - **uncertain (bool)**: Flag indicating uncertainty in the shock tracking process.

    Returns:
        tuple: A tuple containing:
            - list: Updated shock location.
            - bool: Flag indicating uncertainty.

    Example:
        >>> shock_loc, is_uncertain = GradShocktracking(grad_values, Plot=True, count=10, ShockLocation=[0], uncertain=False)

        This function performs shock tracking based on gradient values extracted from a slice of data. It updates the shock location and determines if there's uncertainty in the tracking process.

    """
    ShockLocation.append(np.argmax(GradSlice))
    return ShockLocation, uncertain

def DarkestSpotShocktracking(SnapshotSlice: list[int],
                             Plot: bool, count: int, ShockLocation: list[float], uncertain: list[tuple[int, float]]) -> tuple[list,list]:
    """
    Perform shock tracking based on the location of the darkest spot in a snapshot slice.

    This function identifies the position of the darkest spot in a given snapshot slice and
    appends its index to the list of shock locations. Optionally, it also records any uncertainty
    regarding the shock location.

    Parameters:
        - **SnapshotSlice (list[int])**: The snapshot slice to be analyzed for shock tracking.
        - **Plot (bool)**: A flag indicating whether to generate plots during shock tracking.
        - **count (int)**: The count or index of the current snapshot slice.
        - **ShockLocation(list)**: A list containing the indices of previously detected shock locations.
        - **uncertain (list)**: A list to store any uncertain shock locations.

    Returns:
        - A tuple containing the updated ShockLocation list and the uncertain list.

    """
    ShockLocation.append(np.argmin(SnapshotSlice))
    return ShockLocation, uncertain


@calculate_running_time
def GenerateShockSignal(img, method='integral',
                        signalfilter=None, review_slice_tracking = -1,
                        **kwargs) -> tuple[list, list]:
    """
    Find the shockwave locations in a series of snapshots with optional signal processing filters.

    Parameters:
        - **img (numpy.ndarray)**: Input array of shape (num_snapshots, height, width) representing a series of snapshots.
        - **reviewInterval (list, optional)**: List specifying the review interval for plotting. Default is [0, 0].
        - **Signalfilter (str, optional)**: Type of signal filter to apply ('median', 'Wiener', 'med-Wiener'). Default is None.
        - **CheckSolutionTime (bool, optional)**: Flag to measure and display the time taken for shock tracking. Default is True.

    Returns:
        - ShockLocation (list): List of shock locations for each snapshot.
        - uncertain (list): List of uncertain shock locations with additional information.

    Examples:
        >>> # Create an instance of the class
        >>> instance = SOA(f,D)
        >>> # Load a series of snapshots (assuming 'snapshots' is a NumPy array)
        >>> shock_locations, uncertain_locations = instance.FindTheShockwaveImproved(snapshots)

    .. note:
        - Ensure that 'ShockTrackingModule' is properly defined and imported.

    """
    # Initiating Variables
    ShockLocation = [] # ........................... set of shock locations
    uncertain = [] # set of uncertain shock locations [snapshot value, uncertain location]
    count = 0 # ................................ Processed snapshot counter

    # check ploting conditions
    if hasattr(review_slice_tracking, "__len__") and len(review_slice_tracking) == 2:
        review_slice_tracking.sort(); start, end = review_slice_tracking
        plotingInterval = abs(end-start)
        ploting = plotingInterval > 0
    elif not hasattr(review_slice_tracking, "__len__") and review_slice_tracking> -1:
        start = review_slice_tracking; end = review_slice_tracking + 1
        plotingInterval = 1
        ploting = plotingInterval > 0

    # check if the image on grayscale or not and convert if not
    ShockRegion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

    if method == 'integral':
        TrakingMethod = IntegralShocktracking
    elif method == 'darkest_spot':
        TrakingMethod = DarkestSpotShocktracking
    elif method == 'maxGrad':
        ksize = kwargs.get('ksize', 3)
        ShockRegion = GradientGenerator(ShockRegion, KernalDim = ksize)
        if ploting and plotingInterval > 1:
            fig, ax = plt.subplots(figsize=(10,100))
            ax.imshow(ShockRegion[start:end], cmap = 'gray')
        TrakingMethod = GradShocktracking


    nShoots = img.shape[0] # .................... total number of snapshots
    print(f'Processing the shock location using {method} method...')

    for SnapshotSlice in ShockRegion:
        Plot = ploting and start <= count < end
        ShockLocation, uncertain = TrakingMethod(SnapshotSlice,Plot,count,ShockLocation,uncertain)
        count += 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(count/(nShoots/20)), int(5*count/(nShoots/20))))
        sys.stdout.flush()
    print('')

    print(f'Appling {signalfilter} filter...')
    if signalfilter == 'median':
        ShockLocation = signal.medfilt(ShockLocation)
    elif signalfilter == 'Wiener':
        ShockLocation = signal.wiener(np.array(ShockLocation).astype('float64'))
    elif signalfilter == 'med-Wiener':
        ShockLocation = signal.medfilt(ShockLocation)
        ShockLocation = signal.wiener(ShockLocation.astype('float64'))

    return ShockLocation, uncertain