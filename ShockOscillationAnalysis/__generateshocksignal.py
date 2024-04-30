# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 02:41:58 2023

@author: super
"""
import cv2
import sys
import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from .__shocktracking import ShockTraking

def TimeCalculation(timeInSec):
    """
    Convert the given time in seconds into a formatted string representation.

    Parameters:
    - timeInSec (float): The time duration in seconds.

    Returns:
    None

    Example:
    >>> instance = SOA()
    >>> instance.TimeCalculation(3665)

    Note:
    - The function converts the time duration into hours, minutes, and seconds.
    - It prints the total run time in a human-readable format.

    """
    if timeInSec > 3600:
        timeInHr = timeInSec // 3600
        timeInMin = (timeInSec % 3600) // 60
        sec = (timeInSec % 3600) % 60
        print("Processing time: %s Hr, %s Min, %s Sec" % (round(timeInHr), round(timeInMin), round(sec)))
    elif timeInSec > 60:
        timeInMin = timeInSec // 60
        sec = timeInSec % 60
        print("Processing time: %s Min, %s Sec" % (round(timeInMin), round(sec)))
    else:
        print("Processing time: %s Sec" % round(timeInSec))
        
def GradientGenerator(img, KernalDim = 3):
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=KernalDim, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=KernalDim, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def IntegralShocktracking(SnapshotSlice,Plot,count,ShockLocation, uncertain):

    LastShockLocation = ShockLocation[-1] if ShockLocation else -1
    
    minLoc, certain, reason = ShockTraking(SnapshotSlice, 
                                           LastShockLoc = LastShockLocation, 
                                           Plot = Plot,
                                           count = count)
    ShockLocation.append(minLoc)
    if not certain: uncertain.append([count,minLoc,reason])
    return ShockLocation, uncertain
    
def GradShocktracking(GradSlice,Plot,count,ShockLocation, uncertain):
    ShockLocation.append(np.argmax(GradSlice))
    return ShockLocation, uncertain

def DarkestSpotShocktracking(SnapshotSlice,Plot,count,ShockLocation, uncertain):
    ShockLocation.append(np.argmin(SnapshotSlice))
    return ShockLocation, uncertain

def GenerateShockSignal(img, method = 'integral', 
                        signalfilter=None, review_slice_tracking = -1,
                        CheckSolutionTime = True, **kwargs):
    """
    Find the shockwave locations in a series of snapshots with optional signal processing filters.

    Parameters:
    - self: The instance of the class containing this method.
    - img (numpy.ndarray): Input array of shape (num_snapshots, height, width) representing a series of snapshots.
    - reviewInterval (list, optional): List specifying the review interval for plotting. Default is [0, 0].
    - Signalfilter (str, optional): Type of signal filter to apply ('median', 'Wiener', 'med-Wiener'). Default is None.
    - CheckSolutionTime (bool, optional): Flag to measure and display the time taken for shock tracking. Default is True.

    Returns:
    - ShockLocation (list): List of shock locations for each snapshot.
    - uncertain (list): List of uncertain shock locations with additional information.

    Examples:
    ::
        # Create an instance of the class
        instance = SOA(f,D)

        # Load a series of snapshots (assuming 'snapshots' is a NumPy array)
        shock_locations, uncertain_locations = instance.FindTheShockwaveImproved(snapshots)

    Note:
    - Ensure that 'ShockTrackingModule' is properly defined and imported.

    """
    if CheckSolutionTime: start_time = time.time()
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
        TrakingMethod = GradShocktracking
        
        
    nShoots = img.shape[0] # .................... total number of snapshots
    print('Processing the shock location ...')
    
    if ploting: 
        fig, ax = plt.subplots(figsize=(30,300))
        ax.imshow(img, cmap='gray');
        # check ploting conditions        
    for SnapshotSlice in ShockRegion:
        Plot = ploting and start <= count < end
        ShockLocation, uncertain = TrakingMethod(SnapshotSlice,Plot,count,ShockLocation,uncertain)
        count += 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(count/(nShoots/20)), int(5*count/(nShoots/20))))
        # sys.stdout.flush()
    print('')
    
    print(f'Appling {signalfilter} filter...')
    if signalfilter == 'median':
        ShockLocation = signal.medfilt(ShockLocation)
    elif signalfilter == 'Wiener':
        ShockLocation = signal.wiener(np.array(ShockLocation).astype('float64'))
    elif signalfilter == 'med-Wiener':
        ShockLocation = signal.medfilt(ShockLocation)
        ShockLocation = signal.wiener(ShockLocation.astype('float64'))
    
    # Shock tracking time
    if CheckSolutionTime:
        timeInSec =  time.time() - start_time
        TimeCalculation(timeInSec)
        
    return ShockLocation, uncertain