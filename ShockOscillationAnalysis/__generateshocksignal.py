# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 02:41:58 2023

@author: super
"""
import cv2
import sys
import time
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
        print("Total run time: %s Hr, %s Min, %s Sec" % (timeInHr, timeInMin, round(sec)))
    elif timeInSec > 60:
        timeInMin = timeInSec // 60
        sec = timeInSec % 60
        print("Total run time: %s Min, %s Sec" % (timeInMin, round(sec)))
    else:
        print("Total run time: %s Sec" % round(timeInSec))

def GenerateShockSignal(img, reviewInterval = [0,0], Signalfilter=None, CheckSolutionTime = True):
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
    reviewInterval.sort(); start, end = reviewInterval
    plotingInterval = abs(end-start)
    ploting = plotingInterval > 0
    
    # check if the image on grayscale or not and convert if not
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    
    nShoots = img.shape[0] # .................... total number of snapshots
    print('Processing the shock location ...')
    
    if ploting: 
        fig, ax = plt.subplots(figsize=(30,300))
        ax.imshow(img, cmap='gray');
        # check ploting conditions
     
    for SnapshotSlice in img:
        Plot = ploting and start <= count < end
        
        LastShockLocation = ShockLocation[-1] if ShockLocation else -1
        
        minLoc, certain, reason = ShockTraking(SnapshotSlice, 
                                               LastShockLoc = LastShockLocation, 
                                               Plot = Plot,
                                               count = count)
        ShockLocation.append(minLoc)
        if not certain: uncertain.append([count,minLoc,reason])
        count += 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(count/(nShoots/20)), int(5*count/(nShoots/20))))
        sys.stdout.flush()
    print('')
        
    if Signalfilter == 'median':
        print('Appling median filter...')
        ShockLocation = signal.medfilt(ShockLocation)
    elif Signalfilter == 'Wiener':
        print('Appling Wiener filter...')
        ShockLocation = signal.wiener(ShockLocation.astype('float64')+ 1e-10)
    elif Signalfilter == 'med-Wiener':
        print('Appling med-Wiener filter...')
        ShockLocation = signal.medfilt(ShockLocation)
        ShockLocation = signal.wiener(ShockLocation.astype('float64')+ 1e-10)
    
    # Shock tracking time
    if CheckSolutionTime:
        timeInSec =  time.time() - start_time
        TimeCalculation(timeInSec)
        
    return ShockLocation, uncertain