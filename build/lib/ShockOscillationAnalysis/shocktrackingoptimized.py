# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:23:35 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
import matplotlib.pyplot as plt

def findlocalminimums(Reference, slice_pixels, plot = False, ax = [], alpha = 0, Last_Loc = -1):
    """
   Find local minimums in a given slice of pixels.

   Parameters:
       - **Reference (float)**: Reference value for comparison.
       - **slice_pixels (list[float])**: List of pixel values in the slice.
   
   Keyword Arguments:
       - **plot (bool)**: Whether to plot the local minimums. Defaults to False.
       - **ax (matplotlib.axes.Axes)**: Axes object for plotting. Required if plot=True.
       - **alpha (float)**: Threshold for considering a local minimum. Defaults to 0.
       - **Last_Loc (int)**: Last location of the local minimum. Defaults to -1.

   Returns:
       - **tuple**: Tuple containing local minimums, areas, and axes object.
   """
    Pixels = len(slice_pixels)
    Loc_shift = abs(Pixels - Last_Loc) + 1 if Last_Loc > -1 else  0
    Local_Mins_Set = []; MinimumPoint = min(slice_pixels)
    localmin = []; LocMinI = []; AeraSet = []; 
    for pixel in range(Pixels-1,-1,-1):
        if slice_pixels[pixel] < Reference:
            # find the local minimums and store illumination values and location
            localmin.append(slice_pixels[pixel]); LocMinI.append(pixel+Loc_shift)
            # find open local minimum at the end of the slice
            if pixel == 0 and len(localmin) >1: 
                SetMinPoint = min(localmin)
                if plot: ax.fill_between(LocMinI, localmin, Reference , alpha=0.5)
                if SetMinPoint/MinimumPoint > alpha:
                    A = abs(np.trapz(Reference-localmin))
                    AeraSet.append(A)
                    Local_Mins_Set.append([LocMinI,localmin])
                localmin = []; LocMinI = []
        # bounded local minimum
        elif slice_pixels[pixel] >= Reference and len(localmin) > 1: 
            SetMinPoint = min(localmin)
            if plot: ax.fill_between(LocMinI, localmin, Reference , alpha=0.5)                       
            if SetMinPoint/MinimumPoint > alpha:
                A = abs(np.trapz(Reference-localmin))
                AeraSet.append(A)
                Local_Mins_Set.append([LocMinI,localmin])
            localmin = []; LocMinI = []
        else: localmin = [];  LocMinI = []
        
        if Last_Loc > -1: Loc_shift += 2
    return Local_Mins_Set, AeraSet, ax
        
def ShockTraking(SnapshotSlice, LastShockLoc = -1, Plot = False, count = -1, alpha = 0.3):
    """
    Process a given slice to track shock waves and determine the shock location.
    
    Parameters:
        - **SnapshotSlice (numpy.ndarray)**: A slice of the image to process.
        - **LastShockLoc (int, optional)**: The location of the last shock. Default is -1.
        - **Plot (bool, optional)**: Whether to plot the slice illumination values with locations and average line. Default is False.
        - **count (int, optional)**: Counter for plotting. Default is -1.
    
    Returns:
        tuple: A tuple containing:
            - minLoc (float): The determined location of the shock wave.
            - certainLoc (bool): True if the shock location is certain, False otherwise.
            - reason (str): A string providing the reason for uncertainty if certainLoc is False.
    
    Example:
        >>> instance = SOA(f)
        >>> result = instance.ShockTraking(SnapshotSlice, LastShockLoc=10, Plot=True, count=1)
        >>> print(result)
    
    .. important::
        this function is still under testing
    
    """
    
    # Start processing the slice
    avg = np.mean(SnapshotSlice) # ...... Average illumination on the slice
    ax = []
        
    if Plot: # to plot slice illumination values with location and Avg. line
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(SnapshotSlice); ax.axhline(avg,linestyle = ':');
        ax.plot(SnapshotSlice); 
        # ax.plot(AvgLocation,AvgIllumination,linestyle = '-.');
    
    Local_Mins_Set, aera_set, ax = findlocalminimums(avg, 
                                                    SnapshotSlice, 
                                                    plot = Plot, ax = ax, 
                                                    alpha = 0.3)
    
    max_area = max(aera_set)
    ShockRegion = Local_Mins_Set[aera_set.index(max_area)]
    
    
    # check if there is more than one valley in the local minimum
    LocMinAvg = np.mean(ShockRegion[1])
    if Plot: 
        ax.plot([ShockRegion[0][0]+5,ShockRegion[0][-1]-5],[LocMinAvg,LocMinAvg],'-.r')
        print(ShockRegion, LocMinAvg)
    
    sub_local_min_sets, sub_area_set, ax = findlocalminimums(LocMinAvg, 
                                                            ShockRegion[1], 
                                                            plot = Plot,
                                                            Last_Loc = ShockRegion[0][-1],
                                                            ax = ax)
    n = len(sub_area_set)        
    # uncertainity calculation
    certainLoc = True;  reason = ''
    
    # if there is more than one valley in the local minimum, 
    # the closest to the preivous location will be choosen
    if n > 1 and LastShockLoc > -1:
        MinDis = len(SnapshotSlice); 
        for SubLocalMinSet in sub_local_min_sets:
            # check the location of the minimum illumination point from last snapshot location and choose the closest
            minValue = min(SubLocalMinSet[1]) # ........ find minimam illumination in the sub-set
            minLoc = SubLocalMinSet[1].index(minValue) # find the location of the minimam illumination in the sub-set
            
            Distance = abs(LastShockLoc-SubLocalMinSet[0][minLoc])                
            if Distance < MinDis: 
                MinDis = Distance;  ShockRegion = SubLocalMinSet
            if Plot: ax.fill_between(ShockRegion[0], ShockRegion[1],avg , hatch='\\')
        
    elif n > 1 and LastShockLoc == -1: 
        n = 1; 
        certainLoc = False
        reason = 'First pexil slice, No shock location history'
    
    
    # Find the middel of the shock wave as middle point of RMS
    LocMinRMS = avg-np.sqrt(np.mean(np.array(avg-ShockRegion[1])**2))
    if Plot: 
        ax.plot([ShockRegion[0][0]+5,ShockRegion[0][-1]-5],[LocMinRMS,LocMinRMS],'-.k') 
        ax.fill_between(ShockRegion[0], ShockRegion[1],avg , hatch='///') 
        
    shockLoc = [];
    for elment in range(len(ShockRegion[1])):
        if ShockRegion[1][elment] <= LocMinRMS: shockLoc.append(ShockRegion[0][elment])
    minLoc = np.mean(shockLoc) 
    
    if Plot:
        ax.axvline(minLoc, linestyle = '--', color = 'b')
        if count > -1: ax.set_title(count)
        if LastShockLoc > -1:
            ax.axvline(LastShockLoc,linestyle = '--',color = 'orange')  
    
    for area in aera_set:
        Ra = area/max_area
        if Ra > 0.6 and Ra < 1 and certainLoc:
            certainLoc = False;   reason = 'Almost equal Valleys'
    
    if n > 1 and certainLoc:
        sub_max_area = max(sub_area_set)
        for Area in sub_area_set:
            if sub_max_area > 0: Ra = Area/sub_max_area
            if Ra > 0.5 and Ra < 1 and certainLoc: certainLoc = False; reason = 'Almost equal sub-Valleys'   
            if sub_max_area !=  abs(np.trapz(LocMinAvg-ShockRegion[1])) and certainLoc: 
                certainLoc = False; reason = 'different sub-Valleys than smallest'
    
    if (not certainLoc) and Plot: 
        ax.text(0.99, 0.99, 'uncertain: '+ reason,
                ha = 'right', va ='top', transform = ax.transAxes,
                color = 'red', fontsize=14)
    return minLoc, certainLoc, reason