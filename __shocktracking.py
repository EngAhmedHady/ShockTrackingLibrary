# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:27:54 2023

@author: Ahmed H. Hanfy
"""
import numpy as np
import matplotlib.pyplot as plt


        
def ShockTraking(SnapshotSlice, LastShockLoc = -1, Plot = False, count = -1):
    """
    Process a given slice to track shock waves and determine the shock location.
    
    Inputs:
    - SnapshotSlice (numpy.ndarray): A slice of the image to process.
    - LastShockLoc (int, optional): The location of the last shock. Default is -1.
    - Plot (bool, optional): Whether to plot the slice illumination values with locations and average line.
                            Default is False.
    - count (int, optional): Counter for plotting. Default is -1.
    
    Outputs:
    tuple: A tuple containing:
    - minLoc (float): The determined location of the shock wave.
    - certainLoc (bool): True if the shock location is certain, False otherwise.
    - reason (str): A string providing the reason for uncertainty if certainLoc is False.
    
    Example:
    >>> instance = SOA(f)
    >>> result = instance.ShockTraking(SnapshotSlice, LastShockLoc=10, Plot=True, count=1)
    >>> print(result)
    
    Note:
    - The function processes the illumination values of a given image slice to track shock waves.
    - It returns the determined shock location, certainty status, and a reason for uncertainty.
    
    """
    
    # Start processing the slice
    avg = np.mean(SnapshotSlice) # ...... Average illumination on the slice   
    MinimumPoint = min(SnapshotSlice) # ........... minimum (darkest) point
    
    if Plot: # to plot slice illumination values with location and Avg. line
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(SnapshotSlice); ax.axhline(avg,linestyle = ':');
        ax.plot(SnapshotSlice); 
        # ax.plot(AvgLocation,AvgIllumination,linestyle = '-.');
    
    # Initiating Variables 
    MinA = 0 # ............................................... Minimum Area
    Pixels = len(SnapshotSlice) # ............................. Slice width
    localmin = [] # .............. Local Minimum set of illumination values
    LocMinI = [] # ......................... Local Minimum set of Locations 
    AeraSet = [] # ............................... local minimum areas list
   
    # Loop through the slice illumination values
    for pixel in range(Pixels):
        if SnapshotSlice[pixel] < avg: 
        # find the local minimum and store illumination values and location
            localmin.append(SnapshotSlice[pixel]); LocMinI.append(pixel)
            
            # find open local minimum at the end of the slice
            if pixel == Pixels-1 and len(localmin) >1: 
                A = abs(np.trapz(avg-localmin))
                SetMinPoint = min(localmin)
                AeraSet.append(A)
                if Plot: ax.fill_between(LocMinI, localmin,avg , alpha=0.5)
                if A > MinA and SetMinPoint/MinimumPoint > 0.3: 
                    MinA = A;   ShockRegion = [LocMinI,localmin]
                localmin = []; LocMinI = []
        
        # bounded local minimum
        elif SnapshotSlice[pixel] >= avg and len(localmin) > 1: 
            A = abs(np.trapz(avg-localmin))
            SetMinPoint = min(localmin)
            AeraSet.append(A)
            if Plot:ax.fill_between(LocMinI, localmin,avg , alpha=0.5)                        
            if A > MinA and SetMinPoint/MinimumPoint > 0.3:
                MinA = A;   ShockRegion = [LocMinI,localmin]
            localmin = []; LocMinI = []
            
        else: localmin = [];  LocMinI = []
    
    # check if there is more than one valley in the local minimum
    LocMinAvg = np.mean(ShockRegion[1])
    if Plot: ax.plot([ShockRegion[0][0]-5,ShockRegion[0][-1]+5],[LocMinAvg,LocMinAvg],'-.r')
    
    localmin2 = [] # .............................. sub-local minimum value
    LocMinI2 = [] # ............................ sub-local minimum location
    SubLocalMinSets = [] # ........ Sub-local minimum set [Location, Value]
    n = 0 # ................ number of valleys in the sub-local minimum set
    # if Plot: print(min(ShockRegion[1])/MinimumPoint)
    for k in range(len(ShockRegion[1])):
        # check all pixels lays under the valley average line
        if ShockRegion[1][k] < LocMinAvg:
            localmin2.append(ShockRegion[1][k]); LocMinI2.append(ShockRegion[0][k])
        elif ShockRegion[1][k] >= LocMinAvg and len(localmin2) > 1:
            SubLocalMinSets.append([LocMinI2,localmin2])
            n += 1; localmin2 = []; LocMinI2 = []
        else:
            localmin2 = []; LocMinI2 = []
    # uncertainity calculation
    certainLoc = True
    reason = ''
    
    # if there is more than one valley in the local minimum, 
    # the closest to the preivous location will be choosen
    if n > 1 and LastShockLoc > -1:
        # The minimum distance between the sub-valley and last shock location
        # initiated with the full lenght 
        MinDis = Pixels;  
        AreaSet2 = [] # ......................... Set of sub-local minimums
        MaxArea2 = 0 # ................. minimum area in sub-local minimums
        for SubLocalMinSet in SubLocalMinSets:
            # calculating the area of the sub-valley
            A2 = abs(np.trapz(LocMinAvg-SubLocalMinSet[1]))
            AreaSet2.append(A2) # ........................ storing the area
            if A2 > MaxArea2: MaxArea2 = A2 # ...... Check the minimum area
            
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
        ax.plot([ShockRegion[0][0]-5,ShockRegion[0][-1]+5],[LocMinRMS,LocMinRMS],'-.k') 
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
    
    for Area in AeraSet:
        Ra = Area/MinA
        if Ra > 0.6 and Ra < 1 and certainLoc:
            certainLoc = False
            reason = 'Almost equal Valleys'
    
    if n > 1 and certainLoc:
        for Area in AreaSet2:
            if MaxArea2 > 0: Ra = Area/MaxArea2
            if Ra > 0.5 and Ra < 1 and certainLoc: certainLoc = False; reason = 'Almost equal sub-Valleys'   
            if MaxArea2 !=  abs(np.trapz(LocMinAvg-ShockRegion[1])) and certainLoc: 
                certainLoc = False; reason = 'different sub-Valleys than smallest'
    
    if (not certainLoc) and Plot: 
        ax.text(0.99, 0.99, 'uncertain: '+ reason,
                ha = 'right', va ='top', transform = ax.transAxes,
                color = 'red', fontsize=14)
        # ax.set_ylim([-1,1])
    return minLoc, certainLoc, reason