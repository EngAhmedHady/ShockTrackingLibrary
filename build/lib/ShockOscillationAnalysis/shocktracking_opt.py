# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:59:00 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

def _plot_slice_details(ax, SnapshotSlice, avg, MinimumPoint, LocMinAvg, LocMinRMS, ShockRegion, minLoc, LastShockLoc, count, certainLoc, reason):
    """Helper function for plotting details of the slice and shock tracking."""
    ax.plot(SnapshotSlice, label='Light intensity at certain snapshot')
    ax.axhline(avg, linestyle=':', color='tab:green', label='Light intensity average line')
    ax.axhline(MinimumPoint, linestyle='--', color='k')
    ax.set_ylim([-20, 255])
    ax.set_yticks(np.arange(0, 260, 51))
    ax.axhline(0, linestyle=':', color='k', alpha=0.2)
    ax.axhline(255, linestyle=':', color='k', alpha=0.2)

    if ShockRegion: # Only plot if ShockRegion was found
        ax.plot([ShockRegion[0][0], ShockRegion[0][-1]], [LocMinAvg, LocMinAvg], 
                '-.r', label='Average line of largest local minimum')
        ax.plot([ShockRegion[0][0], ShockRegion[0][-1]], [LocMinRMS, LocMinRMS], 
                '-.k', label='RMS line of largest local minimum')
        ax.fill_between(ShockRegion[0], ShockRegion[1], avg, color='#1F79B7', edgecolor='k',
                        hatch='///', label='Largest local minimum')

    ax.axvline(minLoc, linestyle='--', color='tab:purple', label='Middle line of local minimum')
    if count > -1:
        ax.set_title(count)

    if LastShockLoc > -1:
        ax.axvline(LastShockLoc, linestyle='--', color='tab:red', 
                   label='Location shock on previous snapshot')
        handles, labels = plt.gca().get_legend_handles_labels()
        # Ensure 'order' corresponds to the actual labels present.
        # This part might need adjustment if labels change dynamically based on conditions.
        # For a fixed set of labels, ensure indices match.
        try:
            order = [labels.index(lbl) for lbl in ['Light intensity at certain snapshot', 
                                                   'Light intensity average line', 
                                                   'RMS line of largest local minimum', 
                                                   'Largest local minimum', 
                                                   'Average line of largest local minimum', 
                                                   'Middle line of local minimum', 
                                                   'Location shock on previous snapshot'] if lbl in labels]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                      bbox_to_anchor=(1.9, 0.5), loc='right', fontsize=20)
        except ValueError:
            ax.legend(bbox_to_anchor=(1.9, 0.5), loc='right', fontsize=20)

        if abs(LastShockLoc - minLoc) > 15:
            arrow_props = dict(arrowstyle='<|-|>', fc='k', ec='k')
            ax.annotate('', xy=(LastShockLoc, -13.5), xytext=(minLoc, -13.5), 
                        arrowprops=arrow_props)
            ax.text((LastShockLoc + minLoc) / 2, -10, f'{abs(LastShockLoc - minLoc):0.1f}px',
                    ha='center', fontsize=14)
        else:
            arrow_props = dict(arrowstyle='-|>', fc='k', ec='k')
            if LastShockLoc > minLoc:
                ax.annotate('', xy=(LastShockLoc, -13.5), xytext=(LastShockLoc + 10, -13.5),
                            arrowprops=arrow_props)
                ax.annotate('', xy=(minLoc, -13.5), xytext=(minLoc - 10, -13.5),
                            arrowprops=arrow_props)
            else:
                ax.annotate('', xy=(LastShockLoc, -13.5), xytext=(LastShockLoc - 10, -13.5),
                            arrowprops=arrow_props)
                ax.annotate('', xy=(minLoc, -13.5), xytext=(minLoc + 10, -13.5),
                            arrowprops=arrow_props)
            
            minX, maxX = ax.get_xlim()
            text_x_offset = 15 if LastShockLoc + 20 < maxX else -15
            ax.text((LastShockLoc + minLoc) / 2 + text_x_offset, -10,
                    f'{abs(LastShockLoc - minLoc):0.1f}px', ha='center', fontsize=14)

    if (not certainLoc):
        ax.text(0.99, 0.99, 'uncertain: ' + reason,
                ha='right', va='top', transform=ax.transAxes,
                color='red', fontsize=14)

def ShockTraking(SnapshotSlice, LastShockLoc=-1, Plot=False, count=-1, Alpha=0.3):
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
        >>> # Assuming SnapshotSlice is a numpy array, e.g., 
        >>> # SnapshotSlice = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        >>> result = ShockTraking(SnapshotSlice, LastShockLoc=10, Plot=True, count=1)
        >>> print(result)

    .. note::
        - The function processes the illumination values of a given image slice to track shock waves.
        - It returns the determined shock location, certainty status, and a reason for uncertainty.
    """
    certainLoc = True
    minLoc = np.nan
    reason = ''
    Pixels = len(SnapshotSlice)

    try:
        avg = np.mean(SnapshotSlice)
        MinimumPoint = np.min(SnapshotSlice) # Use np.min for consistency with numpy
    except Exception as e:
        print(count, SnapshotSlice, e)
        certainLoc = False
        if Plot: # Plotting only if it was requested
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(SnapshotSlice)
            ax.axhline(avg, linestyle=':')
            ax.text(0.99, 0.99, f'Error: {e}', ha='right', va='top', transform=ax.transAxes,
                    color='red', fontsize=14)
            if count > -1: ax.set_title(count)
        return minLoc, certainLoc, 'Error in initial calculations'

    # Find local minimums using vectorized operations
    # A valley is characterized by values below average, bounded by values above average or slice ends.
    below_avg = SnapshotSlice < avg
    # Find start and end indices of consecutive 'True' (below_avg) segments
    diffs = np.diff(np.concatenate(([False], below_avg, [False])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    MinA = 0
    
    ShockRegion = [[], []] 
    AeraSet = []

    for start, end in zip(starts, ends):
        localmin_indices = np.arange(start, end)
        if len(localmin_indices) > 1:
            locs = localmin_indices.tolist()
            values = SnapshotSlice[localmin_indices].tolist()
            
            current_area = abs(np.trapezoid(avg - np.array(values)))
            current_min_point = np.min(values)
            
            AeraSet.append(current_area)

            if (avg - current_min_point) / (avg - MinimumPoint) > Alpha:
                if current_area > MinA:
                    MinA = current_area
                    ShockRegion = [locs, values]
                    
    if not ShockRegion[0]: # If no suitable ShockRegion was found
        certainLoc = False
        reason = 'No significant local minimum found'
        if Plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(SnapshotSlice, label='Light intensity at certain snapshot')
            ax.axhline(avg, linestyle=':', color='tab:green', label='Light intensity average line')
            ax.text(0.99, 0.99, 'uncertain: ' + reason, ha='right', va='top',
                    transform=ax.transAxes, color='red', fontsize=14)
            if count > -1: ax.set_title(count)
        return minLoc, certainLoc, reason

    # Check for sub-local minimums within the main ShockRegion
    LocMinAvg = np.mean(ShockRegion[1])
    
    sub_below_avg = np.array(ShockRegion[1]) < LocMinAvg
    sub_diffs = np.diff(np.concatenate(([False], sub_below_avg, [False])))
    sub_starts = np.where(sub_diffs == 1)[0]
    sub_ends = np.where(sub_diffs == -1)[0]

    SubLocalMinSets = []
    AreaSet2 = []
    MaxArea2 = 0
    
    for s_start, s_end in zip(sub_starts, sub_ends):
        sub_localmin_relative_indices = np.arange(s_start, s_end)
        if len(sub_localmin_relative_indices) > 1:
            # Map relative indices back to original slice indices using ShockRegion[0]
            sub_locs = [ShockRegion[0][i] for i in sub_localmin_relative_indices]
            sub_values = [ShockRegion[1][i] for i in sub_localmin_relative_indices]
            
            current_sub_area = abs(np.trapezoid(LocMinAvg - np.array(sub_values)))
            AreaSet2.append(current_sub_area)
            SubLocalMinSets.append([sub_locs, sub_values])
            if current_sub_area > MaxArea2:
                MaxArea2 = current_sub_area

    n = len(SubLocalMinSets) # Number of sub-valleys

    # If there is more than one sub-valley and a LastShockLoc is available,
    # choose the sub-valley closest to LastShockLoc.
    if n > 1 and LastShockLoc > -1:
        MinDis = Pixels
        best_sub_region = None
        for sub_set in SubLocalMinSets:
            min_value_in_sub = np.min(sub_set[1])
            # Find the index of minValue in the sub_set[1] (values)
            # Then use this index to get the corresponding location from sub_set[0]
            min_loc_in_sub_set = sub_set[0][np.argmin(sub_set[1])]
            
            distance = abs(LastShockLoc - min_loc_in_sub_set)
            if distance < MinDis:
                MinDis = distance
                best_sub_region = sub_set
        if best_sub_region: # Update ShockRegion to the chosen sub-region
            ShockRegion = best_sub_region
            
    elif n > 1 and LastShockLoc == -1:
        # If multiple sub-valleys but no history, it's uncertain
        certainLoc = False
        reason = 'First pixel slice, No shock location history with multiple sub-valleys'
        # No change to ShockRegion in this case, it remains the initially largest local minimum

    # Find the middle of the shock wave as middle point of RMS
    # Ensure avg-ShockRegion[1] is non-negative before sqrt
    diff_from_avg = np.array(avg - ShockRegion[1])
    # Handle cases where diff_from_avg might contain negative values (i.e., ShockRegion[1] > avg)
    # This shouldn't happen if ShockRegion is correctly identified as a "local minimum" (below avg)
    # but defensive coding helps. For a true 'valley' below avg, (avg - val) should be positive.
    LocMinRMS = avg - np.sqrt(np.mean(diff_from_avg**2))

    if LocMinRMS < np.min(ShockRegion[1]): # Ensure RMS line doesn't go below the actual minimum
        LocMinRMS = np.min(ShockRegion[1])

    # Calculate final shock location
    shockLoc_values = [ShockRegion[0][i] for i, val in enumerate(ShockRegion[1]) if val <= LocMinRMS]
    minLoc = np.mean(shockLoc_values) if shockLoc_values else np.nan # Ensure list is not empty

    # Check for uncertainty conditions
    # This part should ideally be done *after* all potential ShockRegion adjustments.
    if MinA > 0: # Ensure MinA is not zero to avoid division by zero
        for Area in AeraSet:
            if MinA > 0: # Defensive check
                Ra = Area / MinA
                if 0.6 < Ra < 1 and certainLoc:
                    certainLoc = False
                    reason = 'Almost equal Valleys'
                    break # No need to check further areas if already uncertain
    
    if n > 1 and certainLoc:
        try:
            if MaxArea2 > 0: # Ensure MaxArea2 is not zero to avoid division by zero
                for Area in AreaSet2:
                    Ra = Area / MaxArea2
                    if 0.5 < Ra < 1 and certainLoc:
                        certainLoc = False
                        reason = 'Almost equal sub-Valleys'
                        break
                # Re-check this condition: 
                #    if MaxArea2 != abs(np.trapezoid(LocMinAvg-ShockRegion[1]))
                # This seems to check if the max area among sub-valleys is 
                # NOT the area of the *currently selected* ShockRegion
                # which could be problematic if ShockRegion was updated to one of the sub-valleys.
                # Assuming this checks against the original 'largest' sub-valley identified for MaxArea2
                if MaxArea2 > 0 and MaxArea2 != abs(np.trapezoid(LocMinAvg - np.array(ShockRegion[1]))) and certainLoc:
                     certainLoc = False
                     reason = 'Different dominant sub-valley after selection'
            else: # MaxArea2 is 0, implying no sub-valleys were found or none met criteria
                certainLoc = False
                reason = 'No significant sub-valleys found when expected multiple'

        except Exception as e: # Catch errors specifically related to AreaSet2 or sub-valley calculations
            if Plot: # Only plot if it was requested
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(SnapshotSlice)
                ax.axhline(avg, linestyle=':')
                ax.text(0.99, 0.99, f'Error: {e}', ha='right', va='top', transform=ax.transAxes,
                        color='red', fontsize=14)
                if count > -1: ax.set_title(count)
            certainLoc = False
            print(f"\n The error is: {e}")
            return minLoc, certainLoc, reason

    if Plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_slice_details(ax, SnapshotSlice, avg, MinimumPoint, LocMinAvg, LocMinRMS,
                            ShockRegion, minLoc, LastShockLoc, count, certainLoc, reason)
        plt.tight_layout(rect=[0, 0, 0.7, 1]) # Adjust layout to make room for legend
        plt.show()

    return minLoc, certainLoc, reason

