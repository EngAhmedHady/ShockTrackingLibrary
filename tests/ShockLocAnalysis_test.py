# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:23:30 2024

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
sys.path.append(r'..\ShockOscillationAnalysis')
from ShockOscillationAnalysis import SOA

if __name__ == '__main__':
    # define the slice list file
    imgPath = r'results\2.0kHz_10mm_0.12965964343598055mm-px_tk_60px_-SliceList.png'

    f = 2000    # images sampling rate

    # from the file name or can be passed directly from:
    # SliceListGenerator.GenerateSlicesArray function
    scale = 0.12965964343598055  # mm/px

    # import the slice list
    slicelist = cv2.imread(imgPath)
    n = slicelist.shape[0]  # time

    # iniate the ShockOscillationAnalysis module
    SA = SOA(f)

    # spacify the shock region (Draw 2 vertical lines)
    newref = SA.LineDraw(slicelist, 'V', 0, Intialize=True)
    newref = SA.LineDraw(SA.clone, 'V', 1)
    # to make sure the spacified lines are correctly sorted
    newref.sort()
    # to crop the slicelist to the shock region
    shock_region = slicelist[:, newref[0]:newref[1]]
    # the width of the slice list in pixels
    xPixls = (newref[1]-newref[0])
    # the width of the slice list in mm
    shock_region_mm = xPixls*scale
    print(f'Shock Regions: {newref},\t Represents: {xPixls}px, \t Shock Regions in mm:{shock_region_mm}')

    # %% slice list cleaning
    # [subtracting the average, subtracting ambiant light frequency,
    # improve brightness/contrast/sharpness]
    shock_region = SA.CleanSnapshots(shock_region, 'Average')

    # %% Find shock location
    shock_loc_px, uncer = SA.ShockTrakingAutomation(shock_region,
                                                    method='integral',        # There is also 'maxGrad' and 'darkest_spot'
                                                    reviewInterval=[11, 14],  # to review the tracking process within this range
                                                    Signalfilter='med-Wiener')

    print(f'uncertainty ratio: {(len(uncer)/len(shock_loc_px))*100:0.2f}%')

    # unpack and scale the output values
    # to scale the shock location output to mm
    shock_loc_mm = scale * np.array(shock_loc_px)

    # unpack uncertainity columns
    snapshot_indx, uncertain, reason = zip(*uncer)
    # to scale the uncertain locatshock location output to mm
    uncertain_mm = scale * np.array(uncertain)

    # plotting the output
    fig1, ax1 = plt.subplots(figsize=(8, 50))
    ax1.imshow(shock_region,
               extent=[0, shock_region_mm, shock_region.shape[0], 0],
               aspect='0.1', cmap='gray')
    ax1.plot(shock_loc_mm, range(n), 'x', lw=1, color='g', ms=7)
    ax1.plot(uncertain_mm, snapshot_indx, 'x', lw=1, color='r', ms=5)

    # %% Apply welch method for PSD
    # find the average shock location
    avg_shock_loc = np.average(shock_loc_mm)
    # to shift the signal to the average
    shock_loc_mm = shock_loc_mm - avg_shock_loc

    Freq, psd = signal.welch(x=shock_loc_mm, fs=f, window='barthann',
                             nperseg=512, noverlap=0, nfft=None,
                             detrend='constant', return_onesided=True,
                             scaling='density')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.loglog(Freq, psd, lw='2')
    ax.set_ylabel(r"PSD [mm$^2$/Hz]")
    ax.set_xlabel("Frequency [Hz]")
