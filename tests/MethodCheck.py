# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:30:31 2024

@author: admin
"""

import os
import sys
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from ShockOscillationAnalysis import SOA
# from ShockOscillationAnalysis import importSchlierenImages as ImpS

imgPath = '..\\Shock wave pixels\\Paper uncertainity\\10kHz\\*.png'
f = 10000
D = 80     # Distance in mm
HLP = 7   # Horizontal line position [slice location to a reference line]
Scale = 0.13029315960912052
SliceThickness = 80
n = -1
CaseName = 'MidRe'


Folders = imgPath.split("\\"); FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (f'{Folders[i]}\\')
# NewFileDirectory = os.path.join(FileDirectory, "shock_sliceThickness-NoRotation")
NewFileDirectory = os.path.join(FileDirectory, "shock_sliceThickness-Final")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)
File = f"{f/1000}kHz_{HLP}mm_{Scale}mm-px_ts_{SliceThickness}_slice--.png"


SA = SOA(f,D);
if os.path.exists(f"{NewFileDirectory}\\{File}"): ImgList = cv2.imread(f"{NewFileDirectory}\\{File}")
else: print('File is not exist!!'); sys.exit() 

NewRef = [283, 440]
# NewRef = SA.LineDraw(ImgList, 'V', 0, 1)
# NewRef = SA.LineDraw(SA.clone, 'V', 1)
# NewRef.sort()

ShockwaveRegion = ImgList[:,NewRef[0]:NewRef[1]]
xPixls = (NewRef[1]-NewRef[0])
ShockResionScale = xPixls*Scale
print('Image scale: ', Scale, 'mm/px') # indicates the displacement accuracy
print('Shock Regions:',NewRef,'\t Represents:' ,xPixls, 'px \t Shock Regions in mm:', ShockResionScale)

print('Cleaning illumination instability ...')


methods = ['integral','maxGrad','darkest_spot']
fig,ax = plt.subplots(figsize=(10,10))
fig2,ax2 = plt.subplots(figsize=(10,10))
# fig1, ax1 = plt.subplots(figsize=(60,800))
fig1, ax1 = plt.subplots(figsize=(5,10))

for method in methods:
    ShockLocation, Uncer = SA.ShockTrakingAutomation(ShockwaveRegion, 
                                                     method = method,
                                                     reviewInterval = [0,0], 
                                                     Signalfilter = None)
    
    print('uncertainty ratio:', round((len(Uncer)/len(ShockLocation))*100,2),'%')
    
    A = Scale * np.array(ShockLocation)   
    avg = np.average(A)
    ShockLocation = A - avg
    
    n = ShockwaveRegion.shape[0]
    k = 0; domain = []
    while k < n:
        shockLoc =[]; j = 0
        while j <= 500 and k < n:
            shockLoc.append(ShockLocation[k])
            j += 1; k += 1
        domain.append(max(shockLoc)-min(shockLoc))
    avgDomain = np.mean(domain)
    
    print("Shock oscillation domain",max(ShockLocation)-min(ShockLocation))
    print("Average Shock oscillation domain",avgDomain)

    # Apply welch method for PSD
    Freq, psd = signal.welch(x = ShockLocation, fs = f, window='barthann',
                          nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
                          return_onesided=True, scaling='density')
    
    # choose which is more convenient (log on both axes or only on x)
    ax.loglog(Freq, psd, lw = '2', label = method)
    # ax.semilogx(Freq, psd, lw = '2')   
    ax.set_ylabel(r"PSD [mm$^2$/Hz]"); 
    ax.set_xlabel("Frequency [Hz]");
    # ax.set_title('Shock displacement PSD')
    ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

    # Find velocity oscillation using finite difference method
    # (centeral difference in general and forward/backward at boundaries)
    T = n/f
    print("Total measuring time: ", T, "sec")

    V = SA.VelocitySignal(ShockLocation, T)
    Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
                          nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
                          return_onesided=True, scaling='density')
    
    ax2.semilogx(Freq2, psd2 , lw = '2', label = method)
    ax2.axvline(x =Freq2[psd2.argmax(axis=0)],ls='--',color='k',alpha=0.4)
    ax2.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
    ax2.set_xlabel("Frequency [Hz]");
    ax2.set_title('Shock velocity PSD')
    ax2.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
    
    
    ax1.set_yticks(np.arange(0, n+1, 100))
    ax1.set_ylim([32900,32700])
    xPixls = (NewRef[1]-NewRef[0])
    ShockResionScale = xPixls*Scale
    
    ax1.invert_yaxis()
    ax1.set_ylabel(r"snapshots (n)"); 
    ax1.set_xlabel(r"Shock oscillation domain ($x$) [mm]");
    ax1.plot(A, range(n),'x', lw = 1, ms = 3, label = method)
    
ShockwaveRegion = SA.CleanSnapshots(ShockwaveRegion,
                                    'Average', 'FFT', #'Brightness and Contrast',
                                    filterCenter = [(0, 233)], D = 20, n = 5,
                                    ShowIm = False)
ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, n, 0], aspect='0.1', cmap='gray');

ax.legend();
ax2.legend();
