# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:52:13 2023

@author: admin
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ShockOscillationAnalysis import SOA
from scipy import signal

f = 15000
D = 80      # distance in mm
HLP = 7     # Horizontal line position [slice location to a reference line]
Scale = 0.12944983818770225
n = 0;
# imgPath = 'C:\\Users\\Hady-PC\\Desktop\\PhD\\Schlieren\\2kHz_2_20220729\\*.png'
imgPath = '*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Half Open\\2022_11_24 - half opend without 3rd plate - NoSuction\\2.5kHz\\*.png'
# imgPath = 'D:\\TFAST\TEAMAero experiments\\2023_05_10\\Smooth-2kHz-5sec (test 5)\\*.png'
# CaseName = '17suc-ReflecEff-Leading'
CaseName = 'Ref'
# CaseName = '17suc-fullsuc-ReflecEff-passage'

Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
NewFileDirectory = os.path.join(FileDirectory, "shock_signal")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)

SA = SOA(f,D)
# ShockwaveRegion ,n ,H_line, Scale = SA.ImportSchlierenImages(imgPath,
#                                                               HLP = HLP,
#                                                               FullImWidth = True,
#                                                               ScalePixels= True,
#                                                               OutputDirectory = NewFileDirectory,
#                                                               SliceThickness = 35,
#                                                               comment = 'ts_35_slice')

# ImgList = cv2.imread(NewFileDirectory+'\\2.0kHz_50mm_0.1360544217687075mm-px_ts_10_slice.png')
File = 'P4_' +str(f/1000)+'kHz_'+str(HLP)+'mm_'+str(Scale)+'mm-px_ts_35_slice.png'

print(NewFileDirectory+'\\'+File)
if os.path.exists(NewFileDirectory+'\\'+File): ImgList = cv2.imread(NewFileDirectory+'\\'+File)
else: 
    print('File is not exist!!')
    sys.exit()

# spacify the shock region
# (Draw 2 vertical lines)
# NewRef = SA.LineDraw(ImgList, 'V', 0, 1)
# NewRef = SA.LineDraw(SA.clone, 'V', 1)
# NewRef.sort()
    
    
# or (Spacify x location of 2 vertical lines)
# NewRef = [262, 520]
NewRef = [262, 480]

ShockwaveRegion = ImgList[:,NewRef[0]:NewRef[1]]
xPixls = (NewRef[1]-NewRef[0])
ShockResionScale = xPixls*Scale
print('Image scale: ', Scale, 'mm/px') # indicates the displacement accuracy
print('Shock Regions:',NewRef,'\t Represents:' ,xPixls, 'px \t Shock Regions in mm:', ShockResionScale)

# Image cleaning [subtracting the average, subtracting ambiant light frequency]

ShockwaveRegion = SA.Average(ShockwaveRegion)

# CleanIlluminationEffects
# function inputs: 1- the image needed to be cleaned
#                   2- location/center of the light frequency peak in FFT domain (Spectlocation)
#                   3- butterworth function parameter (D = circle diameter, n = function power)
#                   4- see the FFT domain before and after filtering (True/False)
print('Cleaning illumination instability ...')
ShockwaveRegion = SA.CleanIlluminationEffects(ShockwaveRegion, 
                                              Spectlocation = [0, 230], 
                                              D = 20, n = 5, 
                                              ShowIm = False)

# ShockwaveRegion = SA.CleanIlluminationEffects(ShockwaveRegion, 
#                                               Spectlocation = [0, 180], 
#                                               D = 20, n = 5, 
#                                               ShowIm = False)


# Find shock location
# function inputs: 1- the oscillation domain image where the shock will be tarcked
#                  2- reviewInterval: to plot the image slices in defined interval (to evaluate the tracking if needed)
#                     [defult is [0,0] which mean notheing to review]
#                  3- Signalfilter: ['median','Wiener','med-Wiener']
ShockLocation, Uncer = SA.FindTheShockwaveImproved(ShockwaveRegion, 
                                                   reviewInterval = [0,3], 
                                                   Signalfilter = 'med-Wiener')
# print(Uncer)
print('uncertainty ratio:', round((len(Uncer)/len(ShockLocation))*100,2),'%')
# print(Uncer)

uncertain = []; Loc = []
for i in Uncer:
    uncertain.append(i[1]*Scale)
    Loc.append(i[0])
    
A = Scale * np.array(ShockLocation)   
avg = np.average(A)
ShockLocation = A - avg

if n < 1: n = ShockwaveRegion.shape[0]

k = 0; domain = []
while k < n:
    shockLoc =[]; j = 0
    while j <= 500 and k < n:
        shockLoc.append(ShockLocation[k])
        j += 1; k += 1
    domain.append(max(shockLoc)-min(shockLoc))
avgDomain = np.mean(domain)

ShockLocationfile = []
k = 0
for i in range(n):
    uncer = 1
    if len(Loc) < k and i == Loc[k]:  uncer == 0; k +=1
    ShockLocationfile.append([i, ShockLocation[i], uncer])
    


# ShockLocationfile = np.transpose([range(n),ShockLocation])
# ShockLocationfile = np.transpose(ShockLocationfile)
np.savetxt(NewFileDirectory + '\\ShockLocation-' + CaseName +'-'+str(NewRef)+'.txt', 
           ShockLocationfile,  delimiter = ",")

print("Shock oscillation domain",max(ShockLocation)-min(ShockLocation))
print("Average Shock oscillation domain",avgDomain)

# Apply welch method for PSD
Freq, psd = signal.welch(x = ShockLocation, fs = f, window='barthann',
                      nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
                      return_onesided=True, scaling='density')



fig,ax = plt.subplots(figsize=(10,10))

# choose which is more convenient (log on both axes or only on x)
ax.loglog(Freq, psd, lw = '2')
# ax.semilogx(Freq, psd, lw = '2')   
ax.set_ylabel(r"PSD [mm$^2$/Hz]"); 
ax.set_xlabel("Frequency [Hz]");
ax.set_title('Shock displacement PSD')
ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax.minorticks_on()
ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

# To define the peakes on PSD signal with horizontal line
# ax.axvline(x = 19.5,ls='--',color='k',alpha=0.4)
# ax.axvline(x = 287,ls='--',color='k',alpha=0.4)


# Find velocity oscillation using finite difference method
# (centeral difference in general and forward/backward at boundaries)
T = n/f
print("Total measuring time: ", T, "sec")

dx_dt = []; dt = T/n; t = np.linspace(0,T,n);
for xi in range(n):
    if xi > 0 and xi < n-1:
        dx_dt.append((ShockLocation[xi+1]-ShockLocation[xi-1])/(2*dt*1000))
    elif xi == 0: 
        dx_dt.append((ShockLocation[xi+1]-ShockLocation[xi])/(dt*1000))
    elif xi == n-1:
        dx_dt.append((ShockLocation[xi]-ShockLocation[xi-1])/(dt*1000))


V_avg = np.mean(dx_dt) 
V = dx_dt - V_avg

Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
                      nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
                      return_onesided=True, scaling='density')

# Find maximum peak in velocity PSD
domFreq = Freq2[psd2.argmax(axis=0)]

print('max peak at:', domFreq, 'Hz')

fig2,ax2 = plt.subplots(figsize=(10,10))
ax2.semilogx(Freq2, psd2 , lw = '2')
ax2.axvline(x =Freq2[psd2.argmax(axis=0)],ls='--',color='k',alpha=0.4)
ax2.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
ax2.set_xlabel("Frequency [Hz]");
# ax2.set_title('PSD for '+ CaseName +' above LE')
ax2.set_title('Shock velocity PSD')
ax2.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax2.minorticks_on()
ax2.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

fig1, ax1 = plt.subplots(figsize=(40,500))
ax1.set_yticks(np.arange(0, n+1, 100))
xPixls = (NewRef[1]-NewRef[0])
ShockResionScale = xPixls*Scale
ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, n, 0], aspect='0.1', cmap='gray');
ax1.plot(A, range(n),'x', lw = 1, color = 'g', ms = 3)
ax1.plot(uncertain, Loc,'x', lw = 1, color = 'r', ms = 2)
# ax1.set_ylim([2900,3000])
plt.show()

