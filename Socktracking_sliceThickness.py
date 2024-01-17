# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:52:13 2023

@author: admin
"""

import os
import sys
import cv2
import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt
from ShockOscillationAnalysis import SOA
from ShockOscillationAnalysis import importSchlierenImages as ImpS

f = 1500
D = 59
# D = 97.741
HLP = 7
Scale = 0.13245033112582782
# WorkingRange = [56, 673, 470] #Ref-3000
# WorkingRange = [241, 687, 260] #Fully Open with suction-1500
n = 0

SliceThicknesses = [0,2,4,8,10,16,20,25,32,35,40,50,60,64,70,75,80,90,100,110]
# SliceThicknesses = [0,1,2,4,8,10,16,20,25,32,35,40,50,60]
# SliceThicknesses = [4,20,80,128]
# SliceThicknesses = [60]

# SliceThicknesses = [10]

# imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\10kHz_smooth P1 (Test 8)\\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\10kHz_smooth P1 (Test 8)\\*.png'
# imgPath = 'D:\\PhD\\TEAMAero\\2023_03_29 - Fully Open with suction\\Oil-sch test 8 - Final full covered 25fps - 5%\\0_20230329_164729\\*.png'
imgPath = 'D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Fully Open\\2023-03-29_Sync_Oil_sch-fully open\\Oil-sch test 8 - Final full covered\\Schlieren\\*.png'
CaseName = 'Passage-shock_Rotated'

Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
NewFileDirectory = os.path.join(FileDirectory, f"shock_signal")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)

uncertainityRatios = []
AvgOscDomain = []
AvgAngle = []
shockXLoc = [];
SA = SOA(f,D)
ImpImg = ImpS(f,D)
fig,ax = plt.subplots(figsize=(10,10))
fig2,ax2 = plt.subplots(figsize=(10,10))

for thickness in SliceThicknesses:
    print(f'**********[ {CaseName} - Slice thickness {thickness} ]****************')
    ShockwaveRegion ,n ,WR, Scale = ImpImg.GenerateSlicesArray(imgPath,
                                                                  HLP = HLP,
                                                                  FullImWidth = False,
                                                                  nt = -1,
                                                                  WorkingRange = [240, 693, 260], #MidRe-Test 8-leading edge
                                                                  # WorkingRange =[109, 726, 565], #P1-2kHz
                                                                  # WorkingRange =[109, 726, 276], #P1-3kHz
                                                                  # WorkingRange =[111, 725, 75],  #P1-6kHz
                                                                  # WorkingRange =[111, 725, 67],  #P1-10kHz
                                                                  # WorkingRange =[109, 727, 31],  #P1-15kHz
                                                                  ScalePixels= True,
                                                                  ShockAngleSamples = 4500,
                                                                  AngleSamplesReview = 5,
                                                                  SliceThickness = thickness, 
                                                                  # inclinationEst = [90,(479, 99),(478, 46)], #P1-6kHz
                                                                  # inclinationEst = [90,(470, 129),(459, 5)], #P1-10kHz
                                                                  # inclinationEst = [120,(462, 86),(457, 18)], #P1-15kHz
                                                                  # inclinationEst = [70,(87, 296),(124, 210)], #MidRe-Test 8-leading edge
                                                                  inclinationEst = [150,(423, 292),(416, 217)], #MidRe-Test 8
                                                                  OutputDirectory = NewFileDirectory,
                                                                  comment = f'{CaseName}')    
    
    # File = f"{f/1000}kHz_{HLP}mm_{Scale}mm-px_ts_{thickness}_slice_{CaseName}.png"
    File = f"{f/1000}kHz_{HLP}mm_{Scale}mm-px_ts_{thickness}_slice_{CaseName}.png"
    if os.path.exists(NewFileDirectory+'\\'+File): ImgList = cv2.imread(NewFileDirectory+'\\'+File)
    else: print('File is not exist!!'); sys.exit();
    
    # spacify the shock region
    # (Draw 2 vertical lines)
    # NewRef = SA.LineDraw(ImgList, 'V', 0, 1)
    # NewRef = SA.LineDraw(SA.clone, 'V', 1)
    # NewRef.sort()
   
    # or (Spacify x location of 2 vertical lines)
    # NewRef = [292, 446] # Ref-3000
    NewRef = [103, 250] # Fully Open with suction-1500
    # NewRef = [383, 552] # 2kHz P1
    # NewRef = [288, 431] # 3kHz P1
    # NewRef = [286, 433] # 6kHz P1
    # NewRef = [283, 440] # 10kHz P1
    # NewRef = [291, 426] # 15kHz P1
    # NewRef = [47, 140] # MidRe-Test 8-leading edge
    
    ShockwaveRegion = ImgList[:,NewRef[0]:NewRef[1]]
    xPixls = (NewRef[1]-NewRef[0])
    ShockResionScale = xPixls*Scale
    print('Image scale: ', Scale, 'mm/px') # indicates the displacement accuracy
    print('Shock Regions:', NewRef,'\t Represents:', xPixls, 'px \t Shock Regions in mm:', ShockResionScale)
    
    # Image cleaning [subtracting the average, subtracting ambiant light frequency]    
    # CleanIlluminationEffects
    # function inputs: 1- the image needed to be cleaned
    #                   2- location/center of the light frequency peak in FFT domain (Spectlocation)
    #                   3- butterworth function parameter (D = circle diameter, n = function power)
    #                   4- see the FFT domain before and after filtering (True/False)
    print('Cleaning illumination instability ...')
    ShockwaveRegion = SA.CleanSnapshots(ShockwaveRegion,
                                        'Brightness and Contrast',
                                        'Average', 'FFT', 
                                        # filterCenter = [(0, 233)], D = 20, n = 5,
                                        filterCenter = [(0, 465), (-10, 465), (10, 465), (0, 490)], D = 10, n = 5,
                                        Brightness = 1.5, Contrast = 2, Sharpness = 1.5,
                                        ShowIm = False)
    
    # fig1, ax1 = plt.subplots(figsize=(20,200))
    # ShockResionScale = xPixls*Scale
    # ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, n, 0], aspect='0.1', cmap='gray');
    # Find shock location
    # function inputs: 1- the oscillation domain image where the shock will be tarcked
    #                  2- reviewInterval: to plot the image slices in defined interval (to evaluate the tracking if needed)
    #                     [defult is [0,0] which mean notheing to review]
    #                  3- Signalfilter: ['median','Wiener','med-Wiener']
    ShockLocation, Uncer = SA.ShockTrakingAutomation(ShockwaveRegion, 
                                                     reviewInterval = [0,0],
                                                     # Signalfilter = 'None')
                                                     Signalfilter = 'Wiener')
    
    uncertainityRatio = (len(Uncer)/len(ShockLocation))*100
    uncertainityRatios.append(uncertainityRatio)
    print('uncertainty ratio:', round(uncertainityRatio,2),'%')
    print('Uncertainty set:',len(uncertainityRatios))
    
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
    
    AvgOscDomain.append(avgDomain)
    
    if len(WR)>3:
        AvgAngle.append(90-WR[3]);
        shockXLoc.append(WR[4][0]);
    else:
        AvgAngle.append(90);
        shockXLoc.append(467);
    
    ShockLocationfile = np.transpose([range(n),ShockLocation])
    # np.savetxt(NewFileDirectory + '\\ShockLocation-' + CaseName +'-'+str(NewRef)+'.txt', 
    #             ShockLocationfile,  delimiter = ",")
    
    print("Shock oscillation domain",max(ShockLocation)-min(ShockLocation))
    print("Average Shock oscillation domain",avgDomain)
    
    # Apply welch method for PSD
    Freq, psd = signal.welch(x = ShockLocation, fs = f, window='barthann',
                          nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
                          return_onesided=True, scaling='density')
      
    # choose which is more convenient (log on both axes or only on x)
    ax.loglog(Freq, psd, lw = '2', label = 'ts_'+str(thickness)+'_slice')
    ax.semilogx(Freq, psd, lw = '2')   
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
    V = SA.VelocitySignal(ShockLocation, T)

    
    Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
                          nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
                          return_onesided=True, scaling='density')
    
    # Find maximum peak in velocity PSD
    domFreq = Freq2[psd2.argmax(axis=0)]
    
    print('max peak at:', domFreq, 'Hz')
    
    
    ax2.semilogx(Freq2, psd2 , lw = '2', label = 'ts_'+str(thickness)+'_slice')
    # ax2.axvline(x =Freq2[psd2.argmax(axis=0)],ls='--',color='k',alpha=0.4)
    ax2.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
    ax2.set_xlabel("Frequency [Hz]");
    # ax2.set_title('PSD for '+ CaseName +' above LE')
    # ax2.set_title('Shock velocity PSD')
    ax2.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
    
    fig1, ax1 = plt.subplots(figsize=(60,700))
    # fig1, ax1 = plt.subplots(figsize=(5,10))
    ax1.set_yticks(np.arange(0, n+1, 100))
    # ax1.set_ylim([32900,32700])
    ShockResionScale = xPixls*Scale
    ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, n, 0], aspect='0.1', cmap='gray');
    ax1.plot(A, range(n),'x', lw = 1, color = 'g', ms = 5)
    ax1.plot(uncertain, Loc,'x', lw = 1, color = 'r', ms = 3)
    fig1.savefig(f'{NewFileDirectory}\\ShockOsc-WithThickness_{thickness}_{CaseName}.png')

ax.legend(fontsize="14")
ax2.legend(fontsize="14")

def exp_fitting(x, A, seg):
    return A*np.e**(-x/seg)


popt, pcov = optimize.curve_fit(exp_fitting, SliceThicknesses, uncertainityRatios)

fit_A, fit_seg = popt
 
# Sample exp_decay with optimized parameters
opt_A, opt_seg = popt

# print(opt_k,opt_B, opt_k_plus)
print('Opt. A =',opt_A,'Opt. seg =', opt_seg)

uncertainityRatios_fit = exp_fitting(np.array(SliceThicknesses), opt_A, opt_seg)


UncertaintyRatio = np.transpose([SliceThicknesses,AvgAngle,shockXLoc,uncertainityRatios])
np.savetxt(NewFileDirectory + '\\UncertaintyRatio-' + CaseName +'-'+str(NewRef)+'.txt', 
            UncertaintyRatio,  delimiter = ",")

fig3,ax3 = plt.subplots(figsize=(13,10))
ax3.plot(SliceThicknesses,uncertainityRatios,'s')
ax3.plot(SliceThicknesses, uncertainityRatios_fit, linestyle = '--', color = 'k')
ax3.set_ylabel("Uncertainty ratio [\%]"); 
ax3.set_xlabel("Slice thicknesses [mm]");
ax3.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax3.minorticks_on()
ax3.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

fig3.savefig(NewFileDirectory +'\\Uncertainty ratio'+ CaseName +'.png')

AvgOscillationDomain = np.transpose([SliceThicknesses,AvgOscDomain])
np.savetxt(NewFileDirectory + '\\AverageOscillationDomain-' + CaseName +'-'+str(NewRef)+'_ts_'+str(thickness)+'.txt', 
            AvgOscillationDomain,  delimiter = ",")

fig4,ax4 = plt.subplots(figsize=(13,10))
ax4.plot(SliceThicknesses,AvgOscDomain,'-s')
ax4.set_ylabel("Average oscillation domain [mm]"); 
ax4.set_xlabel("Slice thicknesses [mm]");
ax4.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax4.minorticks_on()
ax4.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
fig4.savefig(NewFileDirectory +'\\Average oscillation domain-' + CaseName +'.png')


