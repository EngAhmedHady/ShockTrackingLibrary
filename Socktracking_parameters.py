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
from ShockOscillationAnalysis import importSchlierenImages as ImpS

f = 10000
D = 80

# D = 97.741
HLP = 7
Scale = 0.12965964343598055
# WorkingRange = [56, 673, 470] #Ref-3000
# WorkingRange = [241, 687, 260] #Fully Open with suction-1500
n = 0

# SliceThicknesses = [0,1,2,4,8,10,16,20,25,32,35,40,50,60,64,70,75,80,90,100,110,128]
SliceThickness = 10

# SliceThicknesses = [10]

# imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\6kHz_smooth P1 (Test 7)\\*.png'
imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\10kHz_smooth P1 (Test 8)\\*.png'
CaseName = 'P1-S6500'

Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
NewFileDirectory = os.path.join(FileDirectory, "shock_sliceThicknessParametricCheck")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)




SA = SOA(f,D)
ImpImg = ImpS(f,D)
# fig,ax = plt.subplots(figsize=(10,10))
# fig2,ax2 = plt.subplots(figsize=(10,10))
AvgAngle = []; RMAngle = []
uncertainityRatios = []
shockXLoc = []; shockYLoc = []
AvgOscDomain = []
for trial in range(50):
    
    print(f'*****************[ {CaseName} - Trial No. {trial} ]************************')
    ShockwaveRegion ,n ,WR, Scale = ImpImg.GenerateSlicesArray(imgPath,
                                                                  HLP = HLP,
                                                                  FullImWidth = False,
                                                                  # WorkingRange = [109, 726, 565], #P1_2kHz
                                                                  # WorkingRange = [109, 726, 276], #P1_3kHz
                                                                  # WorkingRange = [111, 725, 75 ], #P1-6kHz
                                                                  WorkingRange = [111, 725, 67 ], #P1-10kHz
                                                                  ScalePixels= True,
                                                                  ShockAngleSamples = 6500,
                                                                  AngleSamplesReview = 0,
                                                                  SliceThickness = SliceThickness,
                                                                  OutputDirectory = NewFileDirectory,
                                                                  # inclinationEst = [80,(473,591),(466,514)], #P1_2kHz
                                                                  # inclinationEst = [80,(491,302),(488,264)], #P1_3kHz
                                                                  # inclinationEst = [90,(479,99 ),(478, 46)], #P1-6kHz  
                                                                  inclinationEst = [90,(470, 129),(459, 5)], #P1-10kHz  
                                                                  comment = f'_{CaseName}')
    
    
    File = f"{f/1000}kHz_{HLP}mm_{Scale}mm-px_ts_{SliceThickness}_slice_{CaseName}.png"
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
    # NewRef = [292, 446] #Ref-3000
    # NewRef = [113, 249] #Fully Open with suction-1500
    # NewRef = [277, 449]
    # NewRef = [383, 552] # 2kHz P1_ full width
    # NewRef = [282, 435] # 2kHz P1
    # NewRef = [269, 445]
    # NewRef = [237, 448]
    # NewRef = [288, 431] # 3kHz P1
    # NewRef = [286, 433] # 6kHz P1
    NewRef = [283, 440] # 10kHz P1
    
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
                                        'Average', 'FFT', #'Brightness and Contrast',
                                        filterCenter = [0, 233], D = 20, n = 5,
                                       # Brightness = 2, Contrast = 1.7, Sharpness = 2,
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
                                                        Signalfilter = 'med-Wiener')
    uncertainityRatio = (len(Uncer)/len(ShockLocation))*100
    uncertainityRatios.append(uncertainityRatio)
    print('uncertainty ratio:', round(uncertainityRatio,2),'%')
    
    
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
    AvgAngle.append(90-WR[3]); RMAngle.append((90-WR[3])**2)
    shockXLoc.append(WR[4][0]); shockYLoc.append(WR[4][1])

    
    ShockLocationfile = np.transpose([range(n),ShockLocation])
    np.savetxt(NewFileDirectory + '\\ShockLocation-' + CaseName +'-'+str(NewRef)+'.txt', 
               ShockLocationfile,  delimiter = ",")
    
    print("Shock oscillation domain",max(ShockLocation)-min(ShockLocation))
    print("Average Shock oscillation domain",avgDomain)
    
    
    fig1, ax1 = plt.subplots(figsize=(20,500))
    ax1.set_yticks(np.arange(0, n+1, 100))
    ShockResionScale = xPixls*Scale
    ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, n, 0], aspect='0.1', cmap='gray');
    ax1.plot(A, range(n),'x', lw = 1, color = 'g', ms = 3)
    ax1.plot(uncertain, Loc,'x', lw = 1, color = 'r', ms = 2)
    fig1.savefig(NewFileDirectory +f'\\ShockOsc-WithThickness_{SliceThickness}_{CaseName}.png')


AnalysisParameters = np.transpose([AvgAngle,RMAngle,uncertainityRatios,shockXLoc,shockYLoc,AvgOscDomain])
np.savetxt(NewFileDirectory + '\\AnalysisParameters-' + CaseName +'-'+str(NewRef)+'.csv', 
            AnalysisParameters,  delimiter = ",")


# ax.legend(fontsize="14")
# ax2.legend(fontsize="14")

# def exp_fitting(x, A, seg):
#     return A*np.e**(-x/seg)

# popt, pcov = optimize.curve_fit(exp_fitting, SliceThicknesses, uncertainityRatios)

# fit_A, fit_seg = popt
 
# # Sample exp_decay with optimized parameters
# opt_A, opt_seg = popt

# # print(opt_k,opt_B, opt_k_plus)
# print('Opt. A =',opt_A,'Opt. seg =', opt_seg)

# uncertainityRatios_fit = exp_fitting(np.array(SliceThicknesses), opt_A, opt_seg)


# UncertaintyRatio = np.transpose([SliceThicknesses,uncertainityRatios])
# np.savetxt(NewFileDirectory + '\\UncertaintyRatio-' + CaseName +'-'+str(NewRef)+'.txt', 
#             UncertaintyRatio,  delimiter = ",")

# fig3,ax3 = plt.subplots(figsize=(13,10))
# ax3.plot(SliceThicknesses,uncertainityRatios,'s')
# ax3.plot(SliceThicknesses, uncertainityRatios_fit, linestyle = '--', color = 'k')
# ax3.set_ylabel("Uncertainty ratio [\%]"); 
# ax3.set_xlabel("Slice thicknesses [mm]");
# ax3.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
# ax3.minorticks_on()
# ax3.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

# fig3.savefig(NewFileDirectory +'\\Uncertainty ratio'+ CaseName +'.png')

# AvgOscillationDomain = np.transpose([SliceThicknesses,AvgOscDomain])
# np.savetxt(NewFileDirectory + '\\AverageOscillationDomain-' + CaseName +'-'+str(NewRef)+'_ts_'+str(thickness)+'.txt', 
#             AvgOscillationDomain,  delimiter = ",")

# fig4,ax4 = plt.subplots(figsize=(13,10))
# ax4.plot(SliceThicknesses,AvgOscDomain,'-s')
# ax4.set_ylabel("Average oscillation domain [mm]"); 
# ax4.set_xlabel("Slice thicknesses [mm]");
# ax4.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
# ax4.minorticks_on()
# ax4.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
# fig4.savefig(NewFileDirectory +'\\Average oscillation domain-' + CaseName +'.png')


