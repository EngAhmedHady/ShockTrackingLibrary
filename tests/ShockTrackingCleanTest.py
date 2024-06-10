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
from ShockOscillationAnalysis import SOA, SliceListGenerator


from scipy import signal
plt.rcParams.update({'font.size': 25})
# px = 1/plt.rcParams['figure.dpi']

f = 2000
D = 80     # Distance in mm
slice_loc = 7   # Horizontal line position in mm[slice location to a reference line]
# Scale = 0.12987012987012986
# Scale = 0.13008130081300814
# Scale = 0.13082039911308205 #PS
# Scale = 0.13024282560706402
Scale = 0.136986301369863
SliceThickness = 60
WR = {'Ref_x0': [0, 1024], 'Ref_y1': 198, 
      'avg_shock_angle': 59.240724672559004, 'avg_shock_loc': 132.00132465155573}

n = -1;
# imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\10kHz_smooth P1 (Test 8)\\*.png'
# imgPath = 'D:\\PhD\\TEAMAero\\2023_03_29 - Fully Open with suction\\Oil-sch test 8 - Final full covered 25fps - 5%\\0_20230329_164729\\*.png'
# imgPath = '*.png'
# imgPath = 'tests\\LowRe\\Leadingedge\\*.png'
# imgPath = 'tests\\LowRe\\Passage\\NotRotated\\*.png'
# imgPath = 'tests\\LowRe\\Passage\\Rotated\\*.png'
# imgPath = '..\\Shock wave pixels\\RoughnessComp\\Ref-P3\\*.png'
# imgPath = 'D:\\PhD\\TEAMAero\\Reference case\\2kHz\\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
imgPath =r'D:\TFAST\TEAMAero experiments\2023_06_12\2kHz_Ref P5_fullSuc_4sec (Test 11)\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Reference Case data\\2022_07_29_FastShileren\\5kHz\\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Fully Open\\2023-03-29_Sync_Oil_sch-fully open\\Oil-sch test 8 - Final full covered\\Schlieren\\*.png'
# imgPath = 'D:\\TFAST\TEAMAero experiments\\2023_05_10\\Smooth-2kHz-5sec (test 5)\\*.png'
# CaseName = '17suc-ReflecEff-Leading'
# CaseName = f'Passage_Rotated-{HLP}mm'
# CaseName = 'LE-R'
# CaseName = 'MidRe-5deg'
# CaseName = 'LE-Ref_2kHz'
# CaseName = 'LE-High-Re'
# CaseName = '17suc-fullsuc-ReflecEff-passage'
CaseName = '100suc-fullsuc-base-passage'

Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (f'{Folders[i]}\\')
# NewFileDirectory = os.path.join(FileDirectory, "15mm-Rotated-tk60")
# NewFileDirectory = os.path.join(FileDirectory, "7mm-NotRotated-tk60")
# NewFileDirectory = os.path.join(FileDirectory, "7mm")
# NewFileDirectory = FileDirectory
NewFileDirectory = os.path.join(FileDirectory, "Opt-code-with_rotation2")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)

SA = SOA(f, D)
ImpImg = SliceListGenerator(f,D)


ShockwaveRegion ,n ,WR, Scale = ImpImg.GenerateSlicesArray(imgPath,
                                                        # n_files = 500,
                                                        # within_range = [100,300],
                                                        # every_n_files = 10,
                                                        scale_pixels = True,
                                                        slice_loc = slice_loc, #mm
                                                        full_img_width = True,
                                                        # Ref_x0 = [109, 726], Ref_y0 = 618, #Ref_y1 = 565,  #P1_2kHz
                                                        # Ref_x0 = [240, 693], Ref_y0 = 311,#Ref_y1 = 199, #MidRe-Test 8-leading edge (15mm)
                                                        #                       Ref_y1 = 260, #MidRe-Test 8-leading edge (7mm)
                                                        # Ref_x0 = [109, 726], Ref_y1 = 276, #P1_3kHz
                                                        # Ref_x0 = [111, 725], Ref_y1 = 75,  #P1-6kHz
                                                        # Ref_x0 = [111, 725], Ref_y1 = 67,  #P1-10kHz
                                                        Ref_x0 = [129, 713], Ref_y0 = 595,   #P5-2kHz-suc study
                                                        slice_thickness = SliceThickness, 
                                                        shock_angle_samples  = 2400, 
                                                        angle_samples_review = 10,
                                                        # n_files = 300,
                                                        inclination_est_info = 110,
                                                        # inclination_est_info =  [110, (474, 591),(463, 482)],
                                                        # inclination_est_info = [93,(87, 296),(124, 210)], #MidRe-Test 8-leading edge
                                                        # inclination_est_info = [147,(423, 292),(416, 217)], #MidRe-Test 8
                                                        # inclination_est_info = [120,(423, 292),(416, 217)], #MidRe-Test 8
                                                        # inclination_est_info = [100,(83, 293),(143, 187)], #MidRe-Test 8
                                                        # inclination_est_info = [80,(473,591),(466,514)], #P1_2kHz
                                                        # inclination_est_info = [80,(473,591),(466,514)], #P1_2kHz
                                                        # inclination_est_info = [90,(491,302),(488,264)], #P1_3kHz
                                                        # inclination_est_info = [90,(479,99 ),(478, 46)], #P1-6kHz  
                                                        # inclination_est_info = [90,(470, 129),(459, 5)], #P1-10kHz
                                                        # avg_shock_angle = 88, avg_shock_loc = 467,
                                                        output_directory = NewFileDirectory,
                                                        comment= f'{CaseName}',
                                                        preview =  True)

# ImgList = cv2.imread(NewFileDirectory+'\\2.0kHz_50mm_0.1360544217687075mm-px_ts_10_slice.png')
# File = f"{f/1000}kHz_{HLP}mm_{Scale}mm-px_ts_{SliceThickness}_slice_{CaseName}.png"
File = f"{f/1000}kHz_{slice_loc}mm_{Scale}mm-px_tk_{SliceThickness}px_{CaseName}.png"

print(NewFileDirectory+'\\'+File)
if os.path.exists(fr"{NewFileDirectory}\{File}"): ImgList = cv2.imread(fr"{NewFileDirectory}\{File}")
else: 
    print(f'{NewFileDirectory}\{File} is not exist!!')
    sys.exit()

# spacify the shock region
# (Draw 2 vertical lines)
NewRef = SA.LineDraw(ImgList, 'V', 0, 1)
NewRef = SA.LineDraw(SA.clone, 'V', 1)
NewRef.sort()
       
# or (Spacify x location of 2 vertical lines)
# NewRef = [294, 442] # 2kHz-P3
# NewRef = [263, 430] # 5kHz-P3
# NewRef = [269, 430]
# NewRef = [315, 477]
# NewRef = [67, 171] # LE-2kHz P1
# NewRef = [383, 552] # 2kHz P1
# NewRef = [280, 443] # 2kHz P1
# NewRef = [288, 431] # 3kHz P1
# NewRef = [103, 250] # MidRe-Test8-P3- Passage (7,15mm)
# NewRef = [55, 141] # MidRe-Test8-P3
# NewRef = [283, 440] # 10kHz P1
# NewRef = [47, 140] #MidRe-Test 8-leading edge (7mm)
# NewRef = [90, 166] #MidRe-Test 8-leading edge (15mm)

# NewRef = [372, 538] #P5-2kHz-suc study
ShockwaveRegion = ImgList[:,NewRef[0]:NewRef[1]]
xPixls = (NewRef[1]-NewRef[0])
ShockResionScale = xPixls*Scale
print('Image scale: ' , Scale, 'mm/px') # indicates the displacement accuracy
print('Shock Regions:',NewRef,'\t Represents:' ,xPixls, 'px \t Shock Regions in mm:', ShockResionScale)

# Image cleaning [subtracting the average, subtracting ambiant light frequency]

# ShockwaveRegion = SA.Average(ShockwaveRegion)
# ShockwaveRegion = SA.CleanSnapshots(ShockwaveRegion,
#                                     'Average', 'FFT', #'Brightness and Contrast',
#                                     filterCenter = [(0, 233)], D = 20, n = 5,
#                                     #Brightness = 2, Contrast = 1.7, Sharpness = 2,
#                                     ShowIm = False)

ShockwaveRegion = SA.CleanSnapshots(ShockwaveRegion,
                                    'Average',
                                    'FFT',
                                    # 'SL_Brightness/Contrast',
                                    # 'FFT',
                                    # filterCenter = [(0, 233)], D = 20, n = 5,
                                    filterCenter = [(0, 190)], D = 20, n = 5,
                                    # filterCenter = [(0, 465), (-10, 465), (10, 465), (0, 495)], D = 25, n = 5,
                                    # Brightness = 0.5, Contrast = 2, Sharpness = 1.5,
                                    # Brightness = 1.15, Contrast = 2, Sharpness = 1,
                                    ShowIm = False)

# Find shock location
# function inputs: 1- the oscillation domain image where the shock will be tarcked
#                  2- reviewInterval: to plot the image slices in defined interval (to evaluate the tracking if needed)
#                     [defult is [0,0] which mean notheing to review]
#                  3- Signalfilter: ['median','Wiener','med-Wiener']
# 

ShockLocation, Uncer = SA.ShockTrakingAutomation(ShockwaveRegion, 
                                                  # method = 'darkest_spot',
                                                  method = 'maxGrad',
                                                  
                                                  # method = 'integral',
                                                  # review_slice_tracking = [10,15],
                                                  # review_slice_tracking = [1780,1790], 
                                                  Signalfilter = 'med-Wiener')
                                                  # Signalfilter = 'Wiener')
                                                  # Signalfilter = 'median')
                                                  # Signalfilter = None)

print('uncertainty ratio:', round((len(Uncer)/len(ShockLocation))*100,2),'%')

uncertain = []; Loc = []
for i in Uncer:
    uncertain.append(i[1]*Scale)
    Loc.append(i[0])

print(len(ShockwaveRegion))
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
    if len(Loc) > k and i == Loc[k]: uncer = 0; k +=1
    ShockLocationfile.append([i, ShockLocation[i], uncer])
    


# ShockLocationfile = np.transpose([range(n),ShockLocation])
# ShockLocationfile = np.transpose(ShockLocationfile)

if len(WR) > 3:
    np.savetxt(f'{NewFileDirectory}\\ShockLocation_{slice_loc}mm_{CaseName}{WR["avg_shock_angle"]:.0f}deg.txt', 
    # np.savetxt(f'{NewFileDirectory}\\ShockLocation_{CaseName}-{NewRef}-{round(WR[3])}deg.txt', 
                ShockLocationfile,  delimiter = ",")
else:
    np.savetxt(f'{NewFileDirectory}\\ShockLocation_{slice_loc}mm_{CaseName}.txt', 
                ShockLocationfile,  delimiter = ",")

print("Shock oscillation domain",max(ShockLocation)-min(ShockLocation))
print("Average Shock oscillation domain",avgDomain)

# Apply welch method for PSD
Freq, psd = signal.welch(x = ShockLocation, fs = f, window='barthann',
                      nperseg = 512, noverlap=0, nfft=None, detrend='constant',
                      return_onesided=True, scaling='density')



fig,ax = plt.subplots(figsize=(10,10))

# choose which is more convenient (log on both axes or only on x)
ax.loglog(Freq, psd, lw = '2')
# ax.semilogx(Freq, psd, lw = '2')   
ax.set_ylabel(r"PSD [mm$^2$/Hz]"); 
ax.set_xlabel("Frequency [Hz]");
# ax.set_title('Shock displacement PSD')
ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax.minorticks_on()
ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

if len(WR) > 3:
    fig.savefig(fr"{NewFileDirectory}\{f/1000}kHz_{slice_loc}mm_tk_{SliceThickness}_{CaseName}_{round(WR['avg_shock_angle'])}deg_DispPSD.png",bbox_inches='tight', pad_inches=0.1)
else:
    fig.savefig(fr"{NewFileDirectory}\{f/1000}kHz_{slice_loc}mm_tk_{SliceThickness}_{CaseName}_DispPSD.png",bbox_inches='tight', pad_inches=0.1)

# To define the peakes on PSD signal with horizontal line
# ax.axvline(x = 19.5,ls='--',color='k',alpha=0.4)
# ax.axvline(x = 287,ls='--',color='k',alpha=0.4)


# Find velocity oscillation using finite difference method
# (centeral difference in general and forward/backward at boundaries)
fig2,ax2 = plt.subplots(figsize=(10,10))
T = n/f

t = np.linspace(0,T,n);


V = SA.VelocitySignal(ShockLocation, T)
Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
                      nperseg = f*512/2000, noverlap=0, nfft=None, detrend='constant',
                      return_onesided=True, scaling='density')

# ax2.semilogx(Freq2, psd2 , lw = '2')

print("Total measuring time: ", T, "sec")

# dx_dt = []; dt = T/n; t = np.linspace(0,T,n);
# for xi in range(n):
#     if xi > 0 and xi < n-1:
#         dx_dt.append((ShockLocation[xi+1]-ShockLocation[xi-1])/(2*dt*1000))
#     elif xi == 0: 
#         dx_dt.append((ShockLocation[xi+1]-ShockLocation[xi])/(dt*1000))
#     elif xi == n-1:
#         dx_dt.append((ShockLocation[xi]-ShockLocation[xi-1])/(dt*1000))
        
# V_avg = np.mean(dx_dt) 
# V = dx_dt - V_avg

# Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
#                       nperseg = f*512/2000, noverlap=0, nfft=None, detrend='constant',
#                       return_onesided=True, scaling='density')

# Find maximum peak in velocity PSD
domFreq = Freq2[psd2.argmax(axis=0)]

print('max peak at:', domFreq, 'Hz')


ax2.semilogx(Freq2, psd2 , lw = '2')
ax2.axvline(x =Freq2[psd2.argmax(axis=0)],ls='--',color='k',alpha=0.4)
ax2.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
ax2.set_xlabel("Frequency [Hz]");
# ax2.set_title('PSD for '+ CaseName +' above LE')
# ax2.set_title('Shock velocity PSD')
ax2.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax2.minorticks_on()
ax2.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
if len(WR) > 3:
    fig2.savefig(fr"{NewFileDirectory}\{f/1000}kHz_{slice_loc}mm_tk_{SliceThickness}_{CaseName}_{round(WR['avg_shock_angle'])}deg_VelPSD.png",bbox_inches='tight', pad_inches=0.1)
else:
    fig2.savefig(fr"{NewFileDirectory}\{f/1000}kHz_{slice_loc}mm_tk_{SliceThickness}_{CaseName}_VelPSD.png",bbox_inches='tight', pad_inches=0.1)   


fig1, ax1 = plt.subplots(figsize=(30,500))
# fig1, ax1 = plt.subplots(figsize=(5,10))
ax1.set_yticks(np.arange(0, n+1, 50))
xPixls = (NewRef[1]-NewRef[0])
ShockResionScale = xPixls*Scale
# ShockwaveRegion = SA.CleanSnapshots(ShockwaveRegion,
#                                     'Average', 'FFT', #'Brightness and Contrast',
#                                     filterCenter = [(0, 233)], D = 20, n = 5,
#                                     #Brightness = 2, Contrast = 1.7, Sharpness = 2,
#                                     ShowIm = False)

ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, n, 0], aspect='0.1', cmap='gray');
# plt.close(fig1)
# ax1.plot([0,ShockResionScale],[25,25],'r--')
# ax1.plot([0,ShockResionScale],[122,122],'r--')
# ax1.plot([0,ShockResionScale],[133,133],'r--')
ax1.invert_yaxis()
ax1.set_ylabel(r"snapshots (n)"); 
ax1.set_xlabel(r"Shock oscillation domain ($x$) [mm]");
ax1.plot(A, range(n),'x', lw = 1, color = 'g', ms = 7)
ax1.plot(uncertain, Loc,'x', lw = 1, color = 'r', ms = 5)
if len(WR) > 3:
    fig1.savefig(fr"{NewFileDirectory}\{f/1000}kHz_{slice_loc}mm_tk_{SliceThickness}_{CaseName}_{round(WR['avg_shock_angle'])}deg_TrackedPoints.png",bbox_inches='tight', pad_inches=0.1)
else:
    fig1.savefig(fr"{NewFileDirectory}\{f/1000}kHz_{slice_loc}mm_tk_{SliceThickness}_{CaseName}_TrackedPoints.png",bbox_inches='tight', pad_inches=0.1)

# ax1.set_ylim([0,400])
# plt.show()

