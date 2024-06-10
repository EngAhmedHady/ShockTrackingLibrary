# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:23:24 2024

@author: admin
"""
import os
import sys
import cv2
import numpy as np
from scipy import signal
import matplotlib.ticker
import matplotlib.pyplot as plt
from ShockOscillationAnalysis import SOA, sliceListGenerator
matplotlib.ticker.ScalarFormatter(useOffset=True,useMathText=True)
plt.rcParams.update({'font.size': 25})


f = 2000              # Sampling rate in Hz
D = 80                # Distance in mm
slice_loc = 55        # Horizontal line position in mm [slice location to a reference line]
slice_thickness = 20  # Number of pixels to be averaged for traking in px

main_directory = r'D:\TFAST\TEAMAero experiments\2023_06_12'

Folders = ['2kHz_Ref P5_fullSuc_4sec (Test 11)',
           '2kHz_Ref P5_1-3Suc_4sec (Test 8)',
           '2kHz_Ref P5_1-2Suc_4sec (Test 7)\Clean',
           '2kHz_Ref P5_2-3Suc_4sec (Test 4)',
           '2kHz_Ref P5_5-6Suc_4sec (Test 2)' ]

Cases = {4: r'$16.7\%$-suc', 
         3: r'$33.3\%$-suc',
         2: r'$50.0\%$-suc', 
         1: r'$66.7\%$-suc',
         0: r'Full-suc'}

Shocks = {'Leading': 'Leading edge Shock', 
          'passage': 'Passage Shock', 
          'Reflec' : 'Reflected Shock'}


fig = plt.figure(figsize=(32,20))
plt.subplots_adjust(hspace=0.3, wspace=0.8)
ax = []
ax.append(plt.subplot2grid((2,6), (0,1), colspan=2))
ax.append(plt.subplot2grid((2,6), (0,3), colspan=2))

SA = SOA(f, D)
ImpImg = sliceListGenerator(f,D)

for i in range(len(Folders)):
    case_path = fr'{main_directory}\{Folders[i]}\*.png'
    NewFileDirectory = os.path.join(f'{main_directory}\{Folders[i]}', "55mm_LRP-shocks_analysis")
    if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)
    ShockwaveRegion ,n ,WR, Scale = ImpImg.GenerateSlicesArray(case_path,
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
                                                        slice_thickness = slice_thickness, 
                                                        shock_angle_samples  = 2400, 
                                                        angle_samples_review = 10,
                                                        inclination_est_info = 60,
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
                                                        OutputDirectory = NewFileDirectory,
                                                        # comment= f'{Cases[i]}_{Shocks[i]}',
                                                        preview =  True)