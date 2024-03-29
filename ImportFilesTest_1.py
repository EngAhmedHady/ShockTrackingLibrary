# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:29:02 2023

@author: admin
"""

import os
import time

from ShockOscillationAnalysis.__generateshocksignal import TimeCalculation
# from .__importImages import importSchlierenImages as SOA
# from ShockOscillationAnalysis.__import_images import importSchlierenImagesOld as SA
from ShockOscillationAnalysis.__import_images import importSchlierenImages as SOA_opt


f = 2000    # images sampling rate
D = 80      # distance in mm
# HLP = 10     # Horizontal line position [slice location to a reference line]
# imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
# imgPath = '*.png'
imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Half Open\\2022_11_24 - half opend without 3rd plate - NoSuction\\2.5kHz\\*.png'
# imgPath = 'D:\\TFAST\TEAMAero experiments\\2023_05_10\\Smooth-2kHz-5sec (test 5)\\*.png'
# imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'

Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
NewFileDirectory = os.path.join(FileDirectory, "shock_signal")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)


if __name__ == '__main__':
    # SA = SA(f,D)
    # start_time = time.time()
    
    # ShockwaveRegion ,n ,H_line, Scale = SA.GenerateSlicesArray(imgPath,
    #                                                             HLP = 10,
    #                                                             FullImWidth = False,
    #                                                             ScalePixels= True,
    #                                                             OutputDirectory = NewFileDirectory,
    #                                                             SliceThickness = 60,
    #                                                             WorkingRange = [108,725, 542],
    #                                                             # nt = 9000,
    #                                                             ShockAngleSamples = 1000,
    #                                                             AngleSamplesReview = 10,
    #                                                             inclinationEst = [110,(125,617),(125,617)],
    #                                                             comment = '-')
    # timeInSec =  time.time() - start_time
    # TimeCalculation(timeInSec)
    
    start_time = time.time()
    SA_opt = SOA_opt(f,D)
    ShockwaveRegion ,n ,WR, Scale = SA_opt.GenerateSlicesArray(imgPath,
                                                                n_files = 500,
                                                                # within_range = [100,300],
                                                                # every_n_files = 10,
                                                                scale_pixels = True,
                                                                slice_loc = 10, #mm
                                                                full_img_width = False,
                                                                slice_thickness = 60, 
                                                                shock_angle_samples = 60, 
                                                                angle_samples_review = 10,
                                                                OutputDirectory = NewFileDirectory,
                                                                comment='-opt', 
                                                                # inclination_est_info =  [110,(125,617),(125,617)],
                                                                # Ref_x0 = [108, 725], Ref_y1 = 542, 
                                                                # avg_shock_angle = 88, avg_shock_loc = 467,
                                                                preview =  True)
    
    timeInSec =  time.time() - start_time
    TimeCalculation(timeInSec)