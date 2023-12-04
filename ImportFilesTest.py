# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:29:02 2023

@author: admin
"""

import os
# from ShockOscillationAnalysis import SOA
from __importImages import importSchlierenImages as SOA


f = 2000    # images sampling rate
D = 80      # distance in mm
# HLP = 10     # Horizontal line position [slice location to a reference line]
# imgPath = 'C:\\Users\\super\\Desktop\\PhD\\TEAMAero\\25-05-2023_2kHz_smooth P1 (Test 5)\\*.png'
# imgPath = '*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Half Open\\2022_11_24 - half opend without 3rd plate - NoSuction\\2.5kHz\\*.png'
# imgPath = 'D:\\TFAST\TEAMAero experiments\\2023_05_10\\Smooth-2kHz-5sec (test 5)\\*.png'
imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'

Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
NewFileDirectory = os.path.join(FileDirectory, "shock_signal")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)

SA = SOA(f,D)
ShockwaveRegion ,n ,H_line, Scale = SA.GenerateSlicesArray(imgPath,
                                                           HLP = 10,
                                                           FullImWidth = False,
                                                           ScalePixels= True,
                                                           OutputDirectory = NewFileDirectory,
                                                           SliceThickness = 60,
                                                           WorkingRange = [109],
                                                           nt = 1500,
                                                           ShockAngleSamples = 1000,
                                                           AngleSamplesReview = 10,
                                                           comment = '-')



