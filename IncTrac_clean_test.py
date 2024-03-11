# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: admin
"""

from ShockOscillationAnalysis import inclinedShockTracking as IncTrac


if __name__ == '__main__':
    f = 5000
    D = 80     # Distance in mm
    # imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
    imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
    
    
    IncTrac = IncTrac(f,D)
    IncTrac.ShockPointsTracking(imgPath, scale_pixels = True,
                                tracking_V_range = [-10,60],
                                nPnts = 5,
                                inclination_info = 80,
                                slice_thickness = 9,
                                n_files = 500)