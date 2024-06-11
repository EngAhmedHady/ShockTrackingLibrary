# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
from ShockOscillationAnalysis import InclinedShockTracking as IncTrac

if __name__ == '__main__':
    # Define the snapshots path with glob[note the extention of imported files]
    imgPath = r'test_files\raw_images\*.png'
    
    # Define the velocity vectors as Vx and Vy with the vertical coordinates y
    inflow_path = r'test_files\upstream_Mwave_vel.csv'
    Vxy = np.genfromtxt(inflow_path, delimiter=',', skip_header = 1)
    
    # iniate the inclined shock tracking module
    IncTrac = IncTrac(D = 80)
    
    # use ShockTracking function
    IncTrac.ShockPointsTracking(imgPath, 
                                scale_pixels = True,
                                tracking_V_range = [5, 13],  # as scaled tracking reference values in mm
                                nPnts = 3,                     # number of slices         
                                inclination_info = 50,         # width of each slice
                                points_opacity = 0.5,          # displayed tracked points transparency
                                avg_preview_mode = 'avg_all',  # to display the estimated shock angle for each snapshot
                                avg_txt_Yloc = 650,            # y-location of the estimated angle value in pixels
                                avg_txt_size = 30,             # font size of estimated angle value in pt
                                flow_Vxy = Vxy,
                                angle_interp_kind = 'linear',
                                preview_angle_interpolation = True,
                                Mach_ang_mode ='Mach_num',
                                M1_color = 'yellow', 
                                M1_txt_size = 18,
                                arc_dia = 50,
                                )