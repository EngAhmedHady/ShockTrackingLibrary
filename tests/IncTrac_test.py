# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: Ahmed H. Hanfy
"""
from ShockOscillationAnalysis import InclinedShockTracking as IncTrac

if __name__ == '__main__':
    # Define the snapshots path with glob [note the extention of imported files]
    imgPath = r'test_files\raw_images\*.png'
    
    # iniate the inclined shock tracking module
    IncTrac = IncTrac()
    
    # use ShockTracking function
    IncTrac.ShockPointsTracking(imgPath, 
                                scale_pixels = False,
                                tracking_V_range = [575, 200], # as not scaled tracking reference values in pixels
                                nPnts = 9,                     # number of slices         
                                inclination_info = 110,        # width of each slice
                                preview = True,                # to preview the final setup before proceeding
                                slice_thickness = 4,           # number of vertical pixels to be be averaged for each slice
                                points_opacity = 0.5,          # displayed tracked points transparency
                                avg_preview_mode = 'avg_ang',  # to display the estimated shock angle for each snapshot
                                avg_show_txt = True,           # to display the estimated shock angle value
                                avg_txt_Yloc = 650,            # y-location of the estimated angle value in pixels
                                avg_txt_size = 30,             # font size of estimated angle value in pt
                                osc_boundary = True,           # to display the oscilation domain
                                )