# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: Ahmed H. Hanfy
"""
import sys
sys.path.append(r'..\ShockOscillationAnalysis')
# from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
from inc_tracking import InclinedShockTracking as IncTrac

if __name__ == '__main__':
    # Define the snapshots path with glob [note extention of imported files]
    imgPath = r'test_files\raw_images\*.png'

    # iniate the inclined shock tracking module
    IncTrac = IncTrac()

    # use ShockTracking function
    print(IncTrac.ShockPointsTracking(imgPath,
                                      scale_pixels=False,
                                      # as not scaled tracking values in pixels
                                      tracking_V_range=[575, 200],
                                      nPnts=9,               # number of slices
                                      # width of each slice
                                      inclination_info=110,
                                      # preview the final pre-proceeding setup
                                      preview=True,
                                      # No. of vertical px to average per slice
                                      slice_thickness=4,
                                      # displayed tracked points transparency
                                      points_opacity=0.5,
                                      # displays the est. angle per snapshot
                                      avg_preview_mode='avg_ang',
                                      # to display the est. shock angle value
                                      avg_show_txt=True,
                                      # y-loc. of the est. angle value in px
                                      avg_txt_Yloc=650,
                                      # font size of est. angle value in pt
                                      avg_txt_size=30,
                                      # to display the oscilation domain
                                      osc_boundary=True,
                                      ))
