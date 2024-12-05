# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: Ahmed H. Hanfy
"""
import sys
import numpy as np
sys.path.append(r'..\ShockOscillationAnalysis')
from ShockOscillationAnalysis import InclinedShockTracking as IncTrac

if __name__ == '__main__':
    # Define the snapshots path with glob[note the extention of imported files]
    imgPath = r'test_files\raw_images\*.png'

    # Define the velocity vectors as Vx and Vy with the vertical coordinates y
    inflow_path = r'test_files\upstream_Mwave_vel.csv'
    Vxy = np.genfromtxt(inflow_path, delimiter=',', skip_header=1)

    # iniate the inclined shock tracking module
    IncTrac = IncTrac(D=80)

    # use ShockTracking function
    IncTrac.ShockPointsTracking(imgPath,
                                scale_pixels=True,
                                # as scaled tracking reference values in mm
                                tracking_V_range=[5, 13],
                                nPnts=3,                  # number of slices
                                inclination_info=50,      # width of each slice
                                # displayed tracked points transparency
                                points_opacity=0.5,
                                # display the est. shock angle for per snapshot
                                avg_preview_mode='avg_all',
                                # y-location of the estimated angle value in px
                                avg_txt_Yloc=650,
                                # font size of estimated angle value in pt
                                avg_txt_size=30,
                                # inflow velocity vectors [y, Vx, Vy]
                                flow_Vxy=Vxy,
                                # inflow data interpolation with slice points
                                angle_interp_kind='linear',
                                # to plot interpolation values for review
                                preview_angle_interpolation=True,
                                # to show the Mach number values
                                Mach_ang_mode='Mach_num',
                                # the displayed Mach number values color
                                M1_color='yellow',
                                # the Mach number values font size in pt
                                M1_txt_size=18,
                                arc_dia=50,       # the flow angle arc diameter
                                )
