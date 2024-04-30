# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: Ahmed H. Hanfy
"""
import os
import numpy as np
import pandas as pd
from ShockOscillationAnalysis import inclinedShockTracking as IncTrac


if __name__ == '__main__':
    f = 5000
    D = 60     # Distance in mm
    # imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
    # imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
    imgPath = r'D:\TFAST\TEAMAero experiments\Inlet valve (Second campaign)\Fully Open\2022_11_23- fully opend without 3rd plate - NoSuction\10kHz\ColorCorrected\*.png'
    inflow_path = r'..\InflowEXPMid-Re_NoMesh.csv'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Inlet valve (Second campaign)\Fully Open\2022_11_22 - fully opend with 3rd plate - NoSuction\10kHz\ColorCorrected\*.png'
    Folders = imgPath.split("\\")
    FileDirectory = ''
    for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
    NewFileDirectory = os.path.join(FileDirectory, "IncTrac_test3")
    if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)
    
    DfEXPRe = pd.read_csv(inflow_path)
    inflow_dir = (np.arctan(DfEXPRe['Vy']/DfEXPRe['Vx'])*180/np.pi).tolist()
    print(inflow_dir)
    
    # IncTrac = IncTrac(f,D)
    # IncTrac.ShockPointsTracking(imgPath, scale_pixels = False,
    #                             # Ref_x0 = [108, 725],Ref_y0 = 619,
    #                             tracking_V_range = [8, 134],# fully open without 3rd plate
    #                             # tracking_V_range = [31, 148],
    #                             nPnts = 4,
    #                             # inclination_info = [80,(368, 0), (0, 823)],
    #                             # inclination_info = [50,(241, 0), (108, 268)],# fully open with 3rd plate
    #                             inclination_info = [80,(269, 0), (147, 216)], # fully open without 3rd plate
    #                             slice_thickness = 10,
    #                             n_files = 1000,
    #                             # points_color = 'tab:orange',
    #                             points_opacity = 0.5,
    #                             # avg_preview_mode = 'avg_all',
    #                             avg_preview_mode = 'avg_ang',
    #                             # avg_preview_mode = None,
    #                             # avg_lin_color ='azure',
    #                             avg_ang_Yloc = 175,# fully open without 3rd plate
    #                             # avg_ang_Yloc = 185,# fully open without 3rd plate
    #                             avg_lin_opacity = 0.8,
    #                             avg_txt_size = 30,
    #                             Mach_ang_mode ='Mach_num',
    #                             M1_color = 'yellow',
    #                             M1_txt_size = 16,
    #                             arc_dia = 50,
    #                             OutputDirectory = NewFileDirectory,
    #                             )