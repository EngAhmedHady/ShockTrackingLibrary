# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:29:05 2024

@author: Ahmed H. Hanfy
"""
import os
import numpy as np
# import pandas as pd
from ShockOscillationAnalysis import InclinedShockTracking as IncTrac


if __name__ == '__main__':
    # f = 5000
    D = 60     # Distance in mm
    # imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
    # imgPath = 'D:\\PhD\\TEAMAero\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Inlet valve (Second campaign)\Fully Open\2022_11_23- fully opend without 3rd plate - NoSuction\10kHz\ColorCorrected\*.png'
    # inflow_path = r'..\InflowEXPMid-Re_NoMesh.csv'
    # inflow_path = r'..\InflowEXPMid-Re.csv'
    # inflow_path = r'..\InflowEXPHi-Re.csv'
    inflow_path = r'..\upstream_Machwave.csv'
    # imgPath = r'..\Test 1\*.jpg'
    imgPath = r'D:\PhD\TEAMAero\2023_03_29 - Fully Open with suction\Oil-sch test 7 - half covered 25fps - 5%\MidRe-Test7\*.png'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Reference Case data\2022_07_29_FastShileren\2kHz\*.png'
    # imgPath = r'..\with full open valve\2022_12_08 - inflow conditions\2022_12_08_*.jpg'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Inlet valve (Second campaign)\Fully Open\2022_12_08 - inflow conditions\2022_12_08_*.jpg'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Inlet valve (Second campaign)\Fully Open\2022_11_22 - fully opend with 3rd plate - NoSuction\2kHz\*.png'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Inlet valve (Second campaign)\Fully Open\2022_11_23- fully opend without 3rd plate - NoSuction\4kHz\*.png'
    # imgPath = r'D:\TFAST\TEAMAero experiments\Reference Case data\2022_06_07\Test 1\*.jpg'
    Folders = imgPath.split("\\")
    FileDirectory = ''
    for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
    NewFileDirectory = os.path.join(FileDirectory, "IncTrac_test3")
    if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)
    
    # DfEXPRe = pd.read_csv(inflow_path)
    # Vxy = list(DfEXPRe.itertuples(index=False, name=None))
    
    Vxy = np.genfromtxt(inflow_path, delimiter=',', skip_header = 1)
    
    # DfEXPRe['alpha'] = np.arctan(DfEXPRe['Vy']/DfEXPRe['Vx'])*180/np.pi
    # inflow_dir = []
    # for i in range(len(DfEXPRe)):
    #     inflow_dir.append([DfEXPRe['y'].iloc[i], DfEXPRe['alpha'].iloc[i]])
    # print(DfEXPRe)
    
    IncTrac = IncTrac(D = D)

    IncTrac.ShockPointsTracking(imgPath, scale_pixels = True,
                                # every_n_files = 3,
                                # resize_img = (1620,1080),
                                # Ref_x0 = [110, 725],Ref_y0 = 616,
                                # Ref_x0 = [129, 713],Ref_y0 = 595,
                                # Ref_x0 = [317, 931],Ref_y0 = 619,
                                # Ref_x0 = [534, 994],Ref_y0 = 662,
                                # Ref_x0 = [523, 997],Ref_y0 = 663,  #3P - Ref (7/6/2022)
                                # tracking_V_range = [8, 134],# fully open without 3rd plate
                                # tracking_V_range = [31, 148], # fully open with 3rd plate
                                # tracking_V_range = [10, 60],
                                # tracking_V_range = [5, 29.7],
                                tracking_V_range = [5, 30],
                                nPnts = 10,
                                inclination_info = 110,
                                # inclination_info = [50, (903, 0), (0, 711)],
                                # inclination_info = [50, (1024, 22), (87, 856)],
                                # inclination_info = [110,(387, 0), (0, 747)],
                                # inclination_info = [65,(1127, 0), (0, 1013)],
                                # inclination_info = [80,(622, 0), (205, 1080)],
                                # inclination_info = [80,(368, 0), (0, 823)],
                                # inclination_info = [50,(241, 0), (108, 268)],# fully open with 3rd plate
                                # inclination_info = [80,(391, 0), (104, 480)], # fully open without 3rd plate
                                slice_thickness = 4,
                                # review_slice_tracking = 0,
                                # within_range = [0,200],
                                n_files = 50,
                                # points_color = 'tab:orange',
                                points_opacity = 0.5,
                                # avg_preview_mode = 'avg_all',
                                avg_preview_mode = 'avg_ang',
                                # avg_preview_mode = None,
                                avg_show_txt = True,
                                # avg_lin_color ='azure',
                                # review_inc_slice_tracking = [30,150],
                                avg_txt_Yloc = 350,# fully open without 3rd plate
                                # avg_txt_Yloc = 185,# fully open without 3rd plate
                                avg_lin_opacity = 0.6,
                                avg_txt_size = 30,
                                # tracking_std = True,
                                # Mach_ang_mode ='Mach_num',
                                M1_color = 'yellow',
                                M1_txt_size = 18,
                                arc_dia = 50,
                                # flow_Vxy = Vxy,
                                output_directory = NewFileDirectory,
                                preview = True,
                                preview_angle_interpolation = False,
                                )