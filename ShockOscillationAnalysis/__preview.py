# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:05:30 2024

@author: Ahmed H. Hanfy
"""

import cv2

def plot_review(ax, img, x_loc, column_y, uncertain, uncertain_y, mid_loc, y):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.plot(x_loc, column_y, '-o', color='yellow', ms=10)
    ax.plot(uncertain, uncertain_y, 'o', color='r', ms=10)
    ax.plot(mid_loc, y, '*', color='g', ms=10)
    
    
def PreviewCVPlots(img, Ref_x0 = [], Ref_y = [], 
                   tk = [], avg_shock_loc = [], **kwargs):
    shp = img.shape;
    if len(Ref_x0):
        Ref_x0_color = kwargs.get('Ref_x0_color', (0,255,0))
        cv2.line(img, (Ref_x0[0],0), (Ref_x0[0],shp[0]), Ref_x0_color, 1)
        cv2.line(img, (Ref_x0[1],0), (Ref_x0[1],shp[0]), Ref_x0_color, 1)
        
    if len(tk)== 2:
        tk_color = kwargs.get('tk_color', (255, 128, 0))
        cv2.line(img, (0, tk[0]), (shp[1], tk[0]), tk_color, 1)
        cv2.line(img, (0, tk[1]), (shp[1], tk[1]), tk_color, 1)
        
    Ref_y1_color = kwargs.get('Ref_y1_color', (255, 128, 255))
    if hasattr(Ref_y, "__len__"):
        if len(Ref_y) > 2: cv2.line(img, (0,Ref_y[1]), (shp[1],Ref_y[1]), Ref_y1_color, 1)
        if Ref_y[0] > -1:
            Ref_y0_color = kwargs.get('Ref_y2_color', (0, 255, 255))
            cv2.line(img, (0,Ref_y[0]), (shp[1],Ref_y[0]), Ref_y0_color, 1)
    elif Ref_y > 0: cv2.line(img, (0,Ref_y), (shp[1],Ref_y), Ref_y1_color, 1)
    
    avg_shock_loc_color = kwargs.get('avg_shock_loc_color', (255, 255, 0))
    if hasattr(avg_shock_loc, "__len__") and len(avg_shock_loc) > 2:
        cv2.line(img, avg_shock_loc[0], avg_shock_loc[1], avg_shock_loc_color, 1)
    else: cv2.line(img, (int(avg_shock_loc), 0), (int(avg_shock_loc),shp[0]), avg_shock_loc_color, 1)
    
    return img