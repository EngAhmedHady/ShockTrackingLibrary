# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:05:30 2024

@author: Ahmed H. Hanfy
"""

import cv2
import numpy as np
# from .ShockOscillationAnalysis import CVColor
from .linedrawingfunctions import InclinedLine, AngleFromSlope
from matplotlib.patches import Arc, FancyArrowPatch



def AvgAnglePlot(ax, img_shp, P, slope, angle , **kwargs):
    avg_ang_Yloc = kwargs.get('avg_ang_Yloc', img_shp[0]-100)
    avg_lin_color = kwargs.get('avg_lin_color', 'w')
    avg_lin_opacity = kwargs.get('avg_lin_opacity', 1)
    avg_txt_size = kwargs.get('avg_txt_size', 26)
    
    P1,P2,avg_slope,a = InclinedLine(P,slope = slope ,imgShape = img_shp)
    X = int((avg_ang_Yloc-a)/slope) if slope != 0 else avg_ang_Yloc

    avg_ang_arc = Arc((X, avg_ang_Yloc),80, 80, theta1= -angle , theta2 = 0, color = avg_lin_color)
    ax.add_patch(avg_ang_arc);
    ax.plot([P1[0],P2[0]], [P1[1],P2[1]], lw = 2,
            color= avg_lin_color, linestyle = (0, (20, 3, 5, 3)), alpha = avg_lin_opacity)
    ax.text(X + 40 ,avg_ang_Yloc-10 , f'${{{angle:0.2f}}}^\circ$', 
            color = avg_lin_color, fontsize = avg_txt_size);
    ax.plot([X-10,X+100], [avg_ang_Yloc,avg_ang_Yloc], lw = 1, 
            color = avg_lin_color, alpha = avg_lin_opacity)


def plot_review(ax, img, shp, x_loc, column_y, uncertain, uncertain_y, avg_slope, avg_ang,
                mid_loc = -1, y = -1, avg_preview_mode = None,Mach_ang_mode = None, **kwargs):
    points_color = kwargs.get('points_color', 'yellow')
    points_opacity = kwargs.get('points_opacity', 1)
    uncertain_point_color = kwargs.get('uncertain_point_color', 'red')
    
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8))
    ax.plot(x_loc, column_y, '-o', color=points_color, ms=12, alpha = points_opacity)
    ax.plot(uncertain, uncertain_y, 'o', color=uncertain_point_color, ms=12, alpha = points_opacity)

    # ploting the middle point as possible center of rotation
    # if mid_loc > 0 and y > 0: ax.plot(mid_loc, y, '*', color='g', ms=10)
    if avg_preview_mode != None:
        AvgAnglePlot(ax, shp, (mid_loc,y), avg_slope, avg_ang, **kwargs)

    if Mach_ang_mode =='Mach_num':
        inflow_dir_deg = kwargs.get('inflow_dir_deg', np.zeros(len(column_y)))
        inflow_dir_rad = kwargs.get('inflow_dir_rad', np.zeros(len(column_y)))
        M1_color = kwargs.get('M1_color', 'orange')
        M1_txt_size = kwargs.get('M1_txt_size', 26)
        arw_len = kwargs.get('arw_len', 50)
        arc_dia = kwargs.get('arc_dia', 80)
        for i in range(1,len(column_y)):
            p1 = (x_loc[i],column_y[i]); p2 = (x_loc[i-1],column_y[i-1]);
            _,_,m,_ = InclinedLine(p1,p2,imgShape = shp)
            xlen = np.cos(inflow_dir_rad[i]); ylen = np.sin(inflow_dir_rad[i])
            local_ang = AngleFromSlope(m)
            inflow_ang = local_ang + inflow_dir_deg[i]
            ax.text(p1[0]+40 ,p1[1]- 5 , f'${{{inflow_ang:0.2f}}}^\circ$',
                    size = M1_txt_size, color = M1_color);
            ax.plot([p1[0]-arw_len*xlen,p1[0]+60*xlen], [p1[1]-arw_len*ylen,p1[1]+60*ylen],color = M1_color, lw = 1)
            
            arc1 = Arc(p1,arc_dia, arc_dia, theta1=-local_ang, theta2=0+inflow_dir_deg[i], color = M1_color)
            ax.add_patch(arc1);
            M1 = 1/np.sin((inflow_ang)*np.pi/180)
            arr = FancyArrowPatch((p1[0] - arw_len*xlen, p1[1] - arw_len*ylen), p1,
                               arrowstyle='-|>, head_length=20, head_width=3, widthA=2', color=M1_color)
            ax.add_patch(arr)
            ax.annotate(f'M$_1 ={{{M1:0.2f}}}$', xy=p1,
                        color=M1_color, size = M1_txt_size,
                        xytext=(p1[0] - arw_len*xlen, p1[1] + arw_len*ylen),
                        horizontalalignment='right', verticalalignment='center')


def PreviewCVPlots(img, Ref_x0 = [], Ref_y = [], 
                   tk = [], avg_shock_loc = [], **kwargs):
    shp = img.shape;
    if len(Ref_x0):
        Ref_x0_color = kwargs.get('Ref_x0_color', CVColor.GREEN)
        cv2.line(img, (Ref_x0[0],0), (Ref_x0[0],shp[0]), Ref_x0_color, 1)
        cv2.line(img, (Ref_x0[1],0), (Ref_x0[1],shp[0]), Ref_x0_color, 1)

    if len(tk)== 2:
        tk_color = kwargs.get('tk_color', CVColor.GREENBLUE)
        cv2.line(img, (0, tk[0]), (shp[1], tk[0]), tk_color, 1)
        cv2.line(img, (0, tk[1]), (shp[1], tk[1]), tk_color, 1)

    Ref_y1_color = kwargs.get('Ref_y1_color', CVColor.FUCHSIPINK)
    if hasattr(Ref_y, "__len__"):
        if len(Ref_y) > 2: cv2.line(img, (0,Ref_y[1]), (shp[1],Ref_y[1]), Ref_y1_color, 1)
        if Ref_y[0] > -1:
            Ref_y0_color = kwargs.get('Ref_y2_color', CVColor.YELLOW)
            cv2.line(img, (0,Ref_y[0]), (shp[1],Ref_y[0]), Ref_y0_color, 1)
    elif Ref_y > 0: cv2.line(img, (0,Ref_y), (shp[1],Ref_y), Ref_y1_color, 1)

    avg_shock_loc_color = kwargs.get('avg_shock_loc_color', CVColor.CYAN)
    if hasattr(avg_shock_loc, "__len__") and len(avg_shock_loc) > 2:
        cv2.line(img, avg_shock_loc[0], avg_shock_loc[1], avg_shock_loc_color, 1)
    else: cv2.line(img, (int(avg_shock_loc), 0), (int(avg_shock_loc),shp[0]), avg_shock_loc_color, 1)

    return img