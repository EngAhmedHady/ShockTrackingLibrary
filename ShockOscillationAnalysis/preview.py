# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:05:30 2024

@author: Ahmed H. Hanfy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .ShockOscillationAnalysis import CVColor
from .linedrawingfunctions import InclinedLine, AngleFromSlope
from matplotlib.patches import Arc, FancyArrowPatch

def AvgAnglePlot(ax, img_shp:tuple , P: tuple, slope: float, angle: float,
                 txt: bool = True, lin_color = 'w', lin_opacity = 1, **kwargs) -> None:
    """
    Plot the average angle line and optional text annotation on a given axis.
    This function uses the `InclinedLine` function to determine the end points
    of the line based on the given slope and image shape.
    It then plots the line and an optional text annotation indicating the angle.

    Parameters:
        - **ax (matplotlib.axes.Axes)**: The axis on which to plot.
        - **img_shp (tuple)**: Shape of the image (height, width).
        - **P (tuple)**: A point (x, y) through which the line passes.
        - **slope (float)**: Slope of the line.
        - **angle (float)**: Angle to display as annotation.
        - **txt (bool)**: Whether to show the text annotation for oscilation boundary.
        - `**kwargs`: Additional keyword arguments for customization:
            - **avg_txt_Yloc (int, optional)**: Y location for the text annotation. Default is image height minus 100.
            - **avg_txt_size (int, optional)**: Font size of the text annotation. Default is 26.

    Example:
        >>> fig, ax = plt.subplots()
        >>> img_shp = (600, 800)
        >>> P = (100, 300)
        >>> slope = 0.5
        >>> angle = 45.0
        >>> AvgAnglePlot(ax, img_shp, P, slope, angle, avg_lin_color='r', avg_show_txt=True)
        >>> plt.show()
    """
    # Handle optional parameter values from **kwargs


    avg_txt_Yloc = kwargs.get('avg_txt_Yloc', img_shp[0]-100)
    avg_txt_size = kwargs.get('avg_txt_size', 26)

    # Calculate the inclined line end points
    P1,P2,_,a = InclinedLine(P,slope = slope ,imgShape = img_shp)

    # Calculate the X position for the text annotation
    if slope != 0 and slope != np.inf: X = int((avg_txt_Yloc-a)/slope)
    elif slope == np.inf: X = P[0]
    else: X = avg_txt_Yloc

    # Plot the inclined line
    ax.plot([P1[0],P2[0]], [P1[1],P2[1]], lw = 2,
            color= lin_color, linestyle = (0, (20, 3, 5, 3)), alpha = lin_opacity)

    # Plot the text annotation if enabled
    if txt:
        # Draw an arc to represent the angle
        avg_ang_arc = Arc((X, avg_txt_Yloc),80, 80, theta1= -angle , theta2 = 0, color = lin_color)
        ax.add_patch(avg_ang_arc);

        # Add the text annotation for the angle
        ax.text(X + 40 ,avg_txt_Yloc-10 , f'${{{angle:0.2f}}}^\circ$',
                color = lin_color, fontsize = avg_txt_size);
        # Plot a horizontal line at the text annotation location to compare the inclination angle
        ax.plot([X-10,X+100], [avg_txt_Yloc,avg_txt_Yloc], lw = 1,
                color = lin_color, alpha = lin_opacity)

def plot_review(ax, img, shp, x_loc, column_y, uncertain, uncertain_y, avg_slope, avg_ang,
                mid_loc = -1, y = -1, avg_preview_mode = None, Mach_ang_mode = None, **kwargs):

    """
    Plot review function to visualize shock points and additional features on an image.

    Parameters:
        - **ax (matplotlib.axes._subplots.AxesSubplot)**: The subplot to draw the plot on.
        - **img (np.ndarray)**: The input image to display.
        - **shp (tuple)**: The shape of the image.
        - **x_loc (list)**: List of x-coordinates for the shock points.
        - **column_y (list)**: List of y-coordinates for the shock points.
        - **uncertain (list)**: List of uncertain points.
        - **uncertain_y (list)**: List of y-coordinates for uncertain points.
        - **avg_slope (float)**: The average slope.
        - **avg_ang (float)**: The average angle.
        - **mid_loc (int, optional)**: The middle location. Defaults to -1.
        - **y (int, optional)**: The y-coordinate. Defaults to -1.
        - **avg_preview_mode (bool, optional)**: Flag indicating whether to show average preview mode. Defaults to None.
        - **Mach_ang_mode (bool, optional)**: Flag indicating whether to show Mach angle mode. Defaults to None.
        - `**kwargs`: Additional keyword arguments for customization.

    Returns:
        None
    """
    # tracking_std = kwargs.get('tracking_std', False)
    points_color = kwargs.get('points_color', 'yellow')
    points_opacity = kwargs.get('points_opacity', 1)
    points_size = kwargs.get('points_size', 12)
    uncertain_point_color = kwargs.get('uncertain_point_color', 'red')

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8))

    if avg_preview_mode is not None:
        avg_lin_color = kwargs.get('avg_lin_color', 'w')
        avg_show_txt = kwargs.get('avg_show_txt', True)
        avg_lin_opacity = kwargs.get('avg_lin_opacity', 1)

        AvgAnglePlot(ax, shp, (mid_loc,y), avg_slope, avg_ang,
                     txt = avg_show_txt, lin_color = avg_lin_color, lin_opacity = avg_lin_opacity,
                     **kwargs)

    osc_boundary = kwargs.get('osc_boundary', False)
    if osc_boundary:
        min_bound, max_bound = kwargs.get('osc_bound_line_info', np.zeros(len(column_y)))
        b_color = kwargs.get('b_color', 'tab:orange')
        b_lins_opacity = kwargs.get('b_lins_opacity', 1)
        osc_range_opacity = kwargs.get('osc_range_opacity', 0.3)
        AvgAnglePlot(ax, shp, (min_bound[2],y), min_bound[1], AngleFromSlope(min_bound[1]),
                     txt= False, lin_color = b_color, lin_opacity = b_lins_opacity)
        AvgAnglePlot(ax, shp, (max_bound[2],y), max_bound[1], AngleFromSlope(max_bound[1]),
                     txt= False, lin_color = b_color, lin_opacity = b_lins_opacity)
        ax.fill_betweenx(column_y, min_bound[0],max_bound[0], color = b_color, alpha = osc_range_opacity)

    conf_interval = kwargs.get('conf_interval', 0)

    if conf_interval > 0:
        conf_info = kwargs.get('conf_info', np.zeros([len(column_y),2]))
        conf_color = kwargs.get('conf_color', 'tab:green')
        conf_range_opacity = kwargs.get('conf_range_opacity', 0.3)
        true_outlier = kwargs.get('true_outlier', None)
        x_new_loc, conf_value = zip(*conf_info)
        x_new_loc = np.array(x_new_loc)
        conf_value = np.array(conf_value)
        for x, y in enumerate(column_y):
            ax.plot([x_loc[x],x_new_loc[x]],[y,y],':', color = 'white')

        ax.fill_betweenx(column_y, x_new_loc-conf_value, x_new_loc+conf_value,
                         color = conf_color, alpha = conf_range_opacity)
        if len(true_outlier) > 0:
            for item in true_outlier:
                ax.plot(item[0],item[1],'x', color = 'green', ms = points_size+2)

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

    ax.plot(x_loc, column_y, '-o',
            color=points_color, ms=points_size, alpha=points_opacity)

    ax.plot(uncertain, uncertain_y, 'o',
            color=uncertain_point_color, ms=points_size, alpha=points_opacity)

def PreviewCVPlots(img, Ref_x0 = [], Ref_y = [],
                   tk = [], avg_shock_loc = [], **kwargs):
    """
    PreviewCVPlots function is used to overlay various plot elements on an image for visualization purposes.

    Parameters:
        - **img (numpy.ndarray)**: Input image.
        - **Ref_x0 (list[int])**: List of x-coordinates for reference lines.
        - **Ref_y (list[int])**: List of y-coordinates for reference lines.
        - **tk (list[int])**: List of y-coordinates for tk lines.
        - **avg_shock_loc (int)**: Average shock location.

    Keyword Arguments:
        - **Ref_x0_color (tuple)**: Color of reference x0 lines. Defaults to CVColor.GREEN.
        - **tk_color (tuple)**: Color of tk lines. Defaults to CVColor.GREENBLUE.
        - **Ref_y1_color (tuple)**: Color of reference y1 lines. Defaults to CVColor.FUCHSIPINK.
        - **Ref_y2_color (tuple)**: Color of reference y2 lines. Defaults to CVColor.YELLOW.
        - **avg_shock_loc_color (tuple)**: Color of average shock location line. Defaults to CVColor.CYAN.

    Returns:
        **numpy.ndarray**: Image with overlaid plot elements.
    """

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

def residual_preview(error, margins, nSlices, count):
    fig, ax = plt.subplots(figsize=(10, 8))
    e_median, Q1, Q2, IQR = margins
    try:
        # Plot histogram
        ax.hist(error, bins=20, edgecolor='black')

        # Update bar colors based on outlier range
        for bar in ax.containers[0]:
            x = bar.get_x() + 0.5 * bar.get_width()
            color = 'red' if x < Q1 - 1.5 * IQR or x > Q2 + 1.5 * IQR else 'tab:blue'
            bar.set_color(color)
            bar.set_edgecolor('black')

        # Draw vertical lines for median and IQR bounds
        ax.vlines([e_median, Q1 - 1.5 * IQR, Q2 + 1.5 * IQR], 0, nSlices,
                  colors=['tab:red', 'tab:orange', 'tab:orange'], linestyles=['-', '--', '--'])

        # Add grid, labels, and title
        ax.grid(True, axis='y', which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw=1.5)
        ax.set_ylim([0, nSlices])
        ax.set_title(f'Outliers have high impact at {count}')
    except Exception as e:
        print(f'Error during plotting: {e}')

# -----------------------------| Draft code |----------------------------------
# ploting the middle point as possible center of rotation
# if mid_loc > 0 and y > 0: ax.plot(mid_loc, y, '*', color='g', ms=10)

# if tracking_std:
#     # pass
#     std_m, avg_loc, std_x_shift, box_shp = kwargs.get('std_line_info', np.zeros(len(column_y)))
#     midloc = std_x_shift -  avg_loc
#     AvgAnglePlot(ax, shp, (np.mean(std_x_shift),y), std_m, avg_ang, avg_lin_color = 'r', **kwargs)
#     ax.plot(std_x_shift, column_y,'x-' ,ms = 10, lw = 2, color= 'tab:orange')
#     # AvgAnglePlot(ax, shp, (avg_loc - std_x_shift,y), std_m, avg_ang, avg_lin_color = 'r', **kwargs)
