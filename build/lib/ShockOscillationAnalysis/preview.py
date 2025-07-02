# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:05:30 2024

@author: Ahmed H. Hanfy
"""

import cv2
import numpy as np
from .constants import CVColor
import matplotlib.pyplot as plt
from .support_func import log_message
from matplotlib.patches import Arc, FancyArrowPatch
from .linedrawingfunctions import InclinedLine, AngleFromSlope


def angle_txt(ax, X, avg_txt_Yloc, angle, 
              lin_color, lin_opacity, avg_txt_size, 
              op_rotate=False):
        # Draw an arc to represent the angle
        if op_rotate:
            angle -= 90
            theta_1 = 0
            theta_2 = -angle
            txt_xloc = avg_txt_Yloc - 115
            txt_yloc = X + 10
            xline = [X, X]
            yline = [avg_txt_Yloc-10, avg_txt_Yloc+100]
        else:
            theta_1 = -angle
            theta_2 = 0
            txt_xloc = X + 40
            txt_yloc = avg_txt_Yloc - 10
            xline = [X-10, X+100]
            yline = [avg_txt_Yloc, avg_txt_Yloc]
        avg_ang_arc = Arc((X, avg_txt_Yloc),80, 80, 
                          theta1=theta_1 , theta2=theta_2, color=lin_color)
        ax.add_patch(avg_ang_arc)

        # Add the text annotation for the angle
        ax.text(txt_xloc, txt_yloc , f'${{{angle:0.2f}}}^\circ$',
                color=lin_color, fontsize=avg_txt_size)
        # Plot a horizontal line at the text annotation location to compare the inclination angle
        ax.plot(xline, yline, lw=1, color=lin_color, alpha=lin_opacity)
        
def AvgAnglePlot(ax:plt.axes, img_shp:tuple , P: tuple, slope: float, angle: float,
                 txt: bool=True, lin_color='w', lin_opacity=1, **kwargs) -> None:
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
    op_90rotate = kwargs.get('op_90rotate', False)
    avg_txt_Yloc = kwargs.get('avg_txt_Yloc', img_shp[0]-100)
    avg_txt_size = kwargs.get('avg_txt_size', 26)

    # Calculate the inclined line end points
    P1, P2, _, a = InclinedLine(P, slope=slope, imgShape=img_shp)

    # Plot the inclined line
    ax.plot([P1[0],P2[0]], [P1[1],P2[1]], lw=2,
            color=lin_color, linestyle=(0, (20, 3, 5, 3)), alpha=lin_opacity)
    
        
    # Plot the text annotation if enabled
    if txt:
        # ax.text(450, avg_txt_Yloc+100 , f'{P[0]:0.2f}', color=lin_color, fontsize=avg_txt_size)
        # Calculate the X position for the text annotation
        if slope != 0 and slope != np.inf: X = int((avg_txt_Yloc-a)/slope)
        elif slope == np.inf: X = P[0]
        else: X = avg_txt_Yloc
        angle_txt(ax, X, avg_txt_Yloc, angle, 
                  lin_color, lin_opacity, avg_txt_size, op_90rotate)



def plot_review(ax:plt.axes, img: np.ndarray, shp:tuple[int], 
                x_loc:list[float], column_y:list[float], 
                uncertain:list[float], uncertain_y:list[float], 
                avg_slope:float, avg_ang:float, mid_loc:int=-1, y:int=-1, y_avg:float=-1,
                avg_preview_mode=None, Mach_ang_mode=None, **kwargs):

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
        - **y_avg (list(float), optional)**: middle vertical locations from RANSAC. Defaults to -1.
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

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8), extent=(0, shp[1], shp[0], 0))

    if avg_preview_mode is not None:
        avg_lin_color = kwargs.get('avg_lin_color', 'w')
        avg_show_txt = kwargs.get('avg_show_txt', True)
        avg_lin_opacity = kwargs.get('avg_lin_opacity', 1)

        AvgAnglePlot(ax, shp, (mid_loc,y_avg), avg_slope, avg_ang,
                     txt=avg_show_txt, lin_color=avg_lin_color, lin_opacity=avg_lin_opacity,
                     **kwargs)

    osc_boundary = kwargs.get('osc_boundary', False)
    if osc_boundary:
        min_bound, max_bound = kwargs.get('osc_bound_line_info', np.zeros(len(column_y)))
        b_color = kwargs.get('b_color', 'tab:orange')
        b_lins_opacity = kwargs.get('b_lins_opacity', 1)
        osc_range_opacity = kwargs.get('osc_range_opacity', 0.3)
        AvgAnglePlot(ax, shp, (min_bound[2],y), min_bound[1], AngleFromSlope(min_bound[1]),
                     txt= False, lin_color=b_color, lin_opacity=b_lins_opacity)
        AvgAnglePlot(ax, shp, (max_bound[2], y), max_bound[1], AngleFromSlope(max_bound[1]),
                     txt=False, lin_color=b_color, lin_opacity=b_lins_opacity)
        ax.fill_betweenx(column_y, min_bound[0], max_bound[0], 
                         color=b_color, alpha=osc_range_opacity)

    conf_interval = kwargs.get('conf_interval', 0)

    if conf_interval > 0:
        conf_info = kwargs.get('conf_info', np.zeros([len(column_y),3]))
        conf_color = kwargs.get('conf_color', 'tab:green')
        pred_color = kwargs.get('pred_color', 'tab:red')
        conf_range_opacity = kwargs.get('conf_range_opacity', 0.3)
        pred_range_opacity = kwargs.get('pred_range_opacity', 0.3)
        true_outlier = kwargs.get('true_outlier', None)
        x_new_loc, conf_value, pred_value = zip(*conf_info)
        x_new_loc = np.array(x_new_loc)
        conf_value = np.array(conf_value)
        for y, x in enumerate(x_new_loc):
            ax.plot([x_loc[y],x],[column_y[y],column_y[y]],':', color = 'white')

        ax.fill_betweenx(column_y[:len(x_new_loc)], x_new_loc-pred_value, x_new_loc+pred_value,
                         color=pred_color, alpha=pred_range_opacity)

        ax.fill_betweenx(column_y[:len(x_new_loc)], x_new_loc-conf_value, x_new_loc+conf_value,
                         color=conf_color, alpha=conf_range_opacity)


        if true_outlier is not None and len(true_outlier) > 0:
            for item in true_outlier:
                ax.plot(item[0], item[1],'x', color='green', ms=points_size+2)

    if Mach_ang_mode == 'Mach_num':
        inflow_dir_deg = kwargs.get('inflow_dir_deg', np.zeros(len(column_y)))
        inflow_dir_rad = kwargs.get('inflow_dir_rad', np.zeros(len(column_y)))
        M1_color = kwargs.get('M1_color', 'orange')
        M1_txt_size = kwargs.get('M1_txt_size', 26)
        arw_len = kwargs.get('arw_len', 50)
        arc_dia = kwargs.get('arc_dia', 80)
        for i in range(1,len(column_y)-1):
            p1 = (x_loc[i], column_y[i])
            p2 = (x_loc[i-1], column_y[i-1])
            _,_,m,_ = InclinedLine(p1, p2, imgShape=shp)
            xlen = np.cos(inflow_dir_rad[i]) 
            ylen = np.sin(inflow_dir_rad[i])
            local_ang = AngleFromSlope(m)
            inflow_ang = local_ang + inflow_dir_deg[i]
            ax.text(p1[0]+40 ,p1[1]- 5, f'${{{inflow_ang:0.2f}}}^\circ$', 
                    size=M1_txt_size, color=M1_color)
            ax.plot([p1[0]-arw_len*xlen,p1[0]+60*xlen], 
                    [p1[1]-arw_len*ylen,p1[1]+60*ylen], color=M1_color, lw=1)

            arc1 = Arc(p1,arc_dia, arc_dia, theta1=-local_ang, theta2=0+inflow_dir_deg[i],
                       color=M1_color)
            ax.add_patch(arc1)
            M1 = 1/np.sin((inflow_ang)*np.pi/180)
            arr = FancyArrowPatch((p1[0] - arw_len*xlen, p1[1] - arw_len*ylen), p1,
                                  arrowstyle='-|>, head_length=20, head_width=3, widthA=2', 
                                  color=M1_color)
            ax.add_patch(arr)
            ax.annotate(f'M$_1 ={{{M1:0.2f}}}$', xy=p1,
                        color=M1_color, size = M1_txt_size,
                        xytext=(p1[0] - arw_len*xlen, p1[1] + arw_len*ylen),
                        horizontalalignment='right', verticalalignment='center')

    ax.plot(x_loc, column_y, '-o',
            color=points_color, ms=points_size, alpha=points_opacity)

    ax.plot(uncertain, uncertain_y, 'o',
            color=uncertain_point_color, ms=points_size, alpha=points_opacity)

def PreviewCVPlots(img:np.ndarray, Ref_x0:list[int]=None, Ref_y:list[int]|int=None,
                   tk:list[int]=None, avg_shock_loc:list[float]=None, **kwargs):
    """
    PreviewCVPlots function is used to overlay various plot elements on an image for
    visualization purposes.

    Parameters:
        - **img (np.ndarray)**: Input image.
        - **Ref_x0 (list[int])**: List of x-coordinates for reference lines.
        - **Ref_y (list[int]|int)**: List of y-coordinates for reference lines.
        - **tk (list[int])**: List of y-coordinates for tk lines.
        - **avg_shock_loc (list[float])**: Average shock location.

    Keyword Arguments:
        - **Ref_x0_color (tuple)**: Color of reference x0 lines. Defaults to CVColor.GREEN.
        - **tk_color (tuple)**: Color of tk lines. Defaults to CVColor.GREENBLUE.
        - **Ref_y1_color (tuple)**: Color of reference y1 lines. Defaults to CVColor.FUCHSIPINK.
        - **Ref_y2_color (tuple)**: Color of reference y2 lines. Defaults to CVColor.YELLOW.
        - **avg_shock_loc_color (tuple)**: Color of average shock location line. Defaults to CVColor.CYAN.

    Returns:
        **np.ndarray**: Image with overlaid plot elements.
    """

    shp = img.shape
    if Ref_x0 is not None:
        Ref_x0_color = kwargs.get('Ref_x0_color', CVColor.GREEN)
        cv2.line(img, (Ref_x0[0],0), (Ref_x0[0],shp[0]), Ref_x0_color, 1)
        cv2.line(img, (Ref_x0[1],0), (Ref_x0[1],shp[0]), Ref_x0_color, 1)

    if tk is not None and len(tk) == 2:
        tk_color = kwargs.get('tk_color', CVColor.GREENBLUE)
        cv2.line(img, (0, tk[0]), (shp[1], tk[0]), tk_color, 1)
        cv2.line(img, (0, tk[1]), (shp[1], tk[1]), tk_color, 1)

    Ref_y1_color = kwargs.get('Ref_y1_color', CVColor.FUCHSIPINK)
    if hasattr(Ref_y, "__len__"):
        if len(Ref_y) > 2: 
            cv2.line(img, (0,Ref_y[1]), (shp[1],Ref_y[1]), Ref_y1_color, 1)
        if Ref_y[0] > -1:
            Ref_y0_color = kwargs.get('Ref_y2_color', CVColor.YELLOW)
            cv2.line(img, (0,Ref_y[0]), (shp[1],Ref_y[0]), Ref_y0_color, 1)
    elif Ref_y > 0: cv2.line(img, (0,Ref_y), (shp[1],Ref_y), Ref_y1_color, 1)

    avg_shock_loc_color = kwargs.get('avg_shock_loc_color', CVColor.CYAN)
    if hasattr(avg_shock_loc, "__len__") and len(avg_shock_loc) > 2:
        cv2.line(img, avg_shock_loc[0], avg_shock_loc[1], avg_shock_loc_color, 1)
    else: cv2.line(img, (int(avg_shock_loc), 0), 
                        (int(avg_shock_loc), shp[0]), avg_shock_loc_color, 1)

    return img

def residual_preview(error, margins, nSlices, count, log_dirc=''):
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
        error = f'Error during residual plotting: {e}'
        log_message(error, log_dirc)
        print(error)

def visualize_shock_angles(shock_deg: list[float], avg_ang_glob: float, std_mid_Avg: float,
                           output_directory: str = '') -> None:
    """
    Plot a histogram of shock angles and overlay statistical indicators.

    Parameters:
        - **shock_deg (list[float])**: List of shock angles in degrees.
        - **avg_ang_glob (float)**: Global average of the shock angles.
        - **std_mid_Avg (float)**: Standard deviation of the shock angles.
        - **output_directory (str, optional)**: Directory to save the plot.
                                                If empty, the plot is not saved.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot histogram of shock angles
    ax.hist(shock_deg, bins=20, edgecolor='black', alpha=0.8, color='skyblue')

    # Plot average and standard deviation lines
    y_limit = ax.get_ylim()[1]
    ax.vlines([avg_ang_glob, avg_ang_glob - std_mid_Avg, avg_ang_glob + std_mid_Avg],
              0, y_limit, colors=['tab:red', 'tab:orange', 'tab:orange'],
              linestyles=['-', '--', '--'],
              label=['Average Angle', '1-σ Deviation'])

    # Add labels, and grid
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Frequency")
    # ax.set_xlim([74,81])
    ax.grid(True, axis='y', which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw=1.5)
    if output_directory:
        fig.savefig(f"{output_directory}/Hist_Ang_{avg_ang_glob:.2f}_std_{std_mid_Avg:.2f}.png",
                    bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def rotate_axes(ax:plt.axes, angle:float, center:tuple[int]=(0, 0)) -> plt.axes:
    """
    Rotate the elements of a Matplotlib axis around a specified center point.
    This function applies an affine transformation to rotate all plot elements 
    (lines and images) on the given Matplotlib axis by a specified angle around 
    a defined center.

    Parameters:
        - **ax (matplotlib.axes.Axes)**: The Matplotlib axis object containing the elements to be rotated.
        - **angle (float)**: The angle of rotation in degrees (counterclockwise).
        - **center (tuple[int, int])**: A tuple representing the (x, y) coordinates of the rotation center. 
          Defaults to (0, 0).

    Returns:
        - matplotlib.axes.Axes: The axis with rotated elements.

    .. note ::
        - This transformation affects the visual rendering of the axis elements 
          but does not modify the underlying data.
        - Ensure that the input axis contains elements like lines or images for
          the transformation to have an effect.
        - The function works with both line plots and images added to the Matplotlib axis.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([0, 1], [0, 1], label="Line")
        >>> rotate_axes(ax, angle=45, center=(0.5, 0.5))
        >>> plt.show()
"""
    from matplotlib.transforms import Affine2D
    # Create the Affine2D transformation for the specified angle
    trans = Affine2D()
    # Translate the system so that the rotation center becomes the origin
    trans.translate(-center[0], -center[1])  # Move the plot to origin (center to origin)
    # Rotate by the specified angle
    trans.rotate_deg(angle)
    # Translate back to the original position
    trans.translate(center[0], center[1])  # Move the plot back to the original position
    # Apply the transformation to all plot elements (lines and images)
    for line in ax.lines:
        line.set_transform(trans + ax.transData)
    
    for im in ax.images:
        im.set_transform(trans + ax.transData)
    return ax
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
