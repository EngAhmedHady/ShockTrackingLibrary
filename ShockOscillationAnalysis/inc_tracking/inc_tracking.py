# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:47:27 2024

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import glob
import keyboard    # type: ignore
import screeninfo  # To find the  monitor resolution
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .tracking_accuracy import conf_lim
from ..shocktracking import ShockTraking
from ..support_func import bg_manipulation
from ..decorators import calculate_running_time
from ..ShockOscillationAnalysis import SOA, CVColor, BCOLOR
from ..linedrawingfunctions import InclinedLine, AngleFromSlope
from ..preview import plot_review, visualize_shock_angles, rotate_axes
from ..slice_list_generator.list_generation_tools import GenerateIndicesList
from .inc_tracking_support import (anglesInterpolation, v_least_squares, ransac,
                                   shockDomain, ImportingFiles)

px = 1/plt.rcParams['figure.dpi']
plt.rcParams.update({'font.size': 10})
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"


class InclinedShockTracking(SOA):
    def __init__(self, f: int = 1, D: float = 1, pixelScale: float = 1):
        self.f = f  # ----------------------- sampling rate (fps)
        self.D = D  # ----------------------- refrence length for scaling (mm)
        self.pixelScale = pixelScale  # ----- initialize scale of the pixels
        super().__init__(f, D, pixelScale)

    def InclinedShockDomainSetup(self, CheckingWidth: int, CheckingHieght: int|list, inclined_ref_line: int|list[int,tuple,tuple], # define the calculation domain
                                 imgShape: tuple,                                              # define the whole image parameters
                                 VMidPnt: int = 0, nPnts: int = 0,                             # define the slices parameters
                                 preview_img: np.ndarray = None) -> tuple[list, int, int]:     # preview parameters
        """
        Setup shock inclination test, provids the test slices info. with aid of the estimated 
        inclined shock line.

        Parameters:
            - **CheckingWidth (int)**: Width for shock domain checking (sliceWidth).
            - **CheckingHeight (int or list)**: Height for shock domain checking in px. If a list is provided, it represents a range of heights for generating points [upper limit, lower limit].
            - **imgShape (tuple)**: Shape of the image (y-length, x-length).
            - **VMidPnt (int, optional)**: Vertical midpoint. Default is 0.
            - **nPnts (int, optional)**: Number of points to generate for inclined shock lines. Default is 0.
            - **preview_img (np.ndarray, optional)**: Image for preview as background. Default is None.

        Returns:
            tuple: A tuple containing:
                - SlicesInfo (list): List of shock domain slices, [[x-domainStrt,x-domainEnd],y-sliceLoc].
                - nPnts (int): Number of slices generated for inclined shock.
                - inclinationCheck (bool): Boolean indicating whether the shock inclination test is applicable.

        Example:
            >>> from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
            >>> instance = IncTrac(f)
            >>> width = 20
            >>> height = [10, 20]
            >>> shape = (100, 200)
            >>> points = 5
            >>> slices, nPnts, success = instance.InclinedShockDomainSetup(width, height, shape, nPnts=points)
            >>> print(slices, nPnts, success)

        .. note::
            - The function sets up shock inclination testing by visualizing the shock domain.
            - It returns a list of slices location and range, the number of slices, and the inclination applicability.

        """
        print('Shock inclination test and setup ...', end=" ")
        slices_info = []; inclinationCheck = True

        # Generate the points
        if hasattr(CheckingHieght, "__len__"):
            # If CheckingHeight is a list, generate nPnts points within the height range
            Pnts = np.linspace(0, abs(CheckingHieght[1]- CheckingHieght[0]), nPnts)
            DatumY = CheckingHieght[0]
            CheckingHieght = abs(CheckingHieght[1]- CheckingHieght[0])
        else:
            # If CheckingHeight is a single value, create points based on slice thickness
            Ht = int(CheckingHieght / 2)
            DatumY = VMidPnt - Ht

        if nPnts < 1:
            if CheckingHieght > 10:
                Pnts = np.linspace(0, CheckingHieght, 10)
                nPnts = 10
            elif CheckingHieght > 2 and CheckingHieght <= 10:
                Pnts = np.linspace(0,CheckingHieght, round(CheckingHieght))
                nPnts = CheckingHieght
            else:
                print(u'\u2717')
                warning = ' Shock Angle Test Vertical Range is not sufficient!;'
                action = 'Escaping the shock angle checking...'
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
                return slices_info, 0, False

        # IncInfoIndx = len(self.Reference) - 1
        HalfSliceWidth = round(CheckingWidth/2)

        # Define the estimated shock line using 2 points P1, P2 --> User defined
        P1 = (round(inclined_ref_line[0][0]), round(inclined_ref_line[0][1]))
        LineSlope = inclined_ref_line[2]

        # Calculate the y-intercepts of the upper and lower inclined lines
        aUp = shockDomain('up', P1, HalfSliceWidth, LineSlope, imgShape, preview_img)
        aDown = shockDomain('down', P1, HalfSliceWidth, LineSlope, imgShape, preview_img)

        # Calculate y-coordinates of the points
        y_i = np.array(Pnts + DatumY).astype(int)
        if LineSlope != 0 and LineSlope != np.inf:
            # Calculate x-coordinates based on the slope
            x_i1 = np.array((y_i - aUp) / LineSlope).astype(int)
            x_i2 =  np.array((y_i - aDown) / LineSlope).astype(int)
        elif LineSlope == np.inf:
            # Handle the case of vertical lines
            x_i1 = np.full(nPnts, P1[0] - HalfSliceWidth)
            x_i2 = np.full(nPnts, P1[0] + HalfSliceWidth)
        elif LineSlope == 0:
            # if the line is horizontal
            print(u'\u2717')
            error = 'Software is not supporting horizontal shock waves, aborting...!'
            print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
            sys.exit()

        # Optionally, preview the shock domain on the image
        if preview_img is not None:
            for pnt in range(len(Pnts)):
                cv2.circle(preview_img, (x_i1[pnt],y_i[pnt]), 
                           radius=3, color=CVColor.RED, thickness=-1)
                cv2.circle(preview_img, (x_i2[pnt],y_i[pnt]), 
                           radius=3, color=CVColor.RED, thickness=-1)
        slices_info = x_i1,x_i2,y_i
        print(u'\u2713')
        return slices_info, nPnts, inclinationCheck

    @calculate_running_time
    def InclinedShockTracking(self, img_set: list[np.ndarray],                        # image set for line tracking
                              nSlices: int, Ref: list[int], slice_thickness: int = 1, # slices and tracking info.
                              nReview: int|list[int] = 0, output_dirc: str = '', # Review parameters
                              comment: str = '', **kwargs) -> tuple:                  # Other parameters

        """
        Track and analyze the shock angle in a sequence of images.

        Parameters:
            - **img_set (list)**: List of images for shock tracking, the images should be formated as numpy array.
            - **nSlices (int)**: Number of slices to divide the image into for analysis.
            - **Ref (list)**: Reference points for slices [[x_1, x_2, y], ...].
            - **slice_thickness (int, optional)**: Thickness of each slice. Default is 1.
            - **nReview (int or list, optional)**: Number or range of images to review. Default is 0.
            - **output_dirc (str, optional)**: Directory to save the review images. Default is ''.
            - `**kwargs`: Additional keyword arguments:
                - **avg_preview_mode (str)**: Mode for previewing average angle.
                - **review_inc_slice_tracking (list or int)**: Slices to review for tracking.
                - **osc_boundary (bool)**: To display the oscilliation domain depending on the analyised image set.

        Returns:
            tuple: Average global angle (float) and average midpoint location (float).

        Example:
            >>> from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
            >>> instance = IncTrac(f)
            >>> img_set = [img1, img2, img3]
            >>> ref = [[10, 20], [30, 40], [50, 60]]
            >>> avg_angle, avg_mid_loc = instance.InclinedShockTracking(img_set, 2, ref, nReview=5)
            >>> print(avg_angle, avg_mid_loc)

        .. note ::
            - The function performs shock tracking across a series of images and calculates the average shock angle.
            - If `nReview` is specified as number, the first n-images will be reviewd.
            - If `nReview` is specified as list, A list specifying the start and end.
            - If `nReview` is specified, it plots and optionally saves review images for inspection.
            - It uses least squares method to fit the shock locations and calculates the corresponding angle.
        """
        # Initialize variables for tracking
        avg_ang_glob = 0         # Global average angle
        midLocs =[]              # List to store mid-locations
        xLocs = []               # List to store all x-location lists of shocks
        avg_slope = 0            # visual average slope [float or list[float]]
        avg_midloc = 0           # Average mid-location
        columnY = []             # List to store y-coordinates of the slices
        uncertain_list = []      # List to store uncertain x-locations
        uncertainY_list = []     # List to store y-coordinates of uncertain x-locations
        m = []                   # list of slops from least square calculation
        # Optional keyword arguments
        avg_preview_mode = kwargs.get('avg_preview_mode', None)
        review_inc_slice_tracking = kwargs.get('review_inc_slice_tracking', -1)
        img_set_size = len(img_set)
        # Array to review the tracked slices within the iamge set
        slice_ploting_array = np.zeros(img_set_size)
        if hasattr(review_inc_slice_tracking, "__len__") and len(review_inc_slice_tracking) == 2:
            review_inc_slice_tracking.sort(); start, end = review_inc_slice_tracking
            try:
                for i in range(start, end): slice_ploting_array[i] = 1
            except Exception:
                warning = 'Slices to review is out of the image set!;'
                action = 'Only within the range are considered.'
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
                pass

        elif not hasattr(review_inc_slice_tracking, "__len__") and review_inc_slice_tracking > -1:
            slice_ploting_array[review_inc_slice_tracking] = 1

        # Determine half thickness for slice processing
        if slice_thickness > 1: Ht = int(slice_thickness/2)  # Ht -> Half Thickness
        else: Ht = 1; slice_thickness = 2

        # Initialize upper and lower bounds for slices
        upper_bounds = np.zeros(nSlices, dtype = int)
        lower_bounds = np.zeros(nSlices, dtype = int)

        for i in range(nSlices):
            upper_bounds[i] = Ref[2][i] - Ht
            lower_bounds[i] = Ref[2][i] + Ht if slice_thickness%2 == 0 else Ref[2][i] + Ht + 1
            columnY.append(Ref[2][i])
        columnY = np.array(columnY)

        # Determine middle index for slices
        midIndx = nSlices // 2
        midIndx2 = midIndx if nSlices % 2 != 0 else midIndx - 1
        y = (columnY[midIndx2] + columnY[midIndx]) / 2
        LastShockLoc = -1

        xLoc = -1*np.ones(nSlices)
        shock_deg = []
        if nSlices < 5:
            warning = 'Number of points is not sufficient for RANSAC!;'
            action = 'Normal least square will be performed.'
            print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
            print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
            
        
        print('Shock tracking started ...', end=" ")
        if isinstance(img_set, dict):
            # Shape of images in the image set from the first image
            shp = next(iter(img_set.values())).shape
            img_indx = list(img_set.keys())
            img_set = list(img_set.values())

        elif isinstance(img_set, list):
            # If img_set is a list
            shp = img_set[0].shape
            img_indx = np.arange(img_set_size)

        for count, img in enumerate(img_set):
            xLocOld = xLoc.copy()
            xLoc = []; uncertain = []; uncertainY = []
            for i in range(nSlices):
                x_i1, x_i2 = Ref[0][i], Ref[1][i]
                Slice = np.sum(img[upper_bounds[i]-1:lower_bounds[i], x_i1:x_i2], axis=0)/slice_thickness

                LastShockLoc = xLocOld[i]-Ref[0][i]
                ShockLoc, certainLoc, _ = ShockTraking(Slice, LastShockLoc=LastShockLoc,
                                                       count=img_indx[count],
                                                       Plot=slice_ploting_array[count])
                xLoc.append(ShockLoc + Ref[0][i])
                if not certainLoc:
                    uncertain.append(xLoc[-1])
                    uncertainY.append(Ref[2][i])

            # Calculate the slope using least squares method
            # est_shock_slope = v_least_squares(xLoc, columnY, nSlices)
            est_shock_slope, midLoc = ransac(np.array(xLoc), columnY, 1)

            m.append(est_shock_slope)
            shock_deg.append(AngleFromSlope(est_shock_slope))

            # finding the middle point
            # midLocs.append(np.mean(xLoc))
            midLocs.append(midLoc)

            xLocs.append(xLoc)
            uncertain_list.append(uncertain)
            uncertainY_list.append(uncertainY)

        osc_boundary = kwargs.get('osc_boundary', False)
        if osc_boundary:
            max_b = np.zeros(nSlices); min_b = shp[1]*np.ones(nSlices)
            for xloc_list in xLocs:
                for n_count, xloc in enumerate(xloc_list):
                    if xloc > max_b[n_count]: max_b[n_count] = xloc
                    if xloc < min_b[n_count]: min_b[n_count] = xloc
            m_min = v_least_squares(min_b, columnY, nSlices)
            m_max = v_least_squares(max_b, columnY, nSlices)
            mean_min = np.mean(min_b); mean_max = np.mean(max_b)
            kwargs['osc_bound_line_info'] = ([min_b, m_min, mean_min], [max_b, m_max, mean_max])
        print(u'\u2713')

        avg_midloc= np.mean(midLocs)
        avg_ang_glob = np.mean(shock_deg)
        avg_angle_data = np.zeros(5)
        avg_midloc_data = np.zeros(3)
        avg_angle_data[0] = avg_ang_glob
        avg_midloc_data[0] = avg_midloc

        # Initialize list of residual analysis
        e = np.zeros(img_set_size, dtype=np.ndarray)
        pop_ylist = np.empty(img_set_size, dtype=np.ndarray)

        # error integeration for each snapshot
        conf_interval = kwargs.get('conf_interval', 0)

        df = img_set_size-1
        if conf_interval > 0:
            print('Calculating confidance limits...', end='')
            e, pop_ylist, m, shock_deg, midLocs, w_avg_ang, conf_ang=conf_lim(xLocs, midLocs,
                                                                              columnY, y, m,
                                                                              img_indx, shock_deg,
                                                                              e, pop_ylist,
                                                                              uncertainY_list,
                                                                              output_dirc,
                                                                              comment,
                                                                              **kwargs)
            avg_angle_data[2:4] = [w_avg_ang, conf_ang]
            t = stats.t.ppf(conf_interval, df)
            avg_midloc= np.mean(midLocs)
            avg_ang_glob = np.mean(shock_deg)

        std_mid_xloc = np.sqrt(np.sum((midLocs-avg_midloc)**2)/df)
        mid_xloc_conf = t*np.sqrt(std_mid_xloc**2/img_set_size) if conf_interval > 0 else 0
        avg_midloc_data [1:3] = [mid_xloc_conf, std_mid_xloc]

        std_mid_Avg = np.sqrt(np.sum((shock_deg-avg_ang_glob)**2)/df)
        avg_angle_data[4] = std_mid_Avg

        avg_ang_conf = t*np.sqrt(std_mid_Avg**2/img_set_size) if conf_interval > 0 else 0
        avg_angle_data[1] = avg_ang_conf


        print(f'Angle range variation: [{min(shock_deg):0.2f}, {max(shock_deg):0.2f}],', end =' ')
        print(f'\u03C3 = {np.std(shock_deg, ddof=1):0.2f}')
        print(f'Average shock loc.: {avg_midloc:0.2f}\u00B1{mid_xloc_conf:0.2f} px')
        print(f'Average shock angle: {avg_ang_glob:0.2f}\u00B1{avg_ang_conf:0.2f} deg')

        # Visualize
        print('Plotting tracked data ...')
        visualize_shock_angles(shock_deg,avg_ang_glob, std_mid_Avg, output_dirc)

        if hasattr(nReview, "__len__"):
            r_range = [0,0,1] # review range
            for j, element in enumerate(nReview): r_range[j] = element
            r_range = tuple(sorted(r_range[:2])) + (r_range[2],)
            st,en,sp = r_range; n_review = round((en-st)/sp)
        else:
            r_range = (0,nReview,1)
            st,en,sp = r_range; n_review = nReview

        if avg_preview_mode != 'avg_ang':
            avg_slope = np.tan(np.deg2rad(180-avg_ang_glob))*np.ones(n_review)
            avg_midLoc = avg_midloc*np.ones(n_review)
            avg_ang = avg_ang_glob*np.ones(n_review)
        else:
            avg_slope = m
            avg_midLoc = midLocs
            avg_ang = shock_deg

        if en > img_set_size:
            en = img_set_size
            print(f'{BCOLOR.WARNING}Warning: {BCOLOR.ENDC}{BCOLOR.ITALIC}Images to'
                   ' review is out of the image set, only within the range are'
                   ' considered{BCOLOR.ENDC}')

        if n_review > 20:
            info = 'For memory reasons, only 20 images will be displayed.'
            print(f'{BCOLOR.BGOKCYAN}info.:{BCOLOR.ENDC}{BCOLOR.ITALIC} {info}')
            print(f'note: this will not be applied on images storing{BCOLOR.ENDC}')
        
        op_90rotate = kwargs.get('op_90rotate', False)
        rotation_center=(shp[0]/2, shp[0]/2)
        if op_90rotate:
            fig_size = (int(shp[0]*2.25*px), int(shp[1]*2.25*px))
            rotate_angle=90
            img_width, img_height = shp[:2]
            x_lim = [img_width, 0]
            y_lim = [0, img_height]
        else:
            fig_size = (int(shp[1]*2.25*px), int(shp[0]*2.25*px))
            rotate_angle=0
            img_height, img_width = shp[:2]
            x_lim = [0, img_width]
            y_lim = [img_height, 0]
            
        # Output background images (different than which the analysis performed)
        op_bg_path = kwargs.get('op_bg_path', None)
        if op_bg_path is not None:
            bg_files = sorted(glob.glob(op_bg_path))
            bg_y_crop = kwargs.get('bg_y_crop', (0, shp[1]))
            bg_x_crop = kwargs.get('bg_x_crop', (0, shp[0]))
            bg_resize = kwargs.get('bg_resize', (shp[1],shp[0]))
            bg_90rotate = kwargs.get('bg_90rotate', 0)
            if len(bg_files) >= img_set_size:
                img_set = ImportingFiles(bg_files, img_indx, img_set_size, shp,
                                     crop_y_img=bg_y_crop, crop_x_img=bg_x_crop, 
                                     resize_img=bg_resize, rotate90_img=bg_90rotate)
                img_set = list(img_set.values())
            else:
                error = f'Files found are {len(bg_files)}; Files less than expected!'
                action = 'Original file set well be used'
                print(f'{BCOLOR.FAIL}Error:{BCOLOR.ENDC}', end= ' ')
                print(f'{BCOLOR.ITALIC}{error} {action}{BCOLOR.ENDC}')

        if n_review > 0:
            n = 0            
            for i in range(st,en,sp):
                fig, ax = plt.subplots(figsize=fig_size)
                # ax.set_ylim([shp[0],0])
                # ax.set_xlim([0,shp[1]])
                kwargs['conf_info'] = e[i]
                kwargs['true_outlier'] = pop_ylist[i]
                plot_review(ax, img_set[i], shp, xLocs[i], columnY,
                            uncertain_list[i], uncertainY_list[i],
                            avg_slope[i], avg_ang[i], avg_midLoc[i] , y, **kwargs)
                rotate_axes(ax, rotate_angle, center=rotation_center)
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)

                if len(output_dirc) > 0:
                    # SAR : Shock Angle Review
                    file_name = f'SAR_{comment}_Ang{avg_ang_glob:.2f}_{img_indx[i]:05d}'
                    fig.savefig(fr'{output_dirc}\{file_name}.png',
                                bbox_inches='tight', pad_inches=0.1)

                if n > 20:
                    if len(output_dirc) == 0:
                        plt.close(fig); n = n_review
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-20s] %d%%" % ('='*int((n)/(n_review/20)),
                                                           int(5*(n)/(n_review/20))))
                        break
                    else: plt.close(fig)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int((n+1)/(n_review/20)),
                                                   int(5*(n+1)/(n_review/20))))
                n += 1
            print()
        return avg_angle_data, avg_midloc_data


    def ShockPointsTracking(self, path: str,
                            tracking_V_range:list[int | float] = [0,0], 
                            inclination_info: int|list[int,tuple,tuple] = 0, 
                            nPnts: int = 0, scale_pixels = True,
                            preview = True, output_directory = '',comment='', **kwargs):
        """
        This function identifies shock points by slicing a predefined domain of the shock and 
        tracking it at each slice based on the integral method of shock tracking outlined in this 
        `article <https://dx.doi.org/10.2139/ssrn.4797840>`_.
        It operates over a specified vertical range within the images, serving as the core function
        for inclination shock tracking. Additionally, all keyword arguments for output
        customization can be passed through this function.

        Parameters:
            - **path (str)**: Path to the directory containing the image files.
            - **tracking_V_range (list[int | float], optional)**: Vertical range for tracking shock points, specified as a list with two elements representing the upper and lower bounds (default is [0, 0]).
            - **inclination_info (int | list[int, tuple, tuple], optional)**: Information about the inclination of the shock domain. It can be an integer representing the width of the domain or a list containing the width along with the start and end points of the line defining the inclination (default is 0).
            - **nPnts (int, optional)**: Number of points to be tracked (default is 0).
            - **scale_pixels (bool, optional)**: Whether to scale the pixels in the images (default is True).
            - **preview (bool, optional)**: Whether to preview the images (default is True).
            - **output_directory (str, optional)**: Directory to save the output images (default is '').
            - **comment (str, optional)**: Additional comment for the output (default is '').
            - `**kwargs`: Additional keyword

        Additional keyword arguments `**kwargs` may include:
            Importing image options:
                - **n_files (int, optional)**: To import the first n-files from the given path.
                - **every_n_files (int, optional)**: To import files with a step (default is 1).
                - **within_range (list[int], optional)**: To import files within range [start, end]
                - **resize_img (tuple[int], optional)**: Tuple specifying the dimensions to resize the images to (width, height). Default is the original image shape.
                - **BG_path (str, optional)**: Path to the background image to be subtracted. Default is ''.

            Inflow data options:
                - **flow_dir (list, optional)**: List of tuples containing the measured y-coordinates and the corresponding angles [(y_loc, angle),...].
                - **flow_Vxy (list, optional)**: List of tuples containing the measured y-coordinates and the corresponding velocity components [(y_loc, Vx, Vy),...].
                - **angle_interp_kind (str)**: 'linear','CubicSpline' or 'PCHIP' (default is linear)

            Define the domain options and tracking:
                - **Ref_x0 (list[int], optional):** list of x-coordinates for 2-vertical reference lines [Ref_x01, Ref_x02] used for scaling, used instead of drawing.
                - **Ref_y0 (int, optional)**: y-coordinate of the horizontal reference line [y = 0] used as reference for tracking_V_range, used instead of drawing.
                - **slice_thickness (int, optional)**: Thickness of each slice. Default is 1.

            Review and results options:
                - **preview (bool)**: Whether to preview the selected domain for the analysis or not (default is True)
                - **review_inc_slice_tracking (int|list[int])**: To plot all slices of spacific image or range of images (default is 0)
                - **preview_angle_interpolation (bool)**: If True, plot the angle interpolation for preview. (default is False).
                - **avg_preview_mode (str)**: 'avg_all', 'avg_ang' and None (default is None).
                - **Mach_ang_mode (str)**: Flag indicating whether to display the Mach number 'Mach_num' if inflow data is available or 'flow_dir' when the Mach number is available(defaults to None).
                - **osc_boundary (bool)**: To display the oscilliation domain depending on the analyised image set.
                - **output_directory (str)**: The pathe where the output results will be stored (default is '').
                - **store_n_files (int|list[int])**: Specify the first n output results to be stored, or provide a range of output image indices to be stored in the format [start, end].

            Results display options:
                - **points_opacity (float)**: The transperancy of the tracking points from 0 to 1 (default is 1).
                - **points_color (str)**: The color of the tracking points (default is 'yellow')
                - **uncertain_point_color (str)**: The color of the uncertain tracked points (default is 'red')
                - **avg_lin_color (str)**: The average line color when the `avg_preview_mode` is not None (default is 'white')
                - **avg_lin_opacity (float)**: The transperancy of the average line from 0 to 1 (default is 1)
                - **avg_show_txt (bool)**: To display the angle value or not when the `avg_preview_mode` is not None (default is True)
                - **avg_txt_Yloc (int)**: y-location of the angle value text in pixels (default is image height minus 100.)
                - **avg_txt_size (float)**: Font size of the Angle value (default is 26pt)
                - **M1_color (str)**: The calculated values of Mach  when the `avg_preview_mode` is not None (default is 'orange')
                - **M1_txt_size (float)**: Font size of the Mach number and inflow Angle values (default is 26pt)
                - **arc_dia (float)**: inflow angle arc diameter (default is 80px)
                - **arw_len (float)**: inflow arrow length (default is 50px)
                - **b_color (str)**: boundary domain and lines color for the active ``osc_boundary`` (default is 'tab:orange')
                - **osc_range_opacity (float)**: The transperancy of the boundary domain from 0 to 1 (default is 0.3)
                - **b_lins_opacity (float)**: The transperancy of the boundary lines from 0 to 1 (default is 1)

        .. note ::
            - In case of ``scale_pixels = True`` the ``Ref_x0`` and ``Ref_y0`` must be defined either by drawing or as arguments.
            - The values of ``Ref_x0`` and ``Ref_y0`` are in pixels.
            - ``tracking_V_range`` values are in pixels, but if ``scale_pixels = True``, the values should match the scale units [for example mm].
            - The imported files are defined by thier index and sorted by name.
            - For automation it is better to set ``preview`` to False.
            - The plots from ``review_inc_slice_tracking`` give details of finding the shock location, such as local minima, shock location, last image shock location, etc.
            - ``avg_preview_mode`` the display of vertical least squares regression line of the tracked points:
                - 'avg_all': Displays the average of all lines calculated from the tracked points across the entire image dataset.
                - 'avg_ang': Displays the line of the tracked points in each image
            - In this version ``Mach_ang_mode`` can only calculate Mach number when inflow data is available, 'flow_dir' is not yet supported!
            - ``osc_boundary`` Calculated based on the minimum and maximum recorded location at each slice, a vertical least squares regression line is used to define the oscillation boundary and minimize any uncertainty that might occur.

        Returns:
            tuple[float, float]: Average inclination angle and average shock location.

        Example:
            >>> from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
            >>> D = 60
            >>> imgPath = r'C:\\Users\admin\Pictures\*.png'
            >>> IncTrac = IncTrac(D = D)
            >>> IncTrac.ShockPointsTracking(imgPath, scale_pixels = True,
                                            tracking_V_range = [5, 25],
                                            nPnts = 9, inclination_info = [100, (249, 0), (0, 429)], slice_thickness = 4,
                                            points_opacity = 0.0,
                                            avg_preview_mode = 'avg_all', avg_show_txt = True, avg_txt_Yloc = 400, avg_txt_size = 30,
                                            preview = True,
                                            osc_boundary = True)
        Steps:
            1. Define reference vertical boundaries (for scaling). Draw or assine them in this parameter ``Ref_x0``,
            2. Define reference horizontal line as the y-datum. Draw or assine it in this parameter ``Ref_y0``,
            3. Define the estimated line of shock. Draw or assine it as two points in this parameter `inclination_info` as in the example
            4. Run shock tracking function within the selected vertical range ``tracking_V_range``.
            5. The function will perform the tracking after dividing the vertical range into ``nPnts``.
        """
        important_info = kwargs.get('important_info', 0)
        dis_unit = self.univ_unit["dis"]
        if important_info:
            print(f'{BCOLOR.UNDERLINE}Notes:{BCOLOR.ENDC}{BCOLOR.ITALIC}')
            draw_lin = '`one to define the line and one to confirm`'
            print(f'\t- Draw a line requires 2 left clicks {draw_lin}')
            print('\t- To delete a line press right click instead of second left click')
            print('\t- Do not forget to press any key except `Esc` to close the image window')
            print(f'\t- To terminating process press `Esc`{BCOLOR.ENDC}')

        resize_img = kwargs.get('resize_img', None)
        crop_y_img = kwargs.get('crop_y_img', None)
        crop_x_img = kwargs.get('crop_x_img', None)
        bg, n1 = bg_manipulation(path, resize_img,crop_y_img, crop_x_img)
        # In case no file found end the progress and eleminate the program
        if n1 < 1: sys.exit()
            
        op_bg_path = kwargs.get('op_bg_path', None)
        if op_bg_path is not None:
            bg_y_crop = kwargs.get('bg_y_crop', None)
            bg_x_crop = kwargs.get('bg_x_crop', None)
            bg_resize = kwargs.get('bg_resize', None)
            bg_rotate = kwargs.get('bg_90rotate', 0)
            op_bg, n_bg = bg_manipulation(op_bg_path, bg_y_crop, bg_x_crop, 
                                          bg_resize, bg_rotate, n=n1)
            if n_bg < 1:
                action = 'Original file set well be used'
                print(f'{BCOLOR.ITALIC}{action}{BCOLOR.ENDC}')
            
        # Open first file and set the limits and scale
        Refimg = op_bg if op_bg is not None else bg
        # Refimg = cv2.imread(files[0])
        # Refimg = cv2.cvtColor(Refimg, cv2.COLOR_BGR2GRAY)
        # Refimg = cv2.cvtColor(Refimg, cv2.COLOR_GRAY2BGR)
        shp = Refimg.shape; print('Img Shape is:', shp)
        Ref_x0 = kwargs.get('Ref_x0', [0,0])
        Ref_y0 = kwargs.get('Ref_y0', -1)
        resize_img = kwargs.get('resize_img', (shp[1],shp[0]))
        Refimg = cv2.resize(Refimg, resize_img)

        if scale_pixels:
            Ref_x0, Ref_y0, Ref_y1=self.DefineReferences(Refimg, shp, Ref_x0, scale_pixels, Ref_y0)
        else: self.clone = Refimg.copy()

        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        print(f'Screen resolution: {screen_width}, {screen_height}')

        tracking_V_range.sort()
        start, end = tracking_V_range
        y_diff = abs(end-start)
        draw_y = y_diff == 0

        if draw_y:
            tracking_V_range = []
            # Vertical limits and scale
            try:
                Ref_y1 = self.LineDraw(self.clone, 'H', 2, line_color=CVColor.ORANGE)[-1]
            except Exception:
                Ref_y1 = Ref_y0
                warning = 'Nothing was drawn!;'
                action = 'Ref_y1 value is {Ref_y1}'
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
            tracking_V_range.append((Ref_y0 - Ref_y1) * self.pixelScale)
            try:
                Ref_y2 = self.LineDraw(self.clone, 'H', 2, line_color=CVColor.ORANGE)[-1]
            except Exception:
                Ref_y2 = Ref_y1
                warning = 'Nothing was drawn!;'
                action = 'Ref_y2 value is {Ref_y2}'
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')

            tracking_V_range.append((Ref_y0 - Ref_y2) * self.pixelScale)
            if Ref_y1 == Ref_y2:
                error='Vertical range of tracking is not sufficient!'
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
                sys.exit()

            tracking_V_range.sort()

            if Ref_y1 > Ref_y2:
                Ref_y11 = Ref_y2
                Ref_y2 = Ref_y1
                Ref_y1 = Ref_y11

        else:
            tracking_V_range.sort() if Ref_y0 > -1 else tracking_V_range.sort(reverse=True)
            if Ref_y0 > -1:
                Ref_y2, Ref_y1 = [round(Ref_y0 - (x / self.pixelScale)) for x in tracking_V_range]
            else:
                tracking_V_range

            if Ref_y1 < 0 or Ref_y2 > shp[0]:
                v_range_lim = [-(shp[0]-Ref_y0)*self.pixelScale, Ref_y0*self.pixelScale]
                v_range_lim = [ round(elem, 2) for elem in v_range_lim ]
                v_range_lim = f'{v_range_lim}{dis_unit}'
                error=f'Vertical range of tracking is not sufficient!; v_range_lim = {v_range_lim}'
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
                sys.exit()

            cv2.line(self.clone, (0, Ref_y1), (shp[1], Ref_y1),
                     CVColor.ORANGE, 1)
            cv2.line(self.clone, (0, Ref_y2), (shp[1], Ref_y2),
                     CVColor.ORANGE, 1)

        print('Vertical range of tracking points is:')
        v1, v2 = tracking_V_range
        print(f'\t- In ({dis_unit})s from {v1:0.2f}{dis_unit} to {v2:0.2f}{dis_unit}')
        print(f'\t- In pixels from {Ref_y1}px to {Ref_y2}px')

        # estemat shock domain
        if not hasattr(inclination_info, "__len__"):
            CheckingWidth = inclination_info
            if CheckingWidth < 10:
                warning = 'Reference width should not be less than 10px!'
                action = ''
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
                request = 'Please provide reference width >10px: '
                print(f'{BCOLOR.BGOKGREEN}Request:{BCOLOR.ENDC}', end=' ')
                CheckingWidth = int(input(f'{BCOLOR.ITALIC}{request}{BCOLOR.ENDC}'))
            inclined_ref_line = []
            try:
                inclined_ref_line = self.LineDraw(self.clone, 'Inc', 3)[-1]
            except Exception:
                warning = 'Nothing was drawn!;'
                action = 'inclined_ref_line value is {inclined_ref_line}'
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')

            if not hasattr(inclined_ref_line, "__len__") or len(inclined_ref_line) < 4:
                error = 'Reference lines are not sufficient!'
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
                sys.exit()

        elif len(inclination_info) > 2:
            P1,P2,m,a = InclinedLine(inclination_info[1],inclination_info[2],imgShape = shp)
            cv2.line(self.clone, P1, P2, (0,255,0), 1)
            inclined_ref_line = [P1,P2,m,a]
            CheckingWidth = inclination_info[0]

        if nPnts == 0:
            while nPnts == 0:
                request = 'Please provide number of points to be tracked: '
                print(f'{BCOLOR.BGOKGREEN}Request:{BCOLOR.ENDC}', end=' ')
                nPnts = int(input(f'{BCOLOR.ITALIC}{request}{BCOLOR.ENDC}'))
                if nPnts > abs(Ref_y1-Ref_y2):
                    error = 'Insufficient number of points!'
                    print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
                    nPnts = 0

        Ref, nSlices, inclinationCheck = self.InclinedShockDomainSetup(CheckingWidth,
                                                                       [Ref_y1, Ref_y2],
                                                                       inclined_ref_line,
                                                                       shp,
                                                                       nPnts=nPnts,
                                                                       preview_img=self.clone)

        pnts_y_list = []
        for i in range(nSlices):
            pnts_y_list.append((Ref_y0-Ref[2][i])*self.pixelScale)
        flow_dir = kwargs.get('flow_dir', [])
        flow_Vxy = kwargs.get('flow_Vxy', [])
        Mach_ang_mode = kwargs.get('Mach_ang_mode', None)
        if (len(flow_dir) > 0 or len(flow_Vxy) > 0) and Mach_ang_mode is not None:
            kwargs['inflow_dir_deg'] = anglesInterpolation(pnts_y_list, **kwargs)
            kwargs['inflow_dir_rad'] = np.array(kwargs['inflow_dir_deg'])*np.pi/180

        if preview:
            cv2.imshow('Investigation domain before rotating', self.clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if keyboard.read_key() == "esc":
                print('Operation not permitted. Terminating process...')
                sys.exit()
            cv2.waitKey(1)

            cv2.imwrite(fr'{output_directory}\AnalysisDomain-Points.jpg', self.clone)

        import_n_files = kwargs.get('n_files', 0)
        if import_n_files == 0:
            import_n_files = kwargs.get('within_range', [0, 0])
            if abs(import_n_files[0]-import_n_files[1]) > n1:
                import_n_files[import_n_files.index(max(import_n_files))] = n1
        import_step = kwargs.get('every_n_files', 1)
        indices_list, n_images = GenerateIndicesList(n1, import_n_files, import_step)

        if inclinationCheck:
            files = sorted(glob.glob(path))
            img_list = ImportingFiles(files, indices_list, n_images, shp, **kwargs)

        store_n_files = kwargs.get('store_n_files', n_images)

        avg_shock_angle, avg_shock_loc = self.InclinedShockTracking(img_list,
                                                                    nSlices,
                                                                    Ref,
                                                                    nReview=store_n_files,
                                                                    output_dirc=output_directory,
                                                                    **kwargs)

        return avg_shock_angle, avg_shock_loc


# ================================| Draft code |===============================
"""
Screen resize:
        # if shp[0] >= screen_height*0.85:
        #     r = shp[0]/shp[1] # ---------- Image aspect ratio
        #     NewImgSize = (round(screen_height*0.85/r),
                            round(screen_height*0.85))

        #     Refimg = cv2.resize(Refimg, NewImgSize)
        #     reductionRatio = NewImgSize[0]/shp[0]
        #     shp = NewImgSize
        #     print('Warning: Image hieght is larger than your monitor hieght')
        #     print(f'Only reference image will be adjusted to {shp}')

standard deviation of the shock angle:
    tracking_std = kwargs.get('tracking_std', False)
    # if tracking_std:
        # avg_xloc = np.array(xLocs).mean(axis=0)
        # xLoc_std = np.sqrt(np.square(xLocs).mean(axis=0))
        # std_m = self.v_least_squares(xLoc_std, columnY, nSlices)
        # x_min = shp[1]; x_max = 0;
        # for j in range(nSlices):
        #     x_i1, x_i2 = Ref[0][j], Ref[1][j]
        #     if x_min > min([x_i1, x_i2]): x_min = min([x_i1, x_i2])
        #     if x_max < max([x_i1, x_i2]): x_max = max([x_i1, x_i2])
        # print(np.mean(xLoc_std),np.mean(avg_xloc))
        # kwargs['std_line_info'] = (std_m, np.mean(avg_xloc),
                             xLoc_std, (columnY[-1]-columnY[0], x_max - x_min))

Oscillation Domain from two points:
        # kwargs['osc_bound_line_info'] = ([[min_b[0],min_b[-1]],
                     m_min, mean_min], [[max_b[0],max_b[-1]], m_max, mean_max])
"""
