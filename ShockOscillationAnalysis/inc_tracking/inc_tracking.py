# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:47:27 2024

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import glob
import screeninfo  # .......................... To find the  monitor resolution
import numpy as np
import matplotlib.pyplot as plt
from ..preview import plot_review
from ..shocktracking import ShockTraking
from ..ShockOscillationAnalysis import SOA, CVColor, BCOLOR
from ..linedrawingfunctions import InclinedLine, AngleFromSlope
from ..slice_list_generator.list_generation_tools import GenerateIndicesList
from .inc_tracking_support import (anglesInterpolation, v_least_squares,
                                   pearson_corr_coef, shockDomain,
                                   ImportingFiles)

px = 1/plt.rcParams['figure.dpi']
plt.rcParams.update({'font.size': 25})
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
        Setup shock inclination test, provids the test slices info. with aid of the estimated inclined shock line.

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
        else:
            # If CheckingHeight is a single value, create points based on slice thickness
            Ht = int(CheckingHieght/2)
            DatumY = VMidPnt-Ht
            if CheckingHieght > 10:             Pnts = np.linspace(0, CheckingHieght, 10); nPnts = 10
            elif CheckingHieght > 2 and CheckingHieght <= 10: Pnts = range(CheckingHieght); nPnts = CheckingHieght
            else:
                print(u'\u2717')
                print(f'{BCOLOR.BGOKCYAN}info.:{BCOLOR.ENDC}{BCOLOR.ITALIC}Escaping the shock angle checking... \nSlice thickness is not sufficient for check the shock angle{BCOLOR.ENDC}')
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
            print(f'{BCOLOR.FAIL}Error:{BCOLOR.ENDC}{BCOLOR.ITALIC} Software is not supporting horizontal shock waves, aborting...{BCOLOR.ENDC}')
            sys.exit()

        # Optionally, preview the shock domain on the image
        if preview_img is not None:
            for pnt in range(len(Pnts)):
                cv2.circle(preview_img, (x_i1[pnt],y_i[pnt]), radius=3, color=CVColor.RED, thickness=-1)
                cv2.circle(preview_img, (x_i2[pnt],y_i[pnt]), radius=3, color=CVColor.RED, thickness=-1)
        slices_info = x_i1,x_i2,y_i
        print(u'\u2713')
        return slices_info, nPnts, inclinationCheck

    def InclinedShockTracking(self, imgSet: list[np.ndarray],                         # image set for line tracking
                              nSlices: int, Ref: list[int], slice_thickness: int = 1, # slices and tracking info.
                              nReview: int = 0, output_directory: str = '',           # Review parameters
                              **kwargs) -> tuple:                                     # Other parameters

        """
        Track and analyze the shock angle in a sequence of images.

        Parameters:
            - **imgSet (list)**: List of images for shock tracking, the images should be formated as numpy array.
            - **nSlices (int)**: Number of slices to divide the image into for analysis.
            - **Ref (list)**: Reference points for slices [[x_1, x_2, y], ...].
            - **slice_thickness (int, optional)**: Thickness of each slice. Default is 1.
            - **nReview (int, optional)**: Number of images to review. Default is 0.
            - **output_directory (str, optional)**: Directory to save the review images. Default is ''.
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
            - If `nReview` is specified, it plots and optionally saves review images for inspection.
            - It uses least squares method to fit the shock locations and calculates the corresponding angle.
        """
        # Initialize variables for tracking
        avg_ang_glob= 0          # Global average angle
        count = 0                # Image counter
        midLocs =[]              # List to store mid-locations
        xLocs = []               # List to store all x-location lists of shocks
        avg_slope = 0            # visual average slope [float or list[float]]
        AvgMidLoc = 0            # Average mid-location
        columnY = []             # List to store y-coordinates of the slices
        uncertain_list = []      # List to store uncertain x-locations
        uncertainY_list = []     # List to store y-coordinates of uncertain x-locations
        shp = imgSet[0].shape    # Shape of images in the image set from the first image
        m = []                   # list of slops from least square calculation
        # Optional keyword arguments
        avg_preview_mode = kwargs.get('avg_preview_mode', None)
        review_inc_slice_tracking = kwargs.get('review_inc_slice_tracking', -1)

        # Array to review the tracked slices within the iamge set
        slice_ploting_array = np.zeros(len(imgSet))
        if hasattr(review_inc_slice_tracking, "__len__") and len(review_inc_slice_tracking) == 2:
            review_inc_slice_tracking.sort(); start, end = review_inc_slice_tracking
            try:
                for i in range(start, end): slice_ploting_array[i] = 1
            except Exception:
                print(f'{BCOLOR.WARNING}Warning: {BCOLOR.ENDC}{BCOLOR.ITALIC}Slices to review is out of the image set, only within the range are considered{BCOLOR.ENDC}')
                pass

        elif not hasattr(review_inc_slice_tracking, "__len__") and review_inc_slice_tracking > -1:
            slice_ploting_array[review_inc_slice_tracking] = 1

        # Determine half thickness for slice processing
        if slice_thickness > 1: Ht = int(slice_thickness/2)  # Ht -> Half Thickness
        else: Ht = 1; slice_thickness = 2;

        # Initialize upper and lower bounds for slices
        upper_bounds = np.zeros(nSlices, dtype = int); lower_bounds = np.zeros(nSlices, dtype = int)

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
        AngReg = []

        print('Shock tracking started ...', end=" ")
        for count, img in enumerate(imgSet):
            xLocOld = xLoc.copy()
            xLoc = []; uncertain = []; uncertainY = []
            for i in range(nSlices):
                x_i1, x_i2 = Ref[0][i], Ref[1][i]
                Slice = np.sum(img[upper_bounds[i]-1:lower_bounds[i], x_i1:x_i2], axis=0) / slice_thickness

                LastShockLoc = xLocOld[i]-Ref[0][i]
                ShockLoc, certainLoc, _  = ShockTraking(Slice, LastShockLoc = LastShockLoc, count = count, Plot = slice_ploting_array[count])
                # ShockLoc, certainLoc, _  = ShockTraking(Slice, LastShockLoc = LastShockLoc, count = count)
                xLoc.append(ShockLoc + Ref[0][i])
                if not certainLoc: uncertain.append(xLoc[-1]); uncertainY.append(Ref[2][i])

            # finding the middle point
            midLocs.append(np.mean([xLoc[midIndx], xLoc[midIndx2]]))

            # Calculate the slope using least squares method
            m.append(v_least_squares(xLoc, columnY, nSlices))
            AngReg.append(AngleFromSlope(m[-1]))

            # Pearson correlation coefficient
            r = pearson_corr_coef(xLoc, columnY, nSlices)
            if 0.7 < abs(r) < 0.98:
                print(f'good Pearson coefficient: {BCOLOR.OKGREEN}{r:0.2f}{BCOLOR.ENDC} @ {count}')
            elif abs(r) < 0.7:
                print(f'low Pearson coefficient: {BCOLOR.FAIL}{r:0.2f}{BCOLOR.ENDC}, with slope {m[-1]} @ {count}')

            xLocs.append(xLoc);
            uncertain_list.append(uncertain); uncertainY_list.append(uncertainY)

        AvgMidLoc= np.mean(midLocs);  avg_ang_glob = np.mean(AngReg);
        if avg_preview_mode != 'avg_ang':
            avg_slope = np.mean(m)*np.ones(nReview)
            avg_midLoc = AvgMidLoc*np.ones(nReview)
            avg_ang = avg_ang_glob*np.ones(nReview)
        else:
            avg_slope = m; avg_midLoc = midLocs; avg_ang = AngReg

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
        print('Plotting tracked data ...')
        if hasattr(nReview, "__len__"):
            r_range = [0,0,1]
            for j, element in enumerate(nReview): r_range[j] = element
            r_range = tuple(sorted(r_range[:2])) + (r_range[2],)
            st,en,sp = r_range; n_review = round((en-st)/sp)
        else:
            r_range = (0,nReview,1)
            st,en,sp = r_range; n_review = nReview

        if en > len(imgSet):
            en = len(imgSet)
            print(f'{BCOLOR.WARNING}Warning: {BCOLOR.ENDC}{BCOLOR.ITALIC}Images to review is out of the image set, only within the range are considered{BCOLOR.ENDC}')

        if n_review > 20:
             print(f'{BCOLOR.BGOKCYAN}info.:{BCOLOR.ENDC}{BCOLOR.ITALIC} For memory reasons, only 20 images will be displayed.')
             print(f'note: this will not be applied on images storing{BCOLOR.ENDC}')

        if n_review > 0:
            n = 0
            for i in range(st,en,sp):
                fig, ax = plt.subplots(figsize=(int(shp[1]*1.75*px), int(shp[0]*1.75*px)))
                ax.set_ylim([shp[0],0]); ax.set_xlim([0,shp[1]])
                plot_review(ax, imgSet[i], shp, xLocs[i], columnY,
                            uncertain_list[i], uncertainY_list[i],
                            avg_slope[i], avg_ang[i], avg_midLoc[i] , y, **kwargs)
                if len(output_directory) > 0:
                    fig.savefig(fr'{output_directory}\ShockAngleReview_Ang{avg_ang_glob:.2f}_{i:05d}.png', bbox_inches='tight', pad_inches=0.1)

                if n > 20:
                    if len(output_directory) == 0:
                        plt.close(fig); n = n_review
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-20s] %d%%" % ('='*int((n)/(n_review/20)), int(5*(n)/(n_review/20))))
                        break;
                    else: plt.close(fig)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int((n+1)/(n_review/20)), int(5*(n+1)/(n_review/20))))
                n += 1
            print()

        print(f'Angle range variation: [{min(AngReg):0.2f},{max(AngReg):0.2f}], \u03C3 = {np.std(AngReg):0.2f}')
        return avg_ang_glob, AvgMidLoc


    def ShockPointsTracking(self, path: str,
                            tracking_V_range:list[int,float] = [0,0], inclination_info: int|list[int,tuple,tuple] = 0, nPnts: int = 0, scale_pixels = True,
                            preview = True, output_directory = '',comment='', **kwargs):
        """
        This function identifies shock points by slicing a predefined domain of the shock and tracking it at each slice based on the integral method of shock tracking outlined in this `article <https://dx.doi.org/10.2139/ssrn.4797840>`_.
        It operates over a specified vertical range within the images, serving as the core function for inclination shock tracking. Additionally, all keyword arguments for output customization can be passed through this function.

        Parameters:
            - **path (str)**: Path to the directory containing the image files.
            - **tracking_V_range (list[int, float], optional)**: Vertical range for tracking shock points, specified as a list with two elements representing the upper and lower bounds (default is [0, 0]).
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

        files = sorted(glob.glob(path))
        n1 = len(files)
        # In case no file found end the progress and eleminate the program
        if n1 < 1: print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}No files found!{BCOLOR.ENDC}'); sys.exit();
        # Open first file and set the limits and scale
        Refimg = cv2.imread(files[0])
        Refimg = cv2.cvtColor(Refimg, cv2.COLOR_BGR2GRAY)
        Refimg = cv2.cvtColor(Refimg, cv2.COLOR_GRAY2BGR)
        shp = Refimg.shape; print('Img Shape is:', shp)
        Ref_x0 = kwargs.get('Ref_x0', [0,0])
        Ref_y0 = kwargs.get('Ref_y0', -1)
        resize_img = kwargs.get('resize_img', (shp[1],shp[0]))
        Refimg = cv2.resize(Refimg, resize_img)

        if scale_pixels: Ref_x0, Ref_y0, Ref_y1 = self.DefineReferences(Refimg, shp, Ref_x0, scale_pixels, Ref_y0)
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
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} Nothing was drawn!{BCOLOR.ENDC} Ref_y1 value is {Ref_y1}')
            tracking_V_range.append((Ref_y0 - Ref_y1) * self.pixelScale)
            try:
                Ref_y2 = self.LineDraw(self.clone, 'H', 2, line_color=CVColor.ORANGE)[-1]
            except Exception:
                Ref_y2 = Ref_y1;
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC}Nothing was drawn!{BCOLOR.ENDC} Ref_y1 value is {Ref_y2}')

            tracking_V_range.append((Ref_y0 - Ref_y2) * self.pixelScale)
            if Ref_y1 == Ref_y2:
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}Vertical range of tracking is not sufficient!{BCOLOR.ENDC}')
                sys.exit()

            tracking_V_range.sort()

            if Ref_y1 > Ref_y2:
                Ref_y11 = Ref_y2
                Ref_y2 = Ref_y1
                Ref_y1 = Ref_y11

        else:
            tracking_V_range.sort() if Ref_y0 > -1 else tracking_V_range.sort(reverse=True)
            Ref_y2, Ref_y1 = [round(Ref_y0 - (x / self.pixelScale)) for x in tracking_V_range] if Ref_y0 > -1 else tracking_V_range

            if Ref_y1 < 0 or Ref_y2 > shp[0]:
                print('Vertical range of tracking is not sufficient!')
                sys.exit()

            cv2.line(self.clone, (0, Ref_y1), (shp[1], Ref_y1),
                     CVColor.ORANGE, 1)
            cv2.line(self.clone, (0, Ref_y2), (shp[1], Ref_y2),
                     CVColor.ORANGE, 1)

        print(f'Vertical range of tracking points starts from {tracking_V_range[0]:0.2f}mm to {tracking_V_range[1]:0.2f}mm')
        print(f'in pixels from {Ref_y1}px to {Ref_y2}px')

        # estemat shock domain
        if not hasattr(inclination_info, "__len__"):
            CheckingWidth = inclination_info
            if CheckingWidth < 10:
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}Reference width is not sufficient!{BCOLOR.ENDC}');
                CheckingWidth = int(input(f'{BCOLOR.BGOKGREEN}Request: {BCOLOR.ENDC}{BCOLOR.ITALIC}Please provide reference width >10px: {BCOLOR.ENDC}'))
            inclined_ref_line = []
            try:
                inclined_ref_line = self.LineDraw(self.clone, 'Inc', 3)[-1]
            except Exception:
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} Nothing was drawn!{BCOLOR.ENDC} inclined_ref_line value is {inclined_ref_line}')

            if not hasattr(inclined_ref_line, "__len__") or len(inclined_ref_line) < 4:
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}Reference lines are not sufficient!{BCOLOR.ENDC}'); sys.exit()

        elif len(inclination_info) > 2:
            P1,P2,m,a = InclinedLine(inclination_info[1],inclination_info[2],imgShape = shp)
            cv2.line(self.clone, P1, P2, (0,255,0), 1)
            inclined_ref_line = [P1,P2,m,a]
            CheckingWidth = inclination_info[0]

        if nPnts == 0:
            while nPnts == 0:
                nPnts = int(input(f'{BCOLOR.BGOKGREEN}Request: {BCOLOR.ENDC}{BCOLOR.ITALIC}Please provide number of points to be tracked: {BCOLOR.ENDC}'))
                if nPnts > abs(Ref_y1-Ref_y2):
                    print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}insufficient number of points{BCOLOR.ENDC}')
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
            cv2.imshow('investigation domain before rotating', self.clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            cv2.imwrite(fr'{output_directory}\AnalysisDomain-Points.jpg',
                        self.clone)

        import_n_files = kwargs.get('n_files', 0)
        if import_n_files == 0:
            import_n_files = kwargs.get('within_range', [0, 0])
        import_step = kwargs.get('every_n_files', 1)
        indices_list, n_images = GenerateIndicesList(n1, import_n_files,
                                                     import_step)

        if inclinationCheck:
            original_img_list, img_list = ImportingFiles(files, indices_list,
                                                         n_images, shp,
                                                         **kwargs)

        store_n_files = kwargs.get('store_n_files', n_images)
        avg_shock_angle, avg_shock_loc = self.InclinedShockTracking(img_list,
                                                                    nSlices,
                                                                    Ref,
                                                                    nReview=store_n_files,
                                                                    output_directory=output_directory,
                                                                    **kwargs)

        print('Average inclination angle {:.2f} deg'.format(avg_shock_angle))

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
