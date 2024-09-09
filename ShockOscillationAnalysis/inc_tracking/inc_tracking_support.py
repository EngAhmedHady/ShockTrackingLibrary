# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:15:04 2024

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from ..linedrawingfunctions import InclinedLine
from ..ShockOscillationAnalysis import BCOLOR, CVColor
# from linedrawingfunctions import InclinedLine
# from ShockOscillationAnalysis import BCOLOR, CVColor
from scipy.interpolate import CubicSpline, PchipInterpolator

def v_least_squares(xLoc: list[float], columnY:list[float], nSlices: int) -> list[float]:
    """
    Perform a vertical least squares linear regression to find the slope.

    Parameters:
        - **xLoc (list[float])**: List of x-coordinates of the points.
        - **columnY (list[float])**: List of y-coordinates of the points.
        - **nSlices (int)**: Number of slices or data points.

    Returns:
        list[float]: List containing the slope of the best-fit line.

    Example:
        >>> from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
        >>> instance = IncTrac(f)
        >>> xLoc = [1, 2, 3, 4, 5]
        >>> columnY = [2, 4, 6, 8, 10]
        >>> nSlices = 5
        >>> slope = instance.v_least_squares(xLoc, columnY, nSlices)
        >>> print(slope)

    .. note::
        - The function calculates the slope of the best-fit line using the vertical least squares method.
        - It returns the slope as a single-element list.
    """
    xy = np.array(xLoc)*columnY; yy = columnY**2
    x_sum = np.sum(xLoc)       ; y_sum = np.sum(columnY)
    xy_sum = np.sum(xy)        ; yy_sum = np.sum(yy)

    b = 1/((nSlices*xy_sum - x_sum * y_sum)/(nSlices*yy_sum - y_sum**2))

    return b

def pearson_corr_coef(xLoc: list[float], columnY:list[float], nSlices: int) -> list[float]:

    """
    Calculate the Pearson correlation coefficient.

    Parameters:
        - **xLoc (list[float])**: List of x-coordinates of the points.
        - **columnY (list[float])**: List of y-coordinates of the points.
        - **nSlices (int)**: Number of slices or data points.

    Returns:
        float: Pearson correlation coefficient.

    Example:
        >>> from ShockOscillationAnalysis import InclinedShockTracking
        >>> instance = InclinedShockTracking()
        >>> nSlices = 5
        >>> xLoc = [1, 2, 3, 4, 5]
        >>> columnY = [2, 4, 6, 8, 10]
        >>> nSlices = 5
        >>> r = instance.pearson_corr_coef(xLoc, columnY, nSlices)
        >>> print(r)

    .. note::
        - The function calculates the Pearson correlation coefficient using the formula:

          .. math::
              r = \\frac{n \\sum (xy) - (\\sum x)(\\sum y)}{\\sqrt{[n \\sum x^2 - (\\sum x)^2][n \\sum y^2 - (\\sum y)^2]}}

        - It returns the Pearson correlation coefficient as a float.
    """
    xy = np.array(xLoc)*columnY; yy = columnY**2; xx = np.array(xLoc)**2
    x_sum = np.sum(xLoc)       ; y_sum = np.sum(columnY)
    xy_sum = np.sum(xy)        ; yy_sum = np.sum(yy)
    xx_sum = np.sum(xx)

    r_num = nSlices*(xy_sum) - x_sum*y_sum
    r_den = np.sqrt((nSlices*xx_sum - x_sum**2)*(nSlices*yy_sum - y_sum**2))
    r = r_num/r_den

    return r


def error_sum(xloc: list[float], columnY:list[float], nSlices: int,
              l_slope: float, l_yint: float):
    """
    Calculate the normalized sum of absolute errors between the actual x-coordinates and
    the x-coordinates predicted by the linear regression line.

    Parameters:
        - **xloc (list[float])**: List of actual x-coordinates.
        - **columnY (list[float])**: List of y-coordinates.
        - **nSlices (int)**: Number of data points.
        - **l_slope (float)**: Slope of the linear regression line.
        - **l_yint (float)**: y-intercept of the linear regression line.

    Returns:
        float: Normalized sum of absolute errors.

    Example:
        >>> xloc = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> columnY = [2.0, 4.1, 5.9, 8.2, 10.0]
        >>> nSlices = 5
        >>> l_slope = 2.0
        >>> l_yint = 0.0
        >>> e = error_sum(xloc, columnY, nSlices, l_slope, l_yint)
        >>> print(e)

    .. note::
        - The function calculates the normalized sum of absolute errors by comparing the actual x-coordinates
          to the x-coordinates predicted by the linear regression line for each y-coordinate.
        - It returns the average error per slice as a float.
    """
    error = []
    e = 0
    e2 = 0
    for i in range(nSlices):
        x_dash = (columnY[i] - l_yint)/l_slope
        error.append(abs(xloc[i]-x_dash))
        e += error[-1]
    e /= nSlices
    # for i in range(nSlices):
    #     e2 += (error[i]-e)**2
    # e2 /= nSlices
    # print(f'{BCOLOR.OKGREEN}{e:0.2f}\t{BCOLOR.OKCYAN}{e2:0.2f}{BCOLOR.ENDC}')
    # sigma = np.sqrt(e2)
    return e

def anglesInterpolation(pnts_y_list: list[int],                              # Generated points by class
                        flow_dir: list[float] = None, flow_Vxy:list[tuple] = None, # measured data (LDA, CFD, ... )
                        **kwargs) -> list[float]:                                  # other parameters
    """
    Interpolate angles based on given y-coordinates and corresponding angles or velocity components.

    Parameters:
       - **pnts_y_list (list)**: List of y-coordinates to interpolate angles for.
       - **flow_dir (list, optional)**: List of tuples containing the measured y-coordinates and the corresponding angles [(y_loc, angle),...].
       - **flow_Vxy (list, optional)**: List of tuples containing the measured y-coordinates and the corresponding velocity components [(y_loc, Vx, Vy),...].
       - `**kwargs`: Additional keyword arguments:
            - angle_interp_kind (str):
            - preview_angle_interpolation (bool): If True, plot the angle interpolation for preview. Default is False.

    Returns:
        list: Interpolated angles for each y-coordinate in `pnts_y_list`. If the y-domain is out of valid range, returns an empty list.

    Example:
        >>> from ShockOscillationAnalysis import InclinedShockTracking
        >>> instance = InclinedShockTracking()
        >>> pnts_y = [5, 15, 25]
        >>> flow_dir = [(0, 10), (10, 20), (20, 30)]
        >>> interpolated_angles = instance.anglesInterpolation(pnts_y, flow_dir)
        >>> print(interpolated_angles)

    .. note ::
        - interpolation can be performed using multible methods 'linear','CubicSpline' and 'PCHIP' for better inflow representation
            - If 'linear', linear interpolation will be performed. Default is 'linear'.
            - If 'CubicSpline', Interpolate data with a piecewise cubic polynomial which is twice continuously differentiable.
            - If 'PCHIP', PCHIP 1-D monotonic cubic interpolation will be performed.
        - If a y-coordinate in `pnts_y_list` is out of the range defined by `flow_dir` or `flow_Vxy`, the function will consider only boundary angles.
        - If both `flow_dir` and `flow_Vxy` are provided, `flow_dir` will take precedence.

    .. seealso ::
        - For more information about CubicSpline: `scipy.interpolate.CubicSpline`_.
        - For more information about PCHIP: `scipy.interpolate.PchipInterpolator`_.

    .. _scipy.interpolate.CubicSpline: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline
    .. _scipy.interpolate.PchipInterpolator: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html#scipy.interpolate.PchipInterpolator

    """

    if flow_dir is not None:
        # Unzip the angles_list into separate locs and angles lists
        locs, angles = zip(*flow_dir)
    elif flow_Vxy is not None:
        # Unzip the Vxy into separate locs, Vx, Vy lists
        locs, Vx, Vy = zip(*flow_Vxy)
        angles = np.arctan(np.array(Vy)/np.array(Vx))*180/np.pi

    if min(locs) > min(pnts_y_list) or max(locs) < max(pnts_y_list):
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} Provided y-domain is out of valid range{BCOLOR.ENDC}')

    angle_interp_kind = kwargs.get('angle_interp_kind', 'linear')
    if angle_interp_kind == 'linear':
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} Only boundary angles will considered ...{BCOLOR.ENDC}')
        intr_flow_dir = np.interp(pnts_y_list, locs, angles)
    elif angle_interp_kind == 'CubicSpline':
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} First derivative at curves ends will considered zero, overshooting is likely occurs ...{BCOLOR.ENDC}')
        interp_fun = CubicSpline(locs, angles, bc_type = 'clamped')
        intr_flow_dir = interp_fun(pnts_y_list)
    elif angle_interp_kind == 'PCHIP':
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} First derivative at curves ends will considered zero, overshooting is likely occurs ...{BCOLOR.ENDC}')
        interp_fun = PchipInterpolator(locs, angles, extrapolate = 'bool')
        intr_flow_dir = interp_fun(pnts_y_list)

    preview_angle_interpolation = kwargs.get('preview_angle_interpolation', False)
    if preview_angle_interpolation:
        fig, ax = plt.subplots(figsize=(10,20))
        ax.plot(angles, locs, '-o', ms = 5)
        ax.plot(intr_flow_dir, pnts_y_list, 'x', ms = 10)
    return intr_flow_dir

def shockDomain(Loc: str, P1: tuple[int], HalfSliceWidth: int, LineSlope: float,
                imgShape: tuple[int], preview_img: np.ndarray = None) -> float:
    """
    Generate and visualize a shock domain based on the slice width and
    the drawn line parameters (one point and slope).

    Parameters:
        - **Loc (str)**: The shock direction, either 'up' or 'down'.
        - **P1 (tuple)**: The starting point of the shock domain.
        - **HalfSliceWidth (int)**: Half the width of the slice.
        - **LineSlope (float)**: Slope of the inclined line.
        - **imgShape (tuple)**: Image size (y-length, x-length).
        - **preview_img (optional)**: Image for previewing the shock domain. Default is None.

    Returns:
        float: The y-intercept of the inclined line.

    Example:
        >>> from ShockOscillationAnalysis import InclinedShockTracking
        >>> instance = InclinedShockTracking()
        >>> slope_intercept = instance.shockDomain('up', (10, 20), 5, 0.5, (100, 200))
        >>> print(slope_intercept)

    .. note::
        - The function generates and visualizes a shock domain line based on the specified parameters.
        - It returns the y-intercept of the inclined line.

    """
    if Loc =='up': P1new = (P1[0] - HalfSliceWidth, P1[1])
    else: P1new = (P1[0] + HalfSliceWidth, P1[1])
    anew = P1new[1] - LineSlope*P1new[0] # y-intercept
    P1new,P2new,m,a = InclinedLine(P1new, slope = LineSlope, imgShape=imgShape)
    if preview_img is not None: cv2.line(preview_img, P1new, P2new, CVColor.RED, 1)
    return anew

def ImportingFiles(pathlist: list[str], indices_list: list[int], n_images: int, # Importing info.
                   imgs_shp: tuple[int],                                              # Images info.
                   **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:            # Other parameters
    """
    Import images from the specified paths, optionally resize them, and remove the background if provided.

    Parameters:
        - **pathlist (list[str])**: List of paths to the images.
        - **indices_list (list[int])**: List of indices specifying which images to import from the pathlist.
        - **n_images (int)**: Number of images to import.
        - **imgs_shp (tuple[int])**: Shape of the images (height, width).
        - `**kwargs`: Additional parameters.
            - **BG_path (str)**: Path to the background image to be subtracted. Default is ''.
            - **resize_img (tuple[int])**: Tuple specifying the dimensions to resize the images to (width, height). Default is the original image shape.

    Returns:
        - tuple: A tuple containing:
            - original_img_list (list[np.ndarray]): List of original images (resized if specified).
            - img_list (list[np.ndarray]): List of grayscale images with the background removed if provided.

    Example:
        >>> from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
        >>> instance = IncTrac(f)
        >>> pathlist = ['path/to/image1.jpg', 'path/to/image2.jpg']
        >>> indices = [0, 1]
        >>> n_images = 2
        >>> shape = (100, 200)
        >>> original_imgs, processed_imgs = instance.ImportingFiles(pathlist, indices, n_images, shape)
        >>> print(original_imgs, processed_imgs)

    .. note ::
        - The function reads images from the specified paths, converts them to grayscale, and optionally removes a background image.
        - The images can be resized if the `resize_img` parameter is provided in kwargs.

    """
    print(f'Importing {n_images} images ...')
    img_list=[]; n = 0; original_img_list=[]

    # Get additional parameters from kwargs
    BG_path = kwargs.get('BG_path', '')
    resize_img = kwargs.get('resize_img', (imgs_shp[1],imgs_shp[0]))

    # Import images
    for i in indices_list:
        img = cv2.imread(pathlist[i])
        # original_img_list.append(cv2.resize(img.astype('float32'), resize_img))

        # Resize and store the original image if needed, and Convert image to grayscale
        img_list.append(cv2.cvtColor(cv2.resize(img.astype('float32'), resize_img), cv2.COLOR_BGR2GRAY))

        # Print progress
        n += 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n_images/20)), int(5*n/(n_images/20))))
    print('')

    # Remove background if path is provided
    if len(BG_path) > 0:
        print('Removing background image ...', end=" ")
        BG = cv2.imread(BG_path)
        BG = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY).astype(np.float32)

        BG_len , BG_Wid  = BG.shape
        img_len, img_wid = imgs_shp

        # Adjust background size if resizing is specified
        if resize_img != imgs_shp:
            BG_len = resize_img[0]; r = BG_Wid / BG_len
            BG = BG[0:BG_len,0:r*BG_len]

        if BG_len < img_len: img_len = BG_len; img_wid = BG_Wid
        else: BG = BG[0:img_len,0:img_wid]

        # Subtract the background from each image in the list
        for img in img_list:
            img = cv2.subtract(img[0:img_len,0:img_wid],BG)
        print(u'\u2713')
    return original_img_list, img_list