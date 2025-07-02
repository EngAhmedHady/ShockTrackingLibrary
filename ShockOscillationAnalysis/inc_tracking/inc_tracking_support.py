# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:15:04 2024

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from ..support_func import log_message
from ..constants import BCOLOR, CVColor
from ..linedrawingfunctions import InclinedLine
from scipy.interpolate import CubicSpline, PchipInterpolator


def v_least_squares(xLoc: list[float], columnY:list[float], nSlices: int) -> list[float]:
    """
    Perform a vertical least squares linear regression to find the slope.

    Parameters:
        - **xLoc (list[float])**: List of x-coordinates of the points.
        - **columnY (list[float])**: List of y-coordinates of the points.
        - **nSlices (int)**: Number of slices or data points.

    Returns:
        float: The slope of the best-fit line. 
        Returns `inf` if the slope cannot be determined (e.g., division by zero).

    Example:
        >>> xLoc = [1, 2, 3, 4, 5]
        >>> columnY = [2, 4, 6, 8, 10]
        >>> nSlices = 5
        >>> slope = v_least_squares(xLoc, columnY, nSlices)
        >>> print(slope)  # Output: 2.0

    .. note::
        - The function calculates the slope of the best-fit line using the vertical least squares method.
        - If the denominator in the formula is zero, the function returns ``np.inf``.
    
    Equations:
        - The slope :math:`m` is calculated using the formula:
          
          .. math::
            m = \\frac{n \\cdot \\sum (y^2) - (\\sum y)^2}{n \\cdot \\sum (x \\cdot y) - (\sum x)(\sum y)}

          where :math:`n` is the number of points (slices).
    """
    # Compute the required summations
    xy = np.array(xLoc)*columnY; yy = columnY**2
    x_sum = np.sum(xLoc)       ; y_sum = np.sum(columnY)
    xy_sum = np.sum(xy)        ; yy_sum = np.sum(yy)

    # Compute the denominator of the slope formula
    denominator = nSlices * xy_sum - x_sum * y_sum

    # Compute the numerator of the slope formula
    numerator = nSlices * yy_sum - y_sum ** 2

    # Check for division by zero or undefined conditions
    if denominator != 0:
        # Return the slope as a float
        return numerator / denominator
    else:
        # Return infinity if the slope cannot be determined
        return np.inf

def ransac(x:np.ndarray, y:np.ndarray, threshold:float, 
           e:float=0.3, p:float=0.999, n_samples:int=5, 
           max_trials:int=0, log_dirc:str='') -> tuple[float]:
    """
    Perform RANSAC (Random Sample Consensus) algorithm for robust linear model fitting.

    This function identifies the best linear model for a dataset with potential outliers 
    by iteratively fitting models to random subsets of the data and evaluating their inlier scores.

    Parameters:
        - **x (np.ndarray)**: Array of independent variable values.
        - **y (np.ndarray)**: Array of dependent variable values.
        - **threshold (float)**: Threshold for classifying points as inliers based on their residuals.
        - **e (float, optional)**: Estimated outlier ratio. Defaults to 0.3.
          Represents the proportion of outliers in the dataset.
        - **p (float, optional)**: Desired probability of selecting at least one outlier-free subset. Defaults to 0.999.
        - **n_samples (int, optional)**: Number of random points to select for model fitting in each iteration. Defaults to 5.
        - **max_trials (int, optional)**: Maximum number of RANSAC iterations. Defaults to 0, where the number of trials 
          is automatically computed based on `e` and `p`.
        - **log_dirc (str)**: log file directory.

    Returns:
        - tuple:
            - **best_model (float)**: The slope of the best-fitting linear model.
            - **best_inlier_xmean (float)**: Mean of the x-coordinates of the inliers corresponding to the best model.
            - **best_inlier_ymean (float)**: Mean of the y-coordinates of the inliers corresponding to the best model.

    Raises:
        - Exception: If the algorithm fails to find a valid model, the function will print debugging information.
    
    Equations:
        - The number of iterations :math:`n_{tries}`; however, can be roughly determined as a function of the desired probability of success :math:`p` as shown below.
          
          .. math::
            n_{tries} = \\frac{\\log(1 - p)}{\\log(1 - (1 - e)^{n_s})}

          where: :math:`e` is number of inliers in data to number of points in data. 
          here assumed to be 30% of the points are inlier and :math:`n_s` is the 
          sample size among the existed points
        
        - The outliers are estimated based on the distance between the existing points and predected line as follow:
          
          .. math::
            e = |x - x_{pred}|
            
          where:
          
          .. math::
            x_{pred} =             
                \\begin{cases}
                    \\frac{y - c}{m} \\ \\forall m != \\infty \\

                    
                    \\overline{x} \\ \\forall m = \\infty
                \\end{cases}
            

    Example:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.1, 4.2, 6.1, 8.0, 10.2])
        >>> threshold = 0.5
        >>> slope, inlier_mean = ransac(x, y, threshold)
        >>> print("Slope:", slope)
        >>> print("Inlier mean:", inlier_mean)

    .. note::
        - The number of tracking points should be more than the sample size :math:`n_s` by 1 at least
        - This function relies on the :func:`v_least_squares <ShockOscillationAnalysis.inc_tracking.inc_tracking_support.v_least_squares>` function for linear model fitting.
        - Outliers are automatically excluded based on the residual threshold.
        - The function is designed to handle datasets with a moderate proportion of outliers.

    """
    best_model = None
    # best_inliers = []
    best_score = 0
    # e: estimated outlier ratio (n_outlaier/data_set_size)
    # p: probability to select at least 1 outlier within a sample

    N = len(x)

    if N <= n_samples + 1:
        m = v_least_squares(x, y, N)
        return m, np.mean(x)

    n_trials = round(np.log(1-p)/np.log(1-(1-e)**n_samples))
    max_trials=n_trials if max_trials<1 else max_trials
    for _ in range(max_trials):
        # Randomly select n_samples points
        sample_indices = random.sample(range(N), n_samples)

        # Fit a linear model to the sample
        x_sample, y_sample = x[sample_indices], y[sample_indices]
        m = v_least_squares(x_sample, y_sample, 5)
        c = np.mean(y_sample)-m*np.mean(x_sample)

        # Calculate residuals (distance to the fitted line)
        x_pred = (y - c)/m if m != np.inf else np.full_like(x, np.mean(x_sample))
        residuals = np.abs(x - x_pred)

        # Identify inliers (points with residuals below the threshold)
        inliers = np.where(residuals < threshold)[0]

        # Score based on the number of inliers
        if len(inliers) > best_score:
            best_score = len(inliers)
            best_model = m
            best_inliers = inliers
    try:
        return best_model, np.mean(x[best_inliers]), np.mean(y[best_inliers])
    except Exception as e:
        log_message(e, log_dirc)
        print(e, x, y, best_model, inliers)
        

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

def anglesInterpolation(pnts_y_list: list[int],       # Generated points by class
                        flow_dir: list[float] = None, # measured data (LDA, CFD, ... )
                        flow_Vxy:list[tuple] = None,  # measured data (LDA, CFD, ... )
                        log_dirc:str='',
                        **kwargs) -> list[float]:     # other parameters
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
    angle_interp_kind = kwargs.get('angle_interp_kind', 'linear')
    if angle_interp_kind not in ['linear', 'CubicSpline', 'PCHIP']:
        warning = 'Interpolation method is not supported!;'
        action = 'Linear interpolation will be used'
        log_message(f'Warning: {warning}', log_dirc)
        log_message(action, log_dirc)
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
        print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
        angle_interp_kind = 'linear'

    if flow_dir is not None:
        # Unzip the angles_list into separate locs and angles lists
        locs, angles = zip(*flow_dir)
    elif flow_Vxy is not None:
        # Unzip the Vxy into separate locs, Vx, Vy lists
        locs, Vx, Vy = zip(*flow_Vxy)
        angles = np.arctan(np.array(Vy)/np.array(Vx))*180/np.pi

    if min(locs) > min(pnts_y_list) or max(locs) < max(pnts_y_list):
        warning = 'Provided y-domain is out of valid range!;'
        match angle_interp_kind:
            case 'linear': 
                action = 'Only boundary angles will considered ...'
            case _:
                action = 'First derivative at curves ends will considered zero,'
                action += 'overshooting is likely occurs ...'
                
        log_message(f'Warning: {warning}', log_dirc)
        log_message(action, log_dirc)    
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
        print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
    
    if angle_interp_kind == 'linear':
        intr_flow_dir = np.interp(pnts_y_list, locs, angles)
    elif angle_interp_kind == 'CubicSpline':
        interp_fun = CubicSpline(locs, angles, bc_type = 'clamped')
        intr_flow_dir = interp_fun(pnts_y_list)
    elif angle_interp_kind == 'PCHIP':
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

def import_gray(img, resize_img):
    # Resize the image if needed and convert to gray scale
    img = cv2.cvtColor(cv2.resize(img.astype('float32'), resize_img), cv2.COLOR_BGR2GRAY)
    return img

def import_other(img, resize_img):
    # Resize the image if needed
    img = cv2.resize(img.astype('float32'), resize_img)
    return img

def rotate90(img):
    """
    Rotate an image 90 degrees clockwise.

    This function transposes the image matrix (swaps rows and columns) and 
    flips it horizontally to achieve a 90-degree clockwise rotation.

    Parameters:
        - **img (numpy.ndarray)**: Input image as a NumPy array. The image should be in a standard format 
          (e.g., grayscale or RGB) compatible with OpenCV operations.

    Returns:
        - **numpy.ndarray**: Rotated image with the same format as the input.

    Example:
        >>> import cv2
        >>> img = cv2.imread('example.jpg')  # Load an image
        >>> rotated_img = rotate90(img)     # Rotate it 90 degrees
        >>> cv2.imshow('Rotated Image', rotated_img)  # Display the rotated image
        >>> cv2.waitKey(0)
        >>> cv2.destroyAllWindows()
    """
    img = cv2.transpose(img)
    img = cv2.flip(img, 1)
    return img

def doNone(img): return img

def ImportingFiles(pathlist: list[str], indices_list: list[int], n_images: int, # Importing info.
                   imgs_shp: tuple[int], import_type = 'gray_scale',            # Images info.
                   log_dirc:str='', **kwargs) -> dict[int, np.ndarray]:         # Other parameters
    """
    Import images from specified paths, optionally crop, resize or rotate them.

    Parameters:
        - **pathlist (list[str])**: List of paths to the image files to be imported.
        - **indices_list (list[int])**: List of indices indicating which images from `pathlist` should be imported.
        - **n_images (int)**: Number of images to import.
        - **imgs_shp (tuple[int])**: Desired shape of the images (height, width).
        - **import_type (str, optional)**: Type of image import. Can be 'gray_scale' for grayscale images or 'other' for other types. Default is 'other'.
        - **log_dirc (str)**: log file directory.
        - **kwargs (dict, optional)**: Additional parameters:
            - **resize_img (tuple[int], optional)**: Tuple specifying the desired dimensions for resizing the images (width, height). Default is the original shape of the images.
            - **crop_y_img (tuple[int], optional)**: Tuple specifying the cropping range along the y-axis (min, max). Default is to crop the entire image along y.
            - **crop_x_img (tuple[int], optional)**: Tuple specifying the cropping range along the x-axis (min, max). Default is to crop the entire image along x.
            - **rotate90_img (int, optional)**: If set to 1, rotates the images 90 degrees clockwise. Default is 0 (no rotation).

    Returns:
        - tuple:
            - **img_list (dict[int, np.ndarray])**: Dictionary where the keys are indices from `indices_list` and the values are the corresponding processed images.
    
    Example Flow:
        1. Import images from the list of paths specified in `pathlist` based on the indices in `indices_list`.
        2. Apply cropping if specified in `kwargs`.
        3. Resize the images based on `resize_img` parameter if provided.
        4. Rotate the images 90 degrees if `rotate90_img` is set to ``1``.
        5. Return a dictionary of images, indexed by the values in `indices_list`.
        
    Example:
        >>> from ShockOscillationAnalysis import InclinedShockTracking as IncTrac
        >>> instance = IncTrac(f)
        >>> pathlist = ['path/to/image1.jpg', 'path/to/image2.jpg']
        >>> indices = [0, 1]
        >>> n_images = 2
        >>> shape = (100, 200)
        >>> original_imgs, processed_imgs = instance.ImportingFiles(pathlist, indices, n_images, shape)
        >>> print(original_imgs, processed_imgs)

    .. note::
        - The function uses different import functions based on the `import_type` parameter. 
            - If the type is 'gray_scale', the `import_gray` function is used to convert the images to grayscale. 
            - Otherwise, the `other` function is used for other image types.
        - Images are resized to the specified dimensions if `resize_img` is provided in the keyword arguments.
        - Cropping is applied to the images based on `crop_x_img` and `crop_y_img` parameters, and the images can optionally be rotated by 90 degrees.
        - Progress is displayed on the console while the images are being imported.
    """
    new_log = f'Importing {n_images} images...'
    log_message(new_log, log_dirc)
    print(new_log)
    img_list={}

    # Get additional parameters from kwargs
    crop_y_img = kwargs.get('crop_y_img', (0, imgs_shp[0]))
    crop_x_img = kwargs.get('crop_x_img', (0, imgs_shp[1]))
    croped_img_shp = (crop_y_img[1]-crop_y_img[0], crop_x_img[1]-crop_x_img[0])
    resize_img = kwargs.get('resize_img', (croped_img_shp[1], croped_img_shp[0]))
    rotate90_img = kwargs.get('rotate90_img', 0)
    if import_type == 'gray_scale':
        import_func = import_gray
    elif import_type == 'other':
        import_func = import_other 

    if rotate90_img:
        oriant = rotate90
    else:
        oriant = doNone
    # Import images
    for n, i in enumerate(indices_list):
        img = cv2.imread(pathlist[i])
        # print(img.shape)
        # crop the original image if needed
        img = img[crop_y_img[0]: crop_y_img[1],
                  crop_x_img[0]: crop_x_img[1]]
        
        # print(img.shape, resize_img)
        img = import_func(img, resize_img)
        img_list[i] = oriant(img)

        # Print progress
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n_images/20)), int(5*n/(n_images/20))))
    print()
    log_message('Done', log_dirc)
    return img_list
    