# -*- coding: utf-8 -*-
"""
Created on Tue Nov 5 10:15:04 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
from scipy import stats
from ..constants import BCOLOR
from ..preview import residual_preview
from ..support_func import log_message
from .inc_tracking_support import ransac
from ..linedrawingfunctions import AngleFromSlope

def save_data_txt(outlier_p:list[tuple[float, int]], hi:np.ndarray,
                  leverage_lim:float, img_indx:int=None, 
                  output_directory:str='', comment:str=''):
    """
    Save outlier data and leverage information to a text file.

    This function logs detected outliers and leverage information into a text file 
    for analysis and review.

    Parameters:
        - **outlier_p (list[tuple[float, int, any]])**: List of detected outliers.
            Each entry is a tuple containing:
                - **Error (float)**: Magnitude of the error for the outlier.
                - **Position (int)**: Position of the detected outlier.
                - **Index (int)**: Image index.
        - **hi (np.ndarray)**: Array of leverage values for the dataset.
        - **leverage_lim (float)**: Leverage limit for identifying influential points.
        - **img_indx (int, optional)**: Index or identifier for the image being analyzed.
          Default is `None`, which records "N/A" in the log.
        - **output_directory (str, optional)**: Directory to save the output file.
          Default is an empty string, which saves to the current working directory.
        - **comment (str, optional)**: Additional comment to append to the output file name.
          Default is an empty string.

    Returns:
        - None

    Example:
        >>> outliers = [(0.12, 3, None), (0.15, 7, None)]
        >>> leverage_values = np.array([0.1, 0.2, 0.15, 0.08])
        >>> leverage_limit = 0.2
        >>> save_data_txt(outliers, leverage_values, leverage_limit, img_indx=5, output_directory='logs', comment='test')
    
    .. note::
        - The output file is named `outliers_<comment>.txt`, where `<comment>` is an optional user-provided string.
        - Appends data to the file if it already exists; otherwise, creates a new file.
        - Uses leverage values to summarize data points that might have significant influence.
    """
    if len(comment) > 0: comment=f'_{comment}'
    if len(output_directory) > 0:
       log_file_path = f"{output_directory}/outliers{comment}.txt"
       with open(log_file_path, "a") as f:
           for e, pos, _ in outlier_p:
               img_index_info = img_indx if img_indx is not None else "N/A"
               f.write(f'Outlier detected: Error={e}, Position={pos + 1}, ImageIndex={img_index_info}\n')
           f.write(f'Outlier leverage: {np.sum(hi)}, H0 = {leverage_lim}\n')

def IQR(error: list[float], y_dp: list[float],
        columnY: list[int], uncertain_y: list[int],
        count: int=0, img_indx: list[int]=None,
        output_directory: str='', comment: str='', **kwargs) -> list[tuple[float, int, int]]:

    """
    Detect outliers in a dataset using the Interquartile Range (IQR) method, calculate leverage, 
    and log significant results if leverage exceeds a specified threshold.

    Parameters:
        - **error (list[float])**: Array of error values to analyze.
        - **y_dp (list[float])**: Data points used to calculate leverage values for each error.
        - **columnY (list[int])**: Indexes corresponding to the vertical axis or other tracking information.
        - **uncertain_y (list[int])**: Y-values or column indexes considered uncertain.
        - **count (int, optional)**: Index counter for the current dataset or image. Default is `0`.
        - **img_indx (list[int], optional)**: List of image indices for reference in logs. Default is `None`.
        - **output_directory (str, optional)**: Directory to save outlier logs. Default is an empty string.
        - **comment (str, optional)**: Comment to append to log filenames. Default is an empty string.

    Keyword Arguments:
        - **residual_preview (bool, optional)**: If `True`, generates a residuals preview plot for visualization. Default is `False`.

    Returns:
        - **list[tuple[float, int, int]]**: List of detected outliers as tuples:
            - **float**: The error value.
            - **int**: Position (index) of the error in the dataset.
            - **int**: The count index of the current dataset.

    Example:
        >>> errors = [1.2, 0.5, 3.6, 2.1, 1.8]
        >>> y_dp = [0.1, 0.05, 0.2, 0.15, 0.12]
        >>> colY = [1, 2, 3, 4, 5]
        >>> uncertain_y = [2, 3, 4]
        >>> outliers = IQR(errors, y_dp, colY, uncertain_y, count=1, output_directory='logs', comment='test')
        >>> print(outliers)

    .. note::
        - Leverage threshold :math:`H_0 = \\frac{3(p+1)}{n_s}` where :math:`p` is the number of
          independant variable (which is ``1`` in this condition). That ensures that significant 
          points influencing the model are flagged.
        - Uses the IQR method to identify outliers robustly, focusing on uncertain data columns.
        - Relies on :func:`save_data_txt <ShockOscillationAnalysis.inc_tracking.tracking_accuracy.save_data_txt>`
          to log outlier details to a text file.
        - Visualization of residuals requires enabling `residual_preview`.

    .. image:: _static/img/IQR.png
        :width: 400
        :align: center

    Equations:
        - Median:
          :math:`e_{median} = \\text{median}(e^2)` where :math:`e = x - x_{pred}` 
          and :math:`x` the detected shock location and :math:`x_{pred}` is the 
          loction on the fitted line with :func:`RANSAC <ShockOscillationAnalysis.inc_tracking.inc_tracking_support.ransac>`
          or :func:`least square <ShockOscillationAnalysis.inc_tracking.inc_tracking_support.v_least_squares>`
        - Quartiles:
          :math:`Q1, Q2 = \\text{median of lower and upper halves of sorted}(e^2)`
        - Interquartile Range:
          :math:`\\text{IQR} = Q2 - Q1`
        - Outlier Detection:
          :math:`outlier < Q1-1.5IQR < IQR < Q2+1.5IQR < outlier`
        - Leverage of each point:
          :math:`H_i = \\frac{1}{n_{Slices}} + y_{dp[i]}` 
          where :math:`y_{dp[i]} = \\frac{(y_i - \\overline{y})^2}{\\sum{(y_i - \\overline{y})^2}}`
          and :math:`y` is independant variable represents the vertical location of the slice and
          :math:`i` is the point index
    """
    # Number of slices and median of the error array
    nSlices = len(error)
    e_median = np.median(error)

    # Calculate the first and third quartiles (Q1, Q2) of the sorted error array
    Q1_array, Q2_array = np.array_split(sorted(error), 2)

    Q1 = np.median(Q1_array)
    Q2 = np.median(Q2_array)
    # Interquartile range
    IQR = Q2 - Q1

    # Detect outliers based on the IQR range
    outlier = [
               [e, i, count]
               for i, e in enumerate(error)
               if not (Q1 - 1.5 * IQR <= e <= Q2 + 1.5 * IQR) and columnY[i] in uncertain_y
              ]

    # Calculate leverage points for outliers
    hi = [(1 / nSlices) + y_dp[i] for i, e in enumerate(error)
          if not (Q1 - 1.5 * IQR <= e <= Q2 + 1.5 * IQR)]

    # If the total leverage exceeds a threshold, log the outliers
    if len(hi) > 0 and np.sum(hi) > (3*2)/nSlices and len(outlier) > 0:
        lev_th = (3*2)/nSlices
        save_data_txt(outlier, hi, lev_th, img_indx, output_directory, comment)
        # If residual preview is requested, call the preview function
        resid_preview = kwargs.get('residual_preview', False)
        if resid_preview: residual_preview(error, (e_median,Q1,Q2,IQR), nSlices, img_indx,
                                           log_dirc=output_directory)
    else:
        outlier = [] # Clear outliers if leverage is below the threshold
    return outlier

def error_analysis(xloc: list[float], columnY:list[float], nSlices: int,
              l_slope: float, l_yint: float, t: float, y_dp: list[float]):
    """
    Estimate confidence intervals for x-locations based on a linear model and calculate residuals.

    Parameters:
        - **xloc (list[float])**: List of actual x-coordinates.
        - **columnY (list[float])**: List of y-coordinates.
        - **nSlices (int)**: Number of data points.
        - **l_slope (float)**: Slope of the linear regression line.
        - **l_yint (float)**: y-intercept of the linear regression line.

    Returns:
        tuple:
            - **list of tuples**: Each tuple contains:
                - Predicted x-location (float)
                - Confidence interval (float)
                - Prediction interval (float)
            - **float**: Standard error of the residuals.

    Raises:
        ValueError
        If nSlices is less than 3 (as at least 2 degrees of freedom are required).

    Example:
       >>> xloc = [1.2, 2.3, 3.4, 4.5]
        >>> columnY = [2.1, 3.2, 4.3, 5.4]
        >>> nSlices = 4
        >>> l_slope = 1.0
        >>> l_yint = 1.0
        >>> t = 2.776  # t-statistic for 95% confidence with 2 degrees of freedom
        >>> y_dp = [0.1, 0.2, 0.3, 0.4]
        >>> intervals, std_error = error_analysis(xloc, columnY, nSlices, l_slope, l_yint, t, y_dp)
        >>> print(intervals)
        [(1.1, 0.28, 0.38), (2.1, 0.36, 0.47), (3.2, 0.45, 0.56), (4.3, 0.56, 0.68)]
        >>> print(std_error)
        0.134

    .. note::
        This function calculates the residual sum of squares and confidence intervals
        for the given x-locations based on a linear fit to the corresponding y-values. The
        confidence interval is computed using the t-distribution for the specified number
        of slices.

    .. image:: _static/img/CIandPI.png
        :width: 400
        :align: center
    
    Equations:
        - **Residuals** are calculated as:
          
          .. math::
            e_i = x_i - x_{pred [i]}

          where :math:`x_{pred [i]}` is the predicted x-location based on the fitted line.
        - **Confidence Interval**:
          
          .. math::
            CI_i = t_{\\alpha/2} \\cdot s \\cdot \\sqrt{\\frac{1}{n_s} + y_{\\text{dp[i]}}}
            
          where:
            - :math:`t_{\\alpha/2}` is the t-distribution value for a given confidence level.
            - :math:`s = \\sqrt{SSE/dof}` is the standard error where :math:`SSE = \\sum{e_i^2}` 
              is the sum of the squre error and :math:`dof` is the degree of freedom,
              in case of line analysis :math:`dof = n_s - 2` and :math:`n_s` is number of slices. 
            - :math:`y_{dp[i]} = \\frac{(y_i - \\overline{y})^2}{\\sum{(y_i - \\overline{y})^2}}`
              and :math:`y` is independant variable represents the vertical location of the slice

        - **Prediction Interval**:
          
          .. math::
            PI_i = t_{\\alpha/2} \\cdot \\sqrt{s^2 + \\text{CI}^2}

    """
    # Convert to NumPy arrays for vectorized operations
    xloc = np.array(xloc)
    columnY = np.array(columnY)

    # Calculate predicted x-locations from the linear model
    x_dash = (columnY - l_yint) / l_slope if l_slope != np.inf else np.ones(nSlices)*np.mean(xloc)

    error = xloc - x_dash

    # Calculate the sum of squared errors
    Se = np.sum(error ** 2)

    # Standard error and t-statistics
    df = nSlices - 2  # ........ Calculate degree of freedom
    s = np.sqrt(Se / df)

    # Compute confidence intervals for each slice
    Sx = s * np.sqrt((1 / nSlices) + y_dp)

    # Compute prediction interval for each slice
    Spre= np.sqrt(s**2 + Sx**2)
    
    # Package results as a list of tuples
    return list(zip(x_dash, Sx * t, Spre * t)), s

def pop_outlier(indx:int, xloc:np.ndarray, columnY:np.ndarray, 
                n_slice_new:int, log_dirc:str='') -> tuple:
    """
    Removes an outlier at a given index and recomputes the RANSAC-based shock slope.

    Parameters:
        - **indx (int)**: Index of the outlier to remove.
        - **xloc (np.ndarray)**: Array of x-locations.
        - **columnY (np.ndarray)**: Corresponding y-values.
        - **n_slice_new (int)**: Number of slices after removing the outlier (not used in function body).
        - **log_dirc (str, optional)**: log file directory. Default is ''.

    Returns:
        tuple:
            - new_shock_slope (float): Slope estimated after removing the outlier.
            - new_midxloc (np.ndarray): Fitted x-locations from RANSAC.
            - new_midyloc (np.ndarray): Fitted y-locations from RANSAC.
            - newxloc (np.ndarray): xloc after removing the outlier.
            - newyloc (np.ndarray): columnY after removing the outlier.
            - removed_point (list): The removed [x, y] outlier point.
            - new_count (int): Length of the updated y-array.
    """
    newxloc = np.delete(xloc, indx)
    newyloc = np.delete(columnY, indx)
    new_shock_slope, new_midxloc, new_midyloc = ransac(newxloc, newyloc, 1, log_dirc=log_dirc)
    
    return new_shock_slope, new_midxloc, new_midyloc, newxloc, newyloc, [xloc[indx], columnY[indx]], len(newyloc)

def outlier_correction(outliers_set: list[list[float, int, int]],
                       xlocs: list[list[float]], columnY: list[int],
                       t: float, log_dirc:str='') -> list[list[float, float, list[int], float, float]]:
    """
    Corrects for outliers by iteratively removing them, recalculating slopes, midpoints,
    and associated statistics.

    Parameters:
        - **outliers_set (list)**: List of outliers, each described by [value, index, set_index].
        - **xlocs (list)**: List of x-locations for each dataset.
        - **columnY (list)**: List of y-values.
        - **t (float)**: t-value for statistical analysis.

    Returns:
        - correction: List of corrected parameters:
            [new_slope, new_midpoint, removed_outliers, new_Sm, new_Sty].
    """
    nSlices = len(columnY)
    corrections = []
    for outliers in outliers_set:
        # Extract the relevant x-locations and initialize variables
        set_idx = outliers[0][2]
        nxloc = xlocs[set_idx]
        nyloc = columnY.copy()
        n_slice_new = nSlices
        removed_outliers = []

        for outlier in outliers:
            # Update parameters by removing the outlier
            n_slope, n_midxloc, n_midyloc, nxloc, nyloc, popy, n_slice_new = pop_outlier(
                outlier[1], nxloc, nyloc, n_slice_new,log_dirc=log_dirc
            )
            removed_outliers.append(popy)
            for outlier in outliers: outlier[1] -= 1

        # Recalculate averages and error metrics
        # n_y_avg = np.mean(nyloc)
        n_y_int = n_midyloc - n_midxloc * n_slope
        n_y_ss = (nyloc - n_midyloc) ** 2
        n_Sty = np.sum(n_y_ss)
        n_y_dp = n_y_ss / n_Sty

        n_e , n_s = error_analysis(nxloc, nyloc, n_slice_new, n_slope, n_y_int, t, n_y_dp)
        # Append corrected parameters
        corrections.append([n_slope, n_midxloc, n_midyloc, removed_outliers, n_e, n_s, n_Sty])
    return corrections

def compute_weighted_average(slope: np.ndarray, Sm: np.ndarray, img_set_size: int) -> tuple[float]:
    """
    Computes the weighted average slope, uncertainty, and weighted average angle.

    Parameters:
        - **slope (np.ndarray)**: Array of slope values.
        - **Sm (np.ndarray)**: Array of standard diviation error associated with the slopes.
        - **img_set_size (int)**: Total number of images in the dataset.

    Returns:
        tuple[float, float]: A tuple containing:
            - Sm_avg (float): Combined uncertainty of the average slope.
            - w_avg_ang (float): Weighted average angle in degrees, considering zero-uncertainty cases.
    
    .. note::
        - Handles cases where uncertainties are zero by considering their corresponding angles directly in the weighted average.
            
    Equations:
        - Weighted Average Slope has two conditions:

          .. math::
            \overline{m_1}=\\frac{\sum_{i}^{N-r}\\frac{m_i}{Sm_i^2}}{\sum_{i}^{N-r}\\frac{1}{\\sigma_i^2}}\\forall\\ \\sigma_i>0,
            
            \overline{m_2}=\\sum_{i}^{r}m_i\\forall\\ \\sigma_i=0

          where :math:`\\sigma_i=\\frac{s}{\\sqrt{\\sum\left(Y_j-\\overline{Y}\\right)^2}}`, 
          :math:`s` is the standard error. :math:`Y_j` is the slice location,  
          :math:`\\overline{Y}` is the mean $y$-location of the slices,
          :math:`N` is the total number of images, and
          :math:`r` is the number of images when :math:`\\sigma_i = 0`.

        - Compine Weighted Average slope (including zero-uncertainty cases):

          .. math::
            \\overline{m_{wj}}=\\frac{\\left(\\left(N-r\\right)\\overline{m_1}+r\\overline{m_2}\\right)}{N}

        
        - Uses the relationship between slope and angle:
          
          .. math::
            \\theta_i = \\arctan(m_{wj}) \\cdot \\frac{180}{\\pi}

    """

    zero_indices = []
    valid_slope = []
    valid_Sm = []

    # Filter valid slopes and Sm values
    for idx, (m, s) in enumerate(zip(slope, Sm)):
        if s == 0:
            zero_indices.append(idx) # Store indices where Sm is zero
        elif s > 0 and m != np.inf:
            valid_slope.append(m)
            valid_Sm.append(s)

    # Convert valid values to numpy arrays for efficient computation
    valid_slope = np.array(valid_slope)
    valid_Sm = np.array(valid_Sm)

    # Weighted average slope
    m_avg = np.sum(valid_slope / (valid_Sm ** 2)) / np.sum(1 / (valid_Sm ** 2))
    # Combined uncertainty for the weighted average slope
    Sm_avg = np.sqrt(1 / np.sum(1 / (valid_Sm ** 2)))

    # Include zero S indices in average angle
    zero_angles = [AngleFromSlope(slope[idx]) for idx in zero_indices]
    w_avg_ang = AngleFromSlope(m_avg)
    # Weighted average angle including zero-uncertainty cases
    w_avg_ang = (w_avg_ang * (img_set_size - len(zero_indices)) + sum(zero_angles)) / img_set_size

    return Sm_avg, w_avg_ang

def conf_lim(xlocs: list[list[float]], midLocs: list[float],
             columnY: list[int], y_avg: list[float],
             slope: list[float], img_indx: list[int], shock_deg: list[float],
             e: list[list[float]], pop_ylist: list[int],
             uncertainY_list: list[list[int]],
             output_directory: str ='', comment: str='',
             **kwargs) -> tuple[list[list[float]], list[int],
                                list[float], list[float], list[float],
                                float, float]:

    """
    This function calculates the confidence limits for shock angles based on the provided shock tracking data. It identifies outliers using statistical methods and updates the slopes and mid-locations of the shock points. The function also computes the weighted average of the shock angles and the associated confidence interval for the slope.

    Parameters:
        - **xlocs (list[list[float]])**: The x-coordinates for each slice of the shock wave.
        - **midLocs (list[float])**: The midpoint locations for each image.
        - **columnY (list[int])**: Y-values corresponding to each slice.
        - **y_avg (list[float])**: The average Y values used for reference.
        - **slope (list[float])**: The slope values for each image.
        - **shock_deg (list[float])**: The estimated shock angle in degrees for each image.
        - **img_indx (list[int])**: Indexes of the images.
        - **e (list[list[float]])**: Error values for each slice.
        - **pop_ylist (list[int])**: List of Y values for removed points from slices.
        - **uncertainY_list (list[list[int]])**: Indices where the Y-values are uncertain.
        - **output_directory (str, optional)**: Directory to save the output images (default is '').
        - **comment (str, optional)**: Additional comment for the output (default is '').
        - `**kwargs`: additional keyword arguments
            Additional parameters for the functions `error_analysis`, `IQR`, and others.

    Returns:
        tuple
            - `e`: List of error values for each slice.
            - `pop_ylist`: Updated list of Y-values.
            - `slope`: updated list of slops
            - `shock_deg`: updated list of shock angles in degrees
            - `midLocs`: updated list of average x-locations
            - `w_avg_ang`: Weighted average shock angle.
            - `conf_ang`: Confidence angle for the weighted average.


    Example:
        >>> xlocs = [[1.0, 2.0], [2.0, 3.0]]
        >>> midLocs = [5.0, 5.5]
        >>> columnY = [100, 200]
        >>> y_avg = 150
        >>> slope = [0.1, 0.15]
        >>> img_indx = [1, 2]
        >>> e = [[] for _ in range(len(xlocs))]
        >>> pop_ylist = [50, 60]
        >>> conf_lim = 0.95
        >>> uncertainY_list = [[0, 1], [1, 2]]
        >>> result = conf_lim(xlocs, midLocs, columnY, y_avg, slope, img_indx, e, pop_ylist, uncertainY_list, conf_interval=conf_lim)
        >>> print(result)

    .. note ::
        - The `conf_lim` is typically set to values such as 0.95 for 95% confidence.
        - The function assumes that the number of slices is greater than 3; otherwise, it returns an error message.
        - The `outlier_correction` step updates the outlier values based on statistical analysis.
    """

    nSlices = len(columnY)
    img_set_size = len(xlocs)
    conf_interval = kwargs.get('conf_interval', 0)
    # Ensure that the number of slices is sufficient
    if nSlices < 3:
        error = 'nSlices must be at least 3 to have enough degrees of freedom.'
        log_message(error, output_directory)
        print(f'{BCOLOR.FAIL}Error:{BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
        return e, pop_ylist, 0, 0

    df = nSlices - 2
    t = stats.t.ppf(conf_interval, df)

    s = np.zeros(img_set_size)

    for i, xloc in enumerate(xlocs):
        # Calculate y statistics
        y_ss = (columnY-y_avg[i])**2
        y_dp = y_ss/np.sum(y_ss)

        # initiate lists
        outliers_set = []
        Sty = np.ones(img_set_size)*np.sum(y_ss) # y total sum of squres
        y_int = y_avg[i] - midLocs[i]*slope[i]
        e[i], s[i] = error_analysis(xloc, columnY, nSlices, slope[i], y_int, t, y_dp)
        error, _, _ = zip(*e[i])
        error = np.array(error)-xloc
        outliers = IQR(error**2, y_dp, columnY, uncertainY_list[i], i, img_indx[i], 
                       output_directory, comment, **kwargs)
        if outliers != []: outliers_set.append(outliers)

    correction = outlier_correction(outliers_set, xlocs, columnY, t, log_dirc=output_directory)

    for i, outliers in enumerate(outliers_set):
        j = outliers[0][2]
        slope[j], midLocs[j], y_avg[j], pop_ylist[j], e[j], s[j], Sty[j] = correction[i]
        shock_deg[j] = AngleFromSlope(slope[j])

    Sm = s / np.sqrt(Sty)

    Sm_avg, w_avg_ang = compute_weighted_average(slope, Sm, img_set_size)
    # Confidence interval for slope
    m_conf_int = t * Sm_avg
    conf_ang = 180-AngleFromSlope(m_conf_int)
    log_message('Done', output_directory)
    print(u'\u2713')
    # Display results
    new_log=f'weighted average shock angle: {w_avg_ang:0.2f}\u00B1{conf_ang:0.3f} deg'
    new_log2 = f',\t \u03C3 = {Sm_avg:0.5f} deg'
    log_message(f'{new_log}{new_log2}', output_directory)
    print(f'{new_log}{new_log2}')


    return e, pop_ylist, slope, shock_deg, midLocs, y_avg, w_avg_ang, conf_ang

