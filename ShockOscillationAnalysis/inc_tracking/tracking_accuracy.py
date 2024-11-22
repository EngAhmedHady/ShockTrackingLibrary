import numpy as np
from scipy import stats
from ..preview import residual_preview
from ..ShockOscillationAnalysis import BCOLOR
from ..linedrawingfunctions import AngleFromSlope
from .inc_tracking_support import ransac

def save_data_txt(outlier_p, hi, leverage_lim, img_indx=None, output_directory='', comment=''):
    if len(output_directory) > 0:
       log_file_path = f"{output_directory}/outliers_{comment}.txt"
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
    Identifies outliers in an error array using the Interquartile Range (IQR) method and
    logs details if certain leverage conditions are met.

    Parameters:
        - **error (list[float])**: Array of error values to analyze.
        - **y_dp (list[float])**: Array of associated data points to calculate leverage.
        - **count (int)**: Current count index, used for logging purposes.
        - **output_directory (str)**: Directory to save outlier logs. Defaults to an empty string.
        - **comment (str)**: Comment to be added to the log file name. Defaults to an empty string.
        - **img_indx (Optional[list[Any]])**: List of image indices for logging outliers. Defaults to None.

    Keyword Arguments:
        **residual_preview (bool)**: Whether to generate a preview of residuals. Defaults to False.

    Returns:
        list[tuple[float, int, int]]: List of tuples for outlier values, each containing:
            - The error value.
            - Position of the error in the array.
            - The count index for the image.
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
        if resid_preview: residual_preview(error, (e_median,Q1,Q2,IQR), nSlices, img_indx)
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
        list of tuple
            A list of tuples, where each tuple contains:
            - Predicted x-location (float)
            - Corresponding confidence interval (float)

    Raises:
        ValueError
        If nSlices is less than 3 (as at least 2 degrees of freedom are required).

    Example:
        >>> xloc = [1.2, 2.3, 3.4, 4.5]
        >>> columnY = [2.1, 3.2, 4.3, 5.4]
        >>> nSlices = 4
        >>> l_slope = 1.0
        >>> l_yint = 1.0
        >>> conf_est(xloc, columnY, nSlices, l_slope, l_yint)
        [(1.1, 0.28), (2.1, 0.36), (3.2, 0.45), (4.3, 0.56)]

    .. note::
        This function calculates the residual sum of squares and confidence intervals
        for the given x-locations based on a linear fit to the corresponding y-values. The
        confidence interval is computed using the t-distribution for the specified number
        of slices.
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
    Spre= np.sqrt(s**2+Sx**2)

    return list(zip(x_dash, Sx * t, Spre * t)), s

def pop_outlier(indx, xloc, columnY, n_slice_new):
    newxloc = np.delete(xloc, indx)
    newyloc = np.delete(columnY, indx)
    # new_shock_slope = v_least_squares(newxloc, newyloc, n_slice_new - 1)
    new_shock_slope, new_midxloc = ransac(newxloc, newyloc, 1)
    
    # new_midxloc = np.mean(newxloc)
    return new_shock_slope, new_midxloc, newxloc, newyloc, [xloc[indx], columnY[indx]], len(newyloc)
# 
def outlier_correction(outliers_set: list[list[float, int, int]],
                       xlocs: list[list[float]], columnY: list[int],
                       t: float) -> list[list[float, float, list[int], float, float]]:
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
            n_slope, n_midxloc, nxloc, nyloc, popy, n_slice_new = pop_outlier(
                outlier[1], nxloc, nyloc, n_slice_new
            )
            removed_outliers.append(popy)
            for outlier in outliers: outlier[1] -= 1

        # Recalculate averages and error metrics
        n_y_avg = np.mean(nyloc)
        n_y_int = n_y_avg - n_midxloc * n_slope
        n_y_ss = (nyloc - n_y_avg) ** 2
        n_Sty = np.sum(n_y_ss)
        n_y_dp = n_y_ss / n_Sty

        n_e , n_s = error_analysis(nxloc, nyloc, n_slice_new, n_slope, n_y_int, t, n_y_dp)
        # Append corrected parameters
        corrections.append([n_slope, n_midxloc, removed_outliers, n_e, n_s, n_Sty])
    return corrections

def compute_weighted_average(slope: np.ndarray, Sm: np.ndarray, img_set_size: int) -> tuple[float]:
    """
    Computes the weighted average slope, uncertainty, and weighted average angle.

    Parameters:
        - **slope (np.ndarray)**: Array of slope values.
        - **Sm (np.ndarray)**: Array of standard diviation error associated with the slopes.
        - **img_set_size (int)**: Total number of images in the dataset.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - m_avg (float): Weighted average slope.
            - Sm_avg (float): Combined uncertainty of the average slope.
            - w_avg_ang (float): Weighted average angle in degrees, considering zero-uncertainty cases.
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
             columnY: list[int], y_avg: int,
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
        - **y_avg (int)**: The average Y value used for reference.
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
        tuple[List[List[float]], List[int], float, float]
            - `e`: List of error values for each slice.
            - `pop_ylist`: Updated list of Y-values.
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
        min_nSlices = 'nSlices must be at least 3 to have enough degrees of freedom.'
        print(f'{BCOLOR.FAIL}Error:{BCOLOR.ENDC}{BCOLOR.ITALIC}{min_nSlices}{BCOLOR.ENDC}')
        return e, pop_ylist, 0, 0

    df = nSlices - 2
    t = stats.t.ppf(conf_interval, df)

    # Calculate y statistics
    y_ss = (columnY-y_avg)**2
    y_dp = y_ss/np.sum(y_ss)

    # initiate lists
    outliers_set = []
    Sty = np.ones(img_set_size)*np.sum(y_ss) # y total sum of squres
    s = np.zeros(img_set_size)

    for i, xloc in enumerate(xlocs):
        y_int = y_avg - midLocs[i]*slope[i]
        e[i], s[i] = error_analysis(xloc, columnY, nSlices, slope[i], y_int, t, y_dp)
        error, _, _ = zip(*e[i])
        error = np.array(error)-xloc
        outliers = IQR(error, y_dp, columnY, uncertainY_list[i], i, img_indx[i], output_directory, comment, **kwargs)
        if outliers != []: outliers_set.append(outliers)

    correction = outlier_correction(outliers_set, xlocs, columnY, t)

    for i, outliers in enumerate(outliers_set):
        j = outliers[0][2]
        slope[j], midLocs[j], pop_ylist[j], e[j], s[j], Sty[j] = correction[i]
        shock_deg[j] = AngleFromSlope(slope[j])

    Sm = s / np.sqrt(Sty)

    Sm_avg, w_avg_ang = compute_weighted_average(slope, Sm, img_set_size)
    # Confidence interval for slope
    m_conf_int = t * Sm_avg
    conf_ang = 180-AngleFromSlope(m_conf_int)
    print(u'\u2713')
    # Display results
    print(f'weighted average shock angle: {w_avg_ang:0.2f}\u00B1{conf_ang:0.3f} deg',
            end='')
    print(f',\t \u03C3 = {Sm_avg:0.5f}')

    return e, pop_ylist, slope, shock_deg, midLocs, w_avg_ang, conf_ang

