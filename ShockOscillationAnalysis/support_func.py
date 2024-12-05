import cv2
import glob
from .ShockOscillationAnalysis import BCOLOR

def bg_manipulation(path:str, y_crop:tuple[int]=None, x_crop:tuple[int]=None,
                    resize:tuple[int]=None, bg_rotate=0, n: int=-1) -> tuple:
    """
    Perform background image manipulation including cropping, resizing, and rotation.

    This function processes a set of background images by applying optional cropping, resizing, and rotation. 
    The first image in the specified path is used for manipulation.

    Parameters:
        - **path (str)**: File path pattern to locate background image files. Supports wildcards.
        - **y_crop (tuple[int], optional)**: A tuple (y_min, y_max) defining the vertical cropping range. 
          Defaults to the full height of the image.
        - **x_crop (tuple[int], optional)**: A tuple (x_min, x_max) defining the horizontal cropping range. 
          Defaults to the full width of the image.
        - **resize (tuple[int], optional)**: A tuple (width, height) defining the new dimensions for resizing. 
          Defaults to the dimensions after cropping.
        - **bg_rotate (bool, optional)**: Whether to rotate the image 90 degrees clockwise. Defaults to `0` (no rotation).
        - **n (int, optional)**: Number of images expected in the specified path. If `-1`, no specific count is enforced. 
          Defaults to `-1`.

    Returns:
        - tuple:
            - np.ndarray: The processed background image.
            - int: Number of background images found in the specified path.

    .. note ::
        - If the `path` does not contain any files, the function will print an error and return `None`.
        - If fewer files than expected are found, a warning message is displayed.

    Steps:
        1. Verify the availability of files at the given path.
        2. Load the first image from the sorted list of files.
        3. Apply cropping based on the provided `x_crop` and `y_crop` parameters.
        4. Resize the cropped image to the specified `resize` dimensions.
        5. Optionally rotate the image by 90 degrees clockwise.

    Example:
        >>> bg_img, n_images = bg_manipulation("path/to/images/*.png", 
                                               y_crop=(50, 200), 
                                               x_crop=(30, 150), 
                                               resize=(100, 100), 
                                               bg_rotate=True)
"""
    bg_files = sorted(glob.glob(path))
    n_images = len(bg_files)
    if len(bg_files) < 1:
        error = f'Files found are {len(bg_files)}. No files found!;'
        print(f'{BCOLOR.FAIL}Error:{BCOLOR.ENDC}', end= ' ')
        print(f'{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
        return None

    if n > -1 and len(bg_files) < n:
        warning = f'Files found are {len(bg_files)}. Files are less than expected!;'
        action = 'Only the first image will be considered for visualization'
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
        print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')

    bg_img = cv2.imread(bg_files[0])
    bg_shp = bg_img.shape
    bg_y_crop = (0, bg_shp[0]) if y_crop == None else y_crop
    bg_x_crop = (0, bg_shp[1]) if x_crop == None else x_crop
    cropped_shp = (bg_x_crop[1]-bg_x_crop[0], bg_y_crop[1]-bg_y_crop[0])
    bg_resize = cropped_shp if resize is None else resize
    bg_img = bg_img[bg_y_crop[0]: bg_y_crop[1],
                    bg_x_crop[0]: bg_x_crop[1], :]
    bg_img = cv2.resize(bg_img, bg_resize)
    if bg_rotate:
        bg_img = cv2.transpose(bg_img)  
        bg_img = cv2.flip(bg_img, 1)
    else:
        bg_img
    return bg_img, n_images