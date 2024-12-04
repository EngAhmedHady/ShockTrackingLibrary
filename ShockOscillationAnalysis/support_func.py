import cv2
import glob
from .ShockOscillationAnalysis import BCOLOR

def bg_manipulation(path:str, y_crop:tuple[int]=None, x_crop:tuple[int]=None,
                    resize:tuple[int]=None, bg_rotate=0, n: int=-1):
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