# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:45:35 2023

@author: Ahmed H. Hanfy
"""
import sys
import cv2
import glob
import keyboard
import numpy as np
from datetime import datetime as dt
from ..preview import PreviewCVPlots
from ..constants import BCOLOR, CVColor
from ..ShockOscillationAnalysis import SOA
from ..linedrawingfunctions import InclinedLine
from ..support_func import bg_manipulation, log_message
from ..inc_tracking.inc_tracking import InclinedShockTracking
from .list_generation_tools import (genratingRandomNumberList,
                                    GenerateIndicesList)


class SliceListGenerator(SOA):
    def __init__(self, f: int, D: float = 1, pixelScale: float = 1):
        self.inc_trac = InclinedShockTracking(f, D)
        super().__init__(f, D, pixelScale)

    def IntersectionPoint(self, M: list[float], A: list[float],
                          Ref: list[tuple, tuple], log_dirc:str='') -> tuple[tuple[int, int]]:
        """
        Calculate the intersection point between two lines.

        Parameters:
            - **M (list)**: List containing slopes of the two lines.
            - **A (list)**: List containing y-intercepts of the two lines.
            - **Ref (list)**: List containing reference points for each line.
            - **log_dirc (str)**: log file directory.

        Returns:
            tuple:
                - A tuple containing: Pint (tuple): Intersection point coordinates (x, y).

        Example:
            >>> from __importImages import importSchlierenImages
            >>> instance = importSchlierenImages(f)
            >>> slopes = [0.5, -2]
            >>> intercepts = [2, 5]
            >>> references = [(0, 2), (0, 5)]
            >>> intersection, angles = instance.IntersectionPoint(slopes, intercepts, references)
            >>> print(intersection, angles)

        .. note ::
            - The function calculates the intersection point and angles between two lines specified by their slopes and y-intercepts.
            - Returns the intersection point coordinates and angles of the lines in degrees.
        """
        theta1 = np.rad2deg(np.arctan(M[0]))
        theta2 = np.rad2deg(np.arctan(M[1]))

        Xint, Yint = None, None

        if theta1 != 0 and theta2 != 0 and theta1 - theta2 != 0:
            Xint = (A[1] - A[0]) / (M[0] - M[1])
            Yint = M[0] * Xint + A[0]
        elif theta1 == 0 and theta2 != 0:
            Yint = Ref[0][1]
            Xint = (Yint - A[1]) / M[1]
        elif theta2 == 0 and theta1 != 0:
            Xint = Ref[1][0]
            Yint = M[0] * Xint + A[0]
        else:
            warning = 'Lines are parallel!;'
            action = ''
            log_message(f'Warning: {warning}', log_dirc)
            print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
            print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')

        Pint = (round(Xint), round(Yint))
        return Pint

    def ImportingFiles(self, pathlist: list[str], indices_list: list[int],
                       n_images: int, imgs_shp: tuple[int], x_range: tuple[int],
                       tk: tuple[int], M: np.ndarray[float]) -> tuple[np.ndarray, int]:
        """
        Import images from specified paths, and return a concatenated image list.

        Parameters:
            - **pathlist (list)**: List of paths to image files.
            - **indices_list (list)**: List of indices specifying which images to import from `pathlist`.
            - **n_images (int)**: Total number of images to import.
            - **imgs_shp (tuple)**: Tuple specifying the shape of the images to be resized to (height, width).
            - **x_range (tuple)**: Tuple specifying the range of x-values to crop from the images (start, end).
            - **tk (tuple)**: Tuple specifying the range of y-values to crop from the images (start, end).
            - **M (numpy.ndarray)**: 2x3 transformation matrix for image rotation.

        Returns:
            - numpy.ndarray: Concatenated image list.
            - int: Number of imported images

        .. note ::
            - Requires the OpenCV (cv2) and NumPy libraries.
            - Assumes the input images are RGB.
        """
        img_list=[] # List to store processed images
        slice_thickness =  tk[1]-tk[0]  # Calculate slice thickness from `tk`

        # Loop through indices to import and process images
        for n, i in enumerate(indices_list):
            # Read image from specified path
            img = cv2.imread(pathlist[i]) 
            # Rotate the image with M matrix
            img = cv2.warpAffine(img, M, (imgs_shp[1],imgs_shp[0]))
            # cropped image to the region of interest
            cropped_image = np.zeros([1,x_range[1]-x_range[0],3])

            # Average the cropped image to creeat one slice
            for j in range(tk[0],tk[1]):
                cropped_image += img[j-1 : j, x_range[0]: x_range[1]]
            cropped_image /= slice_thickness
            img_list.append(cropped_image.astype('float32'))

            # Increment counter and display progress
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n_images/20)), int(5*n/(n_images/20))))
        print('')

        # Concatenate the list of processed images vertically
        img_list = cv2.vconcat(img_list)
        return img_list, n

    def GenerateSlicesArray(self, path:str, scale_pixels:bool=True , full_img_width:bool=False,   # Domain info.
                            slice_loc:int=0, slice_thickness:int|list[int, str]=0,                # Slice properties
                            shock_angle_samples=30, inclination_est_info:list[int,tuple,tuple]=[], # Angle estimation
                            preview:bool=True, angle_samples_review = 10,                            # preview options
                            output_directory:str='', comment:str='',                               # Data store
                            **kwargs) -> tuple[np.ndarray[int], int, dict, float]:                      # Other
        """
        Generate a sequence of image slices for single horizontal line shock wave analysis.
        This function imports a sequence of images to perform an optimized analysis by extracting
        a single pixel slice from each image as defined by the user, appending them together, and
        generating a single image where each row represents a snapshot.

        Parameters:
            - **path (str)**: Directory path containing the sequence of image files.
            - **scale_pixels (bool)**: Whether to scale the pixels. Default is True.
            - **full_img_width (bool)**: Whether to use the full image width for slicing. Default is False.
            - **slice_loc (int)**: Location of the slice.
            - **slice_thickness (int)**: Thickness of the slice.
            - **shock_angle_samples (int)**: Number of samples to use for shock angle estimation. Default is 30.
            - **inclination_est_info (list[int, tuple, tuple])**: Information for inclination estimation. Default is an empty list.
            - **preview (bool)**: Whether to display a preview of the investigation domain before rotating. Default is True.
            - **angle_samples_review (int)**: Number of samples to review for angle estimation. Default is 10.
            - **output_directory (str)**: Directory to store the output images. Default is an empty string.
            - **comment (str)**: Comment to include in the output filename. Default is an empty string.
            - `**kwargs`: Additional arguments for fine-tuning/Automate the function.

        Returns:
            - tuple:
                - numpy.ndarray: Concatenated image slices.
                - int: Number of images imported.
                - dict: Working range details.
                - float: Pixel scale.

        .. note ::
            - The function assumes the input images are in RGB format.
            - The `kwargs` parameter can include:
                - **Ref_x0 (list[int, int])**: Reference x boundaries.for scaling
                - **Ref_y0 (int)**: Reference y datum (zero y location)
                - **Ref_y1 (int)**: slice location (The scanning line, y-center of rotation)
                - **avg_shock_angle (float)**: Average shock angle. (if known, to skip average shock inc check)
                - **avg_shock_loc (int)**: Average shock location. (if known, x-center of rotation)
                - **sat_vr (int | list[int,'str'])**: Shock Angle Test Vertical Range 
                  (If not provided the vertical renge will be equal to the ``slice_thickness``, could be
                  provided as number then value will be added equally to upper and lower the traking
                  location in pixels, it also can be added as list `[upper bound, lower bound, unit(optional)]` 
                  in current version unit should match the universal units ``'px'``, etc. if the unit not 
                  provided the defualt ``'px'`` will be considered)
                - **n_files (int)**: Number of files to import
                - **within_range (tuple[int, int])**: Range of files to import `(start, end)`
                - **every_n_files (int)**: Step for file import.
                - :func:`Inclind angle tracking parameters <ShockOscillationAnalysis.inc_tracking.inc_tracking.InclinedShockTracking.InclinedShockTracking>`:
                    - **Confidance**: ``nPnts``, ``conf_interval``, ``residual_preview``
                    - **Preview**: ``avg_preview_mode``, ``points_opacity``, ``points_size``, ``avg_lin_color``
                    - **Output Background**: ``op_bg_path``, ``bg_x_crop``, ``bg_y_crop``, ``bg_90rotate``, ``bg_resize``

        Steps:
            1. Define reference vertical boundaries (for scaling).
            2. Define reference horizontal line (slice shifted by HLP from reference).
            3. Optionally define the estimated line of shock.
            4. Run shock tracking function within the selected slice to define the shock angle (if step 3 is valid).
            5. Generate shock rotating matrix (if step 3 is valid).
            6. Import files, slice them, and store the generated slices list into an image.

        Example:
            >>> img_list, n, working_range, pixel_scale = GenerateSlicesArray(r'/path/to/*.ext', 
                                                                              slice_loc=10, 
                                                                              slice_thickness=5)
        """
        important_info = kwargs.get('important_info', 0)
        if important_info:
            print(f'{BCOLOR.UNDERLINE}Notes:{BCOLOR.ENDC}{BCOLOR.ITALIC}')
            draw_lin = '`one to define the line and one to confirm`'
            print(f'\t- Draw a line requires 2 left clicks {draw_lin}')
            print('\t- To delete a line press right click instead of second left click')
            print('\t- Do not forget to press any key except `Esc` to close the image window')
            print(f'\t- To terminating process press `Esc`{BCOLOR.ENDC}')
        
        inclinationCheck = False
        avg_angle = 90
        dis_unit = self.univ_unit["dis"]
        self.outputPath = output_directory
        # Find all files in the directory with the sequence and sorth them by name
        resize_img = kwargs.get('resize_img', None)
        crop_y_img = kwargs.get('crop_y_img', None)
        crop_x_img = kwargs.get('crop_x_img', None)
        bg, n1 = bg_manipulation(path, crop_y_img, crop_x_img, resize_img,
                                 log_dirc=output_directory)
        # In case no file found end the progress and eleminate the program
        if n1 < 1: sys.exit()
        op_bg_path = kwargs.get('op_bg_path', None)
        if op_bg_path is not None:
            bg_y_crop = kwargs.get('bg_y_crop', None)
            bg_x_crop = kwargs.get('bg_x_crop', None)
            bg_resize = kwargs.get('bg_resize', None)
            bg_90rotate = kwargs.get('bg_90rotate', 0)
            op_bg, n_bg = bg_manipulation(op_bg_path, bg_y_crop, bg_x_crop, 
                                          bg_resize, bg_90rotate, n=n1, log_dirc=output_directory)
            if n_bg < 1:
                action = 'Original file set well be used'
                log_message(action, output_directory)
                print(f'{BCOLOR.ITALIC}{action}{BCOLOR.ENDC}')

        # Open first file and set the limits and scale
        img = op_bg if op_bg_path is not None else bg

        shp = img.shape
        new_log = f'Img Shape is: {shp}'
        log_message(new_log, output_directory)
        print(new_log)
        Ref_x0 = kwargs.get('Ref_x0', [0,0])
        Ref_y0 = kwargs.get('Ref_y0', -1)
        Ref_y1 = kwargs.get('Ref_y1', -1)
        if scale_pixels:
            Ref_x0, Ref_y0, Ref_y1 = self.DefineReferences(img, shp,
                                                           Ref_x0, scale_pixels,
                                                           Ref_y0, Ref_y1, slice_loc)
        else: 
            self.clone = img.copy()
            self.Reference=[Ref_x0, Ref_y0]
            try:
                Ref_y1 = self.LineDraw(self.clone, 'H', 2, line_color=CVColor.ORANGE)[-1]
            except Exception:
                warning = 'Nothing was drawn!;'
                action = 'Ref_y1 value is {Ref_y1}'
                log_message(f'Warning: {warning}', output_directory)
                log_message(action, output_directory)
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
            
        new_log='Slice center is located at:'
        log_message(new_log, output_directory)
        print(new_log)
        
        new_log=f'\t- {Ref_y1}px in absolute reference'
        log_message(new_log, output_directory)
        print(new_log)
        
        if scale_pixels:
            dis_in_unit = f'{abs(Ref_y1-Ref_y0)*self.pixelScale:0.2f}{dis_unit}'
            new_log = f'\t- {dis_in_unit} ({abs(Ref_y1-Ref_y0)}px) from reference `Ref_y0`'
            log_message(new_log, output_directory)
            print(new_log)

        if Ref_y1 > 0 and Ref_y1 != Ref_y0: 
            cv2.line(self.clone, (0, Ref_y1), (shp[1], Ref_y1), CVColor.RED, 1)

        if hasattr(slice_thickness, "__len__"):        
            if slice_thickness[1] == dis_unit and self.pixelScale > 0: 
                slice_thickness = int(slice_thickness[0]/self.pixelScale)
            elif slice_thickness[1] == 'px':
                slice_thickness = int(slice_thickness[0])
            else: 
                error = 'Insufficient scale/unit!'
                log_message(f'Error: {error}', output_directory)
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
                sys.exit()
        if slice_thickness > 0: Ht = int(slice_thickness/2)  # Half Thickness
        else: Ht = 1

        upper_bounds =  Ref_y1 - Ht
        lower_bounds =  Ref_y1 + Ht if slice_thickness%2 == 0 else  Ref_y1 + Ht + 1
        cv2.line(self.clone, (0,lower_bounds), (shp[1],lower_bounds), CVColor.ORANGE, 1)
        cv2.line(self.clone, (0,upper_bounds), (shp[1],upper_bounds), CVColor.ORANGE, 1)

        avg_shock_angle = kwargs.get('avg_shock_angle', 90)
        avg_shock_loc = kwargs.get('avg_shock_loc', [0, 0])
        sat_vr = kwargs.get('sat_vr', slice_thickness) # Shock Angle Test Vertical Range
        if hasattr(sat_vr, "__len__"):
            start_vr, end_vr = sat_vr[:2]
            if len(sat_vr) > 2 and sat_vr[2] ==  dis_unit:
               sat_vr = [round(Ref_y1 - (start_vr/self.pixelScale)), 
                         round(Ref_y1 - (end_vr/self.pixelScale))]
            else:
                sat_vr = [round(Ref_y1 - start_vr), round(Ref_y1 - end_vr)]
        elif not hasattr(sat_vr, "__len__"):
              Ht2 = int(sat_vr/2)
              sat_vr = [Ref_y1 - Ht2, Ref_y1 + Ht2]
              start_vr, end_vr = sat_vr[:2]
        
        sat_vr.sort()
        if abs(end_vr-start_vr) == 0:
            sat_vr = [upper_bounds, lower_bounds]

        new_log='Shock angle tracking vertical range above the reference `Ref_y0` is:'
        log_message(new_log, output_directory)
        print(new_log)
        
        v1, v2 = [f'{v:0.2f}{dis_unit}' for v in (Ref_y0-np.array(sat_vr))*self.pixelScale]
        if scale_pixels:
            new_log=f'\t- In ({dis_unit})s from {v1} to {v2}'
            log_message(new_log, output_directory)
            print(new_log)
            
        new_log=f'\t- In pixels from {Ref_y0-sat_vr[0]:0}px to {Ref_y0-sat_vr[1]:0}px'
        log_message(new_log, output_directory)
        print(new_log)
        
        nPnts = kwargs.get('nPnts', 0)
        Inc_shock_setup = self.inc_trac.InclinedShockDomainSetup
        if not hasattr(inclination_est_info, "__len__"):
            self.LineDraw(self.clone, 'Inc', 3)
            if len(self.Reference) < 4:
                error = 'Reference lines are not sufficient!'
                log_message(f'Error: {error}', output_directory)
                print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
                sys.exit()
            P1,P2,m,a = self.Reference[3]
            Ref, nSlices, inclinationCheck = Inc_shock_setup(inclination_est_info,
                                                             sat_vr, [P1, P2, m, a],
                                                             shp, VMidPnt=Ref_y1,
                                                             preview_img=self.clone,
                                                             nPnts=nPnts,
                                                             log_dirc=output_directory)
        elif len(inclination_est_info) > 2:
            P1, P2, m, a = InclinedLine(inclination_est_info[1], inclination_est_info[2],
                                        imgShape=shp,log_dirc=output_directory)
            cv2.line(self.clone, P1, P2, CVColor.GREEN, 1)
            self.Reference.append([P1, P2, m, a])
            Ref, nSlices, inclinationCheck = Inc_shock_setup(inclination_est_info[0],
                                                             sat_vr, [P1, P2, m, a],
                                                             shp, VMidPnt=Ref_y1,
                                                             preview_img=self.clone,
                                                             nPnts=nPnts, 
                                                             log_dirc=output_directory)
        # in case the rotation angle only is provieded in working _range
        elif avg_shock_angle != 90 and avg_shock_loc == [0, 0]:
            request = 'Please, provide the rotation center...'
            log_message(f'Request: {request}', output_directory)
            print(f'{BCOLOR.BGOKGREEN}Request:{BCOLOR.ENDC}', end=' ')    
            print(f'{BCOLOR.ITALIC}{request}{BCOLOR.ENDC}')
            self.LineDraw(self.clone, 'Inc', 3)
            # find the rotation center
            avg_shock_loc = self.IntersectionPoint([0, self.Reference[-1][2]],
                                                   [Ref_y1, self.Reference[-1][3]],
                                                   [(0, Ref_y1), self.Reference[-1][0]], 
                                                   log_dirc=output_directory)

        if preview:
            cv2.imshow('Investigation domain before rotating', self.clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if keyboard.read_key() == "esc":
                new_log = 'Operation not permitted. Terminating process ...'
                log_message(new_log, output_directory)
                print(new_log)
                sys.exit()
            cv2.waitKey(1)

        # number of files to be imported
        files = sorted(glob.glob(path))
        import_n_files = kwargs.get('n_files', 0)
        if import_n_files == 0:
            import_n_files = kwargs.get('within_range', [0, 0])
        import_step = kwargs.get('every_n_files', 1)
        indices_list, n_images = GenerateIndicesList(n1, import_n_files, 
                                                     import_step, log_dirc=output_directory)

        if inclinationCheck:
            randomIndx = genratingRandomNumberList(shock_angle_samples, n1, 
                                                   log_dirc=output_directory)

            samplesList = {}
            k = 0
            new_log = f'Import {len(randomIndx)} images for inclination Check ... '
            log_message(new_log, output_directory)
            print(new_log)
            for indx in randomIndx:
                Sample = cv2.imread(files[indx])
                # check if the image on grayscale or not and convert if not
                if len(Sample.shape) > 2: Sample = cv2.cvtColor(Sample, cv2.COLOR_BGR2GRAY)
                samplesList[indx] = Sample
                k += 1
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int(k/(shock_angle_samples/20)),
                                                       int(5*k/(shock_angle_samples/20))))
            print('')

            if angle_samples_review < shock_angle_samples: NSamplingReview = angle_samples_review
            else:
                NSamplingReview = shock_angle_samples
                warning = 'Number of samples is larger than requested to review!;'
                action = 'All samples will be reviewed'
                log_message(f'Warning: {warning}', output_directory)
                log_message(action, output_directory)
                print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
                print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
            
            new_log = 'Shock inclination estimation ... '
            log_message(new_log, output_directory)
            print(new_log)
            inc_track = self.inc_trac.InclinedShockTracking
            avg_shock_angle, avg_shock_loc = inc_track(samplesList, nSlices, Ref, 
                                                       nReview=NSamplingReview,
                                                       output_dirc=output_directory, 
                                                       comment=comment,**kwargs)

            avg_angle = avg_shock_angle[0] if avg_shock_angle[2] > 0 else avg_shock_angle[0]
        M = cv2.getRotationMatrix2D((avg_shock_loc[0], Ref_y1), 90-avg_angle, 1.0)
        new_img = cv2.warpAffine(img, M, (shp[1], shp[0]))

        new_img = PreviewCVPlots(new_img, Ref_x0, Ref_y=Ref_y1,
                                 tk=[lower_bounds, upper_bounds],
                                 avg_shock_loc=avg_shock_loc[0])

        if avg_angle != 90 and preview:
            cv2.imshow('Final investigation domain', new_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if keyboard.read_key() == "esc":
                new_log = 'Operation not permitted. Terminating process ...'
                log_message(new_log, output_directory)
                print(new_log)
                sys.exit()
            cv2.waitKey(1)

        if len(output_directory) > 0:
            
            file_info = f'{self.f/1000:.1f}kHz_'             # Sampling rate info
            file_info += f'{slice_loc}{dis_unit}_'           # Slice location from `Ref_y0`
            file_info += f'{self.pixelScale}{dis_unit}-px_'  # Image scale
            file_info += f'tk_{slice_thickness}px'           # Slice generated from avg. n-pixels 
            if len(comment) > 0:
                outputPath = fr'{output_directory}\{file_info}_{comment}'
            else:
                now = dt.now()
                now = now.strftime("%d%m%Y%H%M")
                outputPath =f'{output_directory}\\{file_info}_{now}'
            if avg_angle != 90:
                error = f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}Failed!{BCOLOR.ENDC}'
                if cv2.imwrite(f'{outputPath}-RefD{avg_angle:0.2f}deg.png', new_img):
                    txt = u"stored \u2713"
                else: txt = error
                log_message(f'RotatedImage: {txt}', output_directory)
                print('RotatedImage:', txt)
                
            if cv2.imwrite(f'{outputPath}-RefD.png', self.clone):
                txt = u"stored \u2713"
            else: txt = error
            log_message(f'DomainImage: {txt}', output_directory)
            print('DomainImage:', txt)

        if full_img_width:
            x_range = [0, shp[1]]
            working_range = {'Ref_x0': [0, shp[1]], 'Ref_y1': Ref_y1,
                             'avg_shock_angle': avg_shock_angle,
                             'avg_shock_loc': avg_shock_loc}
            new_log = f'scaling lines: Ref_x0 = {Ref_x0}, Ref_y1 = {Ref_y1}'
            log_message(new_log, output_directory)
            print(new_log)

        else:
            x_range = Ref_x0
            working_range = {'Ref_x0': Ref_x0, 'Ref_y1': Ref_y1,
                             'avg_shock_angle': avg_shock_angle,
                             'avg_shock_loc': avg_shock_loc}

        new_log = f'working range is: {working_range}'
        log_message(new_log, output_directory)
        print(new_log)
        
        new_log = f'Importing {n_images} images ...'
        log_message(new_log, output_directory)
        print(new_log)

        img_list, n = self.ImportingFiles(files, indices_list, n_images, shp,
                                          x_range, [upper_bounds, lower_bounds], M)

        if len(output_directory) > 0:
            if cv2.imwrite(f'{outputPath}.png', img_list):
                txt = f"Image list was stored at: {outputPath}.png"
            else: txt = error
            log_message(f'ImageList write: {txt}', output_directory)
            print('ImageList write:', txt)

        return img_list, n, working_range, self.pixelScale
