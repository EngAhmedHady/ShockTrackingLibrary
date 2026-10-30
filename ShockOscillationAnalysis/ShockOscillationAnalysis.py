# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:32:30 2022

@author: Ahmed H. Hanfy
"""
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from .support_func import log_message
from .constants import CVColor, BCOLOR
from .imgcleaningfunctions import (SliceListAverage,
                                   CleanIlluminationEffects,
                                   BrightnessAndContrast)
from .linedrawingfunctions import InclinedLine
from .generateshocksignal import GenerateShockSignal

px = 1/plt.rcParams['figure.dpi']
plt.rcParams.update({'font.size': 16})
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['figure.max_open_warning'] = 0


class SOA:
    def __init__(self, f: int = 1, D: int = 1, pixelScale: float=1):
        self.f = f  # ----------------- sampling rate (univ_unit: 'freq')
        self.D = D  # ----------------- refrence distance (univ_unit: 'dis)
        self.pixelScale = pixelScale  # initialize scale of the pixels
        # - universal units used in the analysis
        self.univ_unit = {'freq':'fps','dis':'mm', 'angle': 'deg'}
        self.ClickCount = 0  # -------- initialize the mouse clicks
        self.TempLine = []  # --------- initialize the temporary line array
        self.Temp = cv2.vconcat([])  #- initialize the temporary image
        self.clone = cv2.vconcat([])  # initialize the editable image copy
        self.Reference = []  # -------- initialize croping limits or line set
        self.line_coordinates = []  # - initialize Line coordinate
        self.outputPath = ''  # ------- image output
        self.log = log_message

    def extract_coordinates(self, event: int,  # call event
                            x: int, y: int, flags: int,  # mouse current status
                            parameters: tuple[str, tuple[int], str]) -> None:
        """
        Record starting (x, y) coordinates on left mouse button click and draw
        a line that crosses all over the image, storing it in a global
        variable. In case of horizontal or vertical lines, it takes the average
        between points.

        Drawing steps:
            1. Push the left mouse on the first point.
            2. Pull the mouse cursor to the second point.
            3. The software will draw a thick red line (indicating the mouse
               locations) and a green line indicating the Final line result.
            4. To confirm, press the left click anywhere on the image, or
               to delete the line, press the right click anywhere on the image.
            5. Press any key to proceed.

        Parameters:
            - event (int): The type of event (e.g., cv2.EVENT_LBUTTONDOWN).
            - x (int): The x-coordinate of the mouse cursor.
            - y (int): The y-coordinate of the mouse cursor.
            - flags (int): Flags associated with the mouse event.
            - parameters (tuple): A tuple containing:
                - Name of the window to display the image.
                - Image shape (tuple of y-length and x-length).
                - Line type ('V' for vertical, 'H' for horizontal,
                             'Inc' for inclined).

        Returns:
            None

        Example:
            >>> instance = SOA()
            >>> cv2.setMouseCallback(window_name, instance.extract_coordinates, parameters)

        .. note::
            - If 'Inc' is provided as the line type, it uses the 'InclinedLine' method
              to calculate the inclined line and display it on the image.

        """

        window_name, img_shape, line_type, line_color, log_dirc = parameters
        avg = None

        # Unconfirmed line drawing functions
        def draw_line(start, end, color, thickness=1):
            # Draws a line on the image.
            cv2.line(self.Temp, start, end, color, thickness)


        def process_line():
            # Processes the drawn line based on the line type.
            if line_type == 'V':
                avg = (self.TempLine[0][0] + self.TempLine[1][0]) // 2
                draw_line((avg, 0), (avg, img_shape[0]), line_color)
            elif line_type == 'H':
                avg = (self.TempLine[0][1] + self.TempLine[1][1]) // 2
                draw_line((0, avg), (img_shape[1], avg), line_color)
            elif line_type == 'Inc':
                avg = InclinedLine(self.TempLine[0], self.TempLine[1], imgShape=img_shape)
                draw_line(avg[0], avg[1], line_color)
            return avg

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ClickCount += 1
            if len(self.TempLine) == 2:
                self.line_coordinates = self.TempLine
            elif len(self.TempLine) == 0:
                self.TempLine = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.TempLine) < 2:
                self.TempLine.append((x,y))
                # Draw temprary line
                draw_line(self.TempLine[0], self.TempLine[1], (0, 0, 255), 2)
                avg = process_line()
                if avg is not None:
                    cv2.imshow(window_name, self.Temp)

            elif self.ClickCount == 2:

                self.Temp = self.clone.copy()
                avg = process_line()
                if avg:
                    self.Reference.append(avg)
                self.clone = self.Temp.copy()
                cv2.imshow(window_name, self.clone)
                new_log = f'Registered line: {avg}'
                log_message(new_log, self.outputPath)
                print(new_log)
                

        # Delete draw line before storing
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.TempLine = []
            if self.ClickCount>0:
                self.ClickCount -= 1
            self.Temp = self.clone.copy()
            cv2.imshow(parameters[0], self.Temp)

    def LineDraw(self, img: np.ndarray[int],              # BG image
                 lineType: str, LineNameInd: int,         # Line info.
                 Intialize=False, **kwargs) -> list:     # Other parameters
        """
        Drive the extract_coordinates function to draw lines.

        Parameters:
            - **img (numpy.ndarray)**: A single OpenCV image.
            - **lineType (str)**:
                - 'V' for Vertical line (starts from top to bottom of the image),
                - 'H' for Horizontal line (starts from the left to the right),
                - 'Inc' for Inclined line (not averaging, takes the exact selected points).
            - **LineNameInd (int)**: Index of the window title from the list.
            - **Initialize (bool, optional)**: To reset the values of Reference and
              line_coordinates for a new line set. True or False (Default: False).

        Returns:
            list: Cropping limits or (line set).

        Example:
            >>> instance = SOA()
            >>> line_set = instance.LineDraw(image, 'V', 0, Initialize=True)
            >>> print(line_set)

        .. note::
            - The function uses the `extract_coordinates` method to interactively draw lines on the
              image.
            - It waits until the user presses a key to close the drawing window.

        .. note::
           ``LineNameInd`` is the index number refering to one of these values as window title:

            0. "First Reference Line (left)",
            1. "Second Reference Line (right)",
            2. "Horizontal Reference Line",
            3. "estimated shock location"

        """

        self.clone = img.copy()
        self.Temp = self.clone.copy()
        self.TempLine = []
        self.ClickCount = 0
        # Window titles
        WindowHeader = ["First Reference Line (left)",
                        "Second Reference Line (right)",
                        "Horizontal Reference Line",
                        "estimated shock location"]
        if Intialize:
            self.Reference = []
            self.line_coordinates = []
        shp = img.shape
        # win_x, win_y = self.screenMidLoc(shp)
        prams = ['new cv window', shp, lineType]
        if   lineType == 'V':
            v_draw_color = kwargs.get('v_draw_color', CVColor.GREEN)
            prams[0] = WindowHeader[LineNameInd]
            prams.append(v_draw_color)
        elif lineType == 'H':
            h_draw_color = kwargs.get('h_draw_color', CVColor.YELLOW)
            prams[0] = WindowHeader[LineNameInd]
            prams.append(h_draw_color)
        elif lineType == 'Inc':
            inc_draw_color = kwargs.get('inc_draw_color', CVColor.BLUE)
            prams[0] = WindowHeader[LineNameInd]
            prams.append(inc_draw_color)
            
        prams.append(self.outputPath)

        cv2.imshow(WindowHeader[LineNameInd], self.clone)
        cv2.setMouseCallback(WindowHeader[LineNameInd],
                             self.extract_coordinates,prams)
        # Wait until user press some key
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return self.Reference

    def n_linecheck(self, line_list: list, min_n: int):
        """
        Checks if the provided list of reference lines meets the minimum required count.

        If the number of lines is less than `min_n`, this method logs the error,
        prints a formatted failure message to the console, and exits the program.

        Parameters:
            - **line_list (list[int])**: A list of reference lines to check.
            - **min_n (int):** The minimum required number of reference lines.

        Raises:
            - **SystemExit**: Exits the program if the line count is insufficient.
        """
        if len(line_list) < min_n:
            error = 'Reference lines are not sufficient!'
            log_message(error, self.outputPath)
            print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}{error}{BCOLOR.ENDC}')
            sys.exit()
            
    def DefineReferences(self, img: np.ndarray[int], shp:tuple[int],
                         Ref_x0:list[int], Ref_y00:list[int], scale_pixels:bool,
                         Ref_y0:int=-1, Ref_y1:int=-1,
                         slice_loc:int=0) -> tuple[list[int], int, int, list[int]]:
        """
        Define reference lines on an image for scalling and further processing.

        Parameters:
            - **img (np.ndarray)**: The image on which to draw the reference lines.
            - **shp (tuple)**: Shape of the image (height, width).
            - **Ref_x0 (list[int])**: List of x-coordinates for vertical reference lines.
            - **scale_pixels (bool)**: Whether to scale pixels based on the reference lines.
            - **Ref_y0 (int, optional)**: y-coordinate of the top horizontal reference line. 
              Default is -1.
            - **Ref_y00 (list[int])**: List of y-coordinates for horizontal reference lines used for scaling.
            - **Ref_y1 (int, optional)**: y-coordinate of the bottom horizontal reference line. 
              Default is -1.
            - **slice_loc (int, optional)**: Location of the slice for horizontal reference lines. 
              Default is 0.

        Returns:
            - tuple: A tuple containing:
                - Ref_x0 (list[int]): Sorted list of x-coordinates for vertical reference lines.
                - Ref_y0 (int): y-coordinate of the top horizontal reference line.
                - Ref_y1 (int): y-coordinate of the bottom horizontal reference line.
                - Ref_y00 (list[int]): Sorted list of horizontal scaling reference lines.

        Example:
            >>> instance = SOA()
            >>> img = cv2.imread('path/to/image.jpg')
            >>> shape = img.shape
            >>> Ref_x0 = [100, 200]
            >>> Ref_y00 = [50, 150]
            >>> scale_pixels = True
            >>> Ref_y0 = -1
            >>> Ref_y1 = -1
            >>> slice_loc = 50
            >>> ref_x0, ref_y0, ref_y1, ref_y00 = instance.DefineReferences(img, shape, Ref_x0,
                                                                            scale_pixels, Ref_y0,
                                                                            Ref_y1, slice_loc)
            >>> print(ref_x0, ref_y0, ref_y1, ref_y00)

        .. note::
            - The function sets up vertical and horizontal reference lines on the image.
            - It calculates the pixel scale if `scale_pixels` is True using the distance 
              between reference lines (vertical or horizontal depending on user prompts). 
        """

        # Ensure the vertical reference lines are sorted if they are provided
        Ref_x0.sort()
        start, end = Ref_x0
        x0_diff = abs(end-start)
        draw_x0 = x0_diff == 0
        
        Ref_y00.sort()
        start, end = Ref_y00
        y00_diff = abs(end-start)
        draw_y00 = y00_diff == 0
        ans1 = True
        if draw_y00:
            ans1 = input("Use vertical lines for scaling? Type Y if yes:").strip()
            ans1 = True if ans1.upper() == 'Y' else False
            if ans1: 
                self.LineDraw(img, 'H', 0, Intialize=True, h_draw_color=CVColor.MAGENTA)
                self.LineDraw(self.clone, 'H', 1)
                Ref_y00 = self.Reference
                self.n_linecheck(Ref_y00, 2)
                Ref_y00.sort()  # to make sure that the limits are properly assigned
        else:
            # set the horizontal scaling reference lines if provided
            self.clone = img.copy()
            cv2.line(self.clone, (0, Ref_y00[0]), (shp[1], Ref_y00[0]), CVColor.MAGENTA, 1)
            cv2.line(self.clone, (0, Ref_y00[1]), (shp[1], Ref_y00[1]), CVColor.MAGENTA, 1)
            self.Reference = Ref_y00[0: 2].copy()

        if draw_x0:
            # Draw vertical reference lines if not provided
            if ans1: 
                ans2 = input("Still need vertical line? Type Y if yes:").strip()
                ans2 = True if ans2.upper() == 'Y' else False
                if ans2:
                    self.LineDraw(img, 'V', 0, Intialize=True)
                    self.LineDraw(self.clone, 'V', 1)
                    Ref_x0 = self.Reference
                    self.n_linecheck(Ref_x0, 2)
                    Ref_x0.sort()  # to make sure that the limits are properly assigned
        else:
            # set the vertical reference lines if provided
            self.clone = img.copy()
            cv2.line(self.clone, (Ref_x0[0], 0), (Ref_x0[0], shp[0]),
                     CVColor.GREEN, 1)
            cv2.line(self.clone, (Ref_x0[1], 0), (Ref_x0[1], shp[0]),
                     CVColor.GREEN, 1)
            self.Reference = Ref_x0[0: 2].copy()

        # Calculate the pixel scale if required
        if scale_pixels and not ans1:
            self.pixelScale = self.D / abs(Ref_x0[1]-Ref_x0[0])
        elif scale_pixels and ans1:
            self.pixelScale = self.D / abs(Ref_y00[1]-Ref_y00[0])
        new_log = f'Image scale: {self.pixelScale} {self.univ_unit["dis"]}/px'
        log_message(new_log, self.outputPath)
        print(new_log)

        # --------------------------------------------------------------------

        # Alocate Horizontal reference lines
        if Ref_y0 == -1 and Ref_y1 == -1:
            self.LineDraw(self.clone, 'H', 2)  # to draw the reference line
            self.n_linecheck(self.Reference, 3)
            Ref_y0 = self.Reference[-1]
            Ref_y1 = self.Reference[-1]-round(slice_loc/self.pixelScale)
        else:
            # Calculate horizontal reference lines if only one is provided
            if Ref_y0 != -1:
                Ref_y1 = Ref_y0-round(slice_loc/self.pixelScale)
            elif Ref_y1 != -1:
                Ref_y0 = Ref_y1+round(slice_loc/self.pixelScale)
            self.Reference.append(Ref_y0)
            cv2.line(self.clone, (0, Ref_y0), (shp[1], Ref_y0), CVColor.YELLOW, 1)
        return Ref_x0, Ref_y0, Ref_y1, Ref_y00

    def CleanSnapshots(self, img:np.ndarray[int], *args, **kwargs) -> np.ndarray[int]:
        """
        Clean and enhance snapshots based on specified corrections. This method takes an original 
        image snapshot `img` and applies specified corrections based on the provided `*args`.
        Supported corrections include 'Brightness/Contrast', 'Average', and 'FFT'.

        Parameters:
            - **img (numpy.ndarray)**: Original image snapshot.
            - ** *args (str)**: Variable-length argument list specifying the corrections to apply.
                           Supported corrections: 'Brightness/Contrast', 'Average', 'FFT'.
            - ** **kwargs**: Additional parameters for correction functions.
                FFT:
                    - **filterCenter (list)**: Overrides the default filter center if provided.
                    - **D (int)**: Overrides the default cut-off frequency if provided.
                    - **n (int)**: Overrides the default filter order if provided.
                    - **ShowIm (bool)**: Overrides the default value for displaying images if 
                      provided.
                Brightness/Contrast:
                    - **Brightness (float, optional)**: Brightness adjustment factor (default: 1). 
                      Valid range: 0 (min) to 2 (max).
                    - **Contrast (float, optional)**: Contrast adjustment factor (default: 1). 
                      Valid range: 0 (min) to 2 (max).
                    - **Sharpness (float, optional)**: Sharpness adjustment factor (default: 1). 
                      Valid range: 0 (min) to 3 (max).

        Returns:
            - numpy.ndarray: Corrected image snapshot.

        Example:
            >>> cleaned_image = instance.CleanSnapshots(original_image, 'Brightness/Contrast', 
                                                        'FFT', Brightness=1.5, D=20)

        .. note::
            - If 'Brightness/Contrast' is in `*args`, the image undergoes brightness and contrast 
              adjustments.
            - If 'Average' is in `*args`, the average illumination effect is removed.
            - If 'FFT' is in `*args`, the illumination effects are corrected using FFT-based 
              filtering.
        """

        new_log = 'Improving image quality ...'
        log_message(new_log, self.outputPath)
        print(new_log)
        
        CorrectedImg = img.copy()
        for arg in args:
            if arg == 'Average':
                CorrectedImg = SliceListAverage(CorrectedImg, self.outputPath)
            if arg == 'FFT':
                CorrectedImg = CleanIlluminationEffects(CorrectedImg, self.outputPath, **kwargs)
            if arg == 'Brightness/Contrast':
                CorrectedImg = BrightnessAndContrast(CorrectedImg, self.outputPath, **kwargs)
        log_message('Done', self.outputPath)
        return CorrectedImg

    def ShockTrakingAutomation(self, img:np.ndarray[int], method:str='integral', 
                               reviewInterval:list[int]=[0, 0], Signalfilter:str=None,
                               **kwargs) -> list[float]:
        """
        This method automates the shock tracking process and generates shock signals based on 
        linescanning technique, where a snapshots list is given as input, three methods of tracking
        can be proposed

            1. `integral`: This method tracks the shock through the largest blocked area by the 
               knife. More information and detailed discrepancies can be found in this article 
               https://dx.doi.org/10.1016/j.ast.2025.111022.
            2. `darkest_spot`: The shock is tracked by the abslute dark point of the schlieren 
               image
            3. `maxGrad`: By performing sobel gradient algorithem, the shock edge is determined as 
               the maximum gradient and tracked. More information can be found in this article 
               https://doi.org/10.1007/s00348-021-03145-3

        for better resolution and to avoid any missed shock location, signal filtering can be 
        applied, the method supports these methods

            1. `median`: run through the signal entry by entry, replacing each entry with the 
               median of the entry and its neighboring entries when entries are 3
            2. `Wiener`: based on minimizing the mean square error between the estimated random 
               process and the desired process.
            3. `med-Wiener`: use both filter sequentially

        Parameters:
            - **img (numpy.ndarray)**: Input image or image data.
            - **method (str)**: Method for shock tracking (integral, darkest_spot, maxGrad). 
              Default is 'integral'.
            - **reviewInterval (list)**: List containing two integers representing the review 
              interval. Available only with 'integral' method. Default is [0, 0].
            - **Signalfilter (str)**: The method for signal filtering (median, Wiener, med-Wiener). 
              Default is None.
            - **CheckSolutionTime (bool)**: Whether to check solution time. Default is True.
            - ** **kwargs**:

        Returns:
            numpy.ndarray: Generated shock signals.

        Example:
            >>> shock_signals = ShockTrakingAutomation(image,
                                                       method='integral',
                                                       reviewInterval=[10, 20],
                                                       Signalfilter='median',
                                                       CheckSolutionTime=True)

        """
        return GenerateShockSignal(img, method, Signalfilter, reviewInterval,
                                   self.outputPath, **kwargs)

    def VelocitySignal(self, Signal:list[float], TotalTime:float) -> list[float]:
        """
        Calculate the velocity signal from the given positional signal.
        The function calculates the velocity at each point in the Signal using
        finite differences. It uses a forward difference for the first point, a
        backward difference for the last point, and a central difference for
        all intermediate points. It then subtracts the average velocity from
        each point to return the velocity signal.

        Parameters:
            - **Signal (list or numpy.ndarray)**: Positional signal data points in mm.
            - **TotalTime (float)**: Total time duration over which the signal is recorded.

        Returns:
            numpy.ndarray: Velocity signal after removing the average velocity.

        Example:
            >>> signal = [0, 1, 2, 3, 4, 5]
            >>> total_time = 5.0
            >>> velocity_signal = VelocitySignal(signal, total_time)
            >>> print(velocity_signal)

        .. note::
            - The velocity is calculated in units per second, while the signal
              amplitudes are measured in millimeters (mm).
            - The returned velocity signal has the mean velocity subtracted.
        """
        n = len(Signal)       # Number of data points
        dx_dt = np.zeros(n)   # Array to store velocity signal
        dt = TotalTime/n      # Time interval between data points

        # forward difference for first point
        dx_dt[0] = (Signal[1] - Signal[0]) / 1000*dt
        # backward difference for last point
        dx_dt[-1] = (Signal[-1] - Signal[-2]) / 1000*dt
        # Central difference for all intermediate points
        for x in range(1, n - 1):
            dx_dt[x] = (Signal[x + 1] - Signal[x - 1]) / (2000 * dt)

        V_avg = np.mean(dx_dt)  # Calculate the average velocity
        V = dx_dt - V_avg       # Subtract the average velocity from each point
        return V
