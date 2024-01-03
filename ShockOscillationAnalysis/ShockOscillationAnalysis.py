# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:32:30 2022

@author: Ahmed H. Hanfy
"""

import cv2
import screeninfo
import matplotlib.pyplot as plt
from .__Imagecleaningfunctions import Average, CleanIlluminationEffects, BrightnessAndContrast
from .__linedrawingfunctions import InclinedLine
from .__generateshocksignal import GenerateShockSignal


 
px = 1/plt.rcParams['figure.dpi']
plt.rcParams.update({'font.size': 30})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"


class SOA:
    def __init__(self, f, D = 1, pixelScale = 1, Type = 'single pixel raw'):
        self.f = f # ----------------------- sampling rate (fps)
        self.D = D # ----------------------- refrence distance (mm)
        self.pixelScale = pixelScale # ----- initialize scale of the pixels
        self.ClickCount = 0 # -------------- initialize the mouse clicks
        self.TempLine = [] # --------------- initialize the temporary line recording array
        self.Temp = cv2.vconcat([]) # ------ initialize the temporary image
        self.clone = cv2.vconcat([]) # ----- initialize the editable image copy
        self.Reference = [] # -------------- initialize croping limits or line set
        self.line_coordinates = [] # ------- initialize Line coordinate
        self.outputPath = '' # ------------- Image output
        self.Type = Type # ----------------- Type of shock analysis ['single pixel raw', 'multi point traking']
    
    
    def screenMidLoc(self, shp):
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        x_pos = (screen_width - shp[1]) // 2
        y_pos = (screen_height - shp[0]) // 2
        return x_pos, y_pos
    
    
    def extract_coordinates(self, event, x, y, flags, parameters):
        """
        Record starting (x, y) coordinates on left mouse button click and draw
        a line that crosses all over the image, storing it in a global variable.
        In case of horizontal or vertical lines, it takes the average between points.

        Drawing steps:
        1. Push the left mouse on the first point.
        2. Pull the mouse cursor to the second point.
        3. The software will draw a thick red line (indicating the mouse locations)
           and a green line indicating the Final line result.
        4. To confirm, press the left click anywhere on the image, or
           to delete the line, press the right click anywhere on the image.
        5. Press any key to proceed.

        Parameters:
        - event (int): The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
        - x (int): The x-coordinate of the mouse cursor.
        - y (int): The y-coordinate of the mouse cursor.
        - flags (int): Flags associated with the mouse event.
        - parameters (tuple): A tuple containing:
            - Name of the window to display the image.
            - Image shape (tuple of y-length and x-length).
            - Line type ('V' for vertical, 'H' for horizontal, 'Inc' for inclined).

        Returns:
        None

        Example:
        >>> instance = SOA()
        >>> cv2.setMouseCallback(window_name, instance.extract_coordinates, parameters)

        Note:
        - If 'Inc' is provided as the line type, it uses the 'InclinedLine' method
          to calculate the inclined line and display it on the image.

        """
        # win_x, win_y = self.screenMidLoc(parameters[1])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ClickCount += 1
            if len(self.TempLine) == 2: 
                self.line_coordinates = self.TempLine;
            elif len(self.TempLine) == 0: self.TempLine = [(x,y)]
            
        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP: 
            if len(self.TempLine) < 2:
                self.TempLine.append((x,y))
                print('Starting: {}, Ending: {}'.format(self.TempLine[0], self.TempLine[1]))
                
                # Draw temprary line
                cv2.line(self.Temp, self.TempLine[0], self.TempLine[1], (0,0,255), 2)
                if parameters[2] == 'V':
                    avg = int((self.TempLine[0][0]+self.TempLine[1][0])/2)
                    cv2.line(self.Temp, (avg,0), (avg,parameters[1][0]), (0,255,0), 1)
                elif parameters[2] == 'H':
                    avg = int((self.TempLine[0][1]+self.TempLine[1][1])/2)
                    cv2.line(self.Temp, (0,avg), (parameters[1][1],avg), (0,255,255), 1)
                elif parameters[2] == 'Inc':
                    P1,P2,m,a = InclinedLine(self.TempLine[0],self.TempLine[1],imgShape = parameters[1])
                    cv2.line(self.Temp, P1, P2, (0,255,0), 1)
                    
                # cv2.namedWindow(parameters[0], cv2.WINDOW_NORMAL)
                # cv2.moveWindow(parameters[0], win_x, win_y)    
                cv2.imshow(parameters[0], self.Temp)
            elif self.ClickCount == 2:
                   
                self.Temp = self.clone.copy()
                # cv2.namedWindow(parameters[0], cv2.WINDOW_NORMAL)
                # cv2.moveWindow(parameters[0], win_x, win_y)
                cv2.imshow(parameters[0], self.clone)
                # storing the vertical line
                if parameters[2] == 'V':
                    avg = int((self.line_coordinates[0][0]+self.line_coordinates[1][0])/2)
                    cv2.line(self.Temp, (avg,0), (avg,parameters[1][0]), (0,255,0), 1)
                
                # storing the Horizontal line
                elif parameters[2] == 'H':
                    avg = int((self.line_coordinates[0][1]+self.line_coordinates[1][1])/2)
                    cv2.line(self.Temp, (0,avg), (parameters[1][1],avg), (0,255,255), 1)
                    
                elif parameters[2] == 'Inc':
                    P1,P2,m,a = InclinedLine(self.line_coordinates[0],self.line_coordinates[1],imgShape = parameters[1])
                    cv2.line(self.Temp, P1, P2, (0,255,0), 1)
                    avg = [P1, P2, m,a]
                
                    
                self.Reference.append(avg)
                self.clone = self.Temp.copy()
                # cv2.namedWindow(parameters[0], cv2.WINDOW_NORMAL)
                # cv2.moveWindow(parameters[0], win_x, win_y)
                cv2.imshow(parameters[0], self.clone)
                
        # Delete draw line before storing    
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.TempLine = []
            if self.ClickCount>0: self.ClickCount -= 1
            self.Temp = self.clone.copy()
            # cv2.namedWindow(parameters[0], cv2.WINDOW_NORMAL)
            # cv2.moveWindow(parameters[0], win_x, win_y)
            cv2.imshow(parameters[0], self.Temp)
               
    def LineDraw(self, img, lineType, LineNameInd, Intialize = False):
        """
        Drive the extract_coordinates function to draw lines.

        Inputs:
        - img (numpy.ndarray): A single OpenCV image.
        - lineType (str): 'V' for Vertical line (starts from top to bottom of the image),
                         'H' for Horizontal line (starts from the left to the right),
                         'Inc' for Inclined line (not averaging, takes the exact selected points).
        - LineNameInd (int): Index of the window title from the list.
        - Initialize (bool, optional): To reset the values of Reference and line_coordinates for a new line set.
                                       True or False (Default: False).

        Outputs:
        list: Cropping limits or (line set).

        Example:
        >>> instance = YourClass()
        >>> line_set = instance.LineDraw(image, 'V', 0, Initialize=True)
        >>> print(line_set)

        Note:
        - The function uses the `extract_coordinates` method to interactively draw lines on the image.
        - It waits until the user presses a key to close the drawing window.

        """
        
        self.clone = img.copy(); 
        self.Temp = self.clone.copy();
        self.TempLine = [];
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
        win_x, win_y = self.screenMidLoc(shp)
        
        if   lineType == 'V':
            prams = [WindowHeader[LineNameInd],shp,lineType]
        elif lineType == 'H':
            prams = [WindowHeader[LineNameInd],shp,lineType]
        elif lineType == 'Inc':
            prams = [WindowHeader[LineNameInd],shp,lineType]
            
        # cv2.namedWindow(self.LineName[LineNameInd], cv2.WINDOW_NORMAL)
        # cv2.moveWindow(self.LineName[LineNameInd], win_x, win_y)
        cv2.imshow(WindowHeader[LineNameInd], self.clone)
        cv2.setMouseCallback(WindowHeader[LineNameInd], self.extract_coordinates,prams)
        # Wait until user press some key
        cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
        return self.Reference
    
    def CleanSnapshots(self, img,*args,**kwargs):
        """
        Clean and enhance snapshots based on specified corrections.
    
        Parameters:
        - img (numpy.ndarray): Original image snapshot.
        - *args (str): Variable-length argument list specifying the corrections to apply. 
                       Supported corrections: 'Brightness and Contrast', 'Average', 'FFT'.
        - **kwargs: Additional keyword arguments for correction functions.
    
        Returns:
        - numpy.ndarray: Corrected image snapshot.
    
        Example:
        >>> cleaned_image = CleanSnapshots(original_image, 'Brightness and Contrast', 'FFT', Brightness=1.5, D=20)
    
        This method takes an original image snapshot 'img' and applies specified corrections based on the provided *args.
        Supported corrections include 'Brightness and Contrast', 'Average', and 'FFT'.
    
        If 'Brightness and Contrast' is in *args, the image undergoes brightness and contrast adjustments.
        If 'Average' is in *args, the average illumination effect is removed.
        If 'FFT' is in *args, the illumination effects are corrected using FFT-based filtering.
    
        Additional keyword arguments (**kwargs) can be provided for fine-tuning the correction parameters.
    
        Returns the corrected image snapshot.
    
        .. note::
           Ensure that the correction functions 'BrightnessAndContrast', 'Average', and 'CleanIlluminationEffects'
           are defined and accessible in the class containing this method.
    
        """
        CorrectedImg = img.copy()
        if 'Brightness and Contrast' in args: CorrectedImg = BrightnessAndContrast(img, **kwargs)
        if 'Average' in args: CorrectedImg = Average(CorrectedImg)
        if 'FFT' in args: CorrectedImg = CleanIlluminationEffects(CorrectedImg, **kwargs)
        return CorrectedImg
    
    
    def ShockTrakingAutomation(self, img, reviewInterval = [0,0], Signalfilter=None, CheckSolutionTime = True):
        return GenerateShockSignal(img, reviewInterval, Signalfilter, CheckSolutionTime)
        