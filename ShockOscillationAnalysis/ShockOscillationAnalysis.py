# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:32:30 2022

@author: Ahmed H. Hanfy
"""

import cv2
import sys
import screeninfo
import numpy as np
import matplotlib.pyplot as plt
from .__imgcleaningfunctions import SliceListAverage, CleanIlluminationEffects, BrightnessAndContrast
from .__linedrawingfunctions import InclinedLine
from .__generateshocksignal import GenerateShockSignal
 
px = 1/plt.rcParams['figure.dpi']
plt.rcParams.update({'font.size': 25})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"

class CVColor:
    # Define class variables for commonly used colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    GREENBLUE = (255, 128, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    FUCHSIPINK = (255, 128, 255)
    GRAY = (128, 128, 128)

class SOA:
    def __init__(self, f = 1, D = 1, pixelScale = 1, Type = 'single pixel raw'):
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
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ClickCount += 1
            if len(self.TempLine) == 2: 
                self.line_coordinates = self.TempLine;
            elif len(self.TempLine) == 0: self.TempLine = [(x,y)]
            
        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP: 
            if len(self.TempLine) < 2:
                self.TempLine.append((x,y))
                
                # Draw temprary line
                cv2.line(self.Temp, self.TempLine[0], self.TempLine[1], (0,0,255), 2)
                if parameters[2] == 'V':
                    avg = int((self.TempLine[0][0]+self.TempLine[1][0])/2)
                    cv2.line(self.Temp, (avg,0), (avg,parameters[1][0]), CVColor.GREEN, 1)
                elif parameters[2] == 'H':
                    avg = int((self.TempLine[0][1]+self.TempLine[1][1])/2)
                    cv2.line(self.Temp, (0,avg), (parameters[1][1],avg), CVColor.YELLOW, 1)
                elif parameters[2] == 'Inc':
                    P1,P2,m,a = InclinedLine(self.TempLine[0],self.TempLine[1],imgShape = parameters[1])
                    cv2.line(self.Temp, P1, P2, CVColor.BLUE, 1)
                    
                cv2.imshow(parameters[0], self.Temp)
            elif self.ClickCount == 2:
                   
                self.Temp = self.clone.copy()
                cv2.imshow(parameters[0], self.clone)
                # storing the vertical line
                if parameters[2] == 'V':
                    avg = int((self.line_coordinates[0][0]+self.line_coordinates[1][0])/2)
                    cv2.line(self.Temp, (avg,0), (avg,parameters[1][0]), CVColor.GREEN, 1)
                
                # storing the Horizontal line
                elif parameters[2] == 'H':
                    avg = int((self.line_coordinates[0][1]+self.line_coordinates[1][1])/2)
                    cv2.line(self.Temp, (0,avg), (parameters[1][1],avg), CVColor.YELLOW, 1)
                    
                elif parameters[2] == 'Inc':
                    P1,P2,m,a = InclinedLine(self.line_coordinates[0],self.line_coordinates[1],imgShape = parameters[1])
                    cv2.line(self.Temp, P1, P2, CVColor.BLUE, 1)
                    avg = [P1, P2, m,a]
                
                self.Reference.append(avg)
                self.clone = self.Temp.copy()
                cv2.imshow(parameters[0], self.clone)
                print('Stored line: {}'.format(avg))
                
        # Delete draw line before storing    
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.TempLine = []
            if self.ClickCount>0: self.ClickCount -= 1
            self.Temp = self.clone.copy()
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
            
        cv2.imshow(WindowHeader[LineNameInd], self.clone)
        cv2.setMouseCallback(WindowHeader[LineNameInd], self.extract_coordinates,prams)
        # Wait until user press some key
        cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
        return self.Reference
    
    def DefineReferences(self, img, shp, Ref_x0, scale_pixels, Ref_y0 = -1, Ref_y1 = -1, slice_loc = 0):
        Ref_x0.sort(); start, end = Ref_x0
        x0_diff = abs(end-start);  draw_x0 = x0_diff == 0       
        
        if draw_x0:
            # Vertical limits and scale 
            self.LineDraw(img, 'V', 0,  Intialize = True)
            self.LineDraw(self.clone, 'V', 1)
            Ref_x0 = self.Reference
            if len(Ref_x0) < 2: print('Reference lines are not sufficient!'); sys.exit()
            
        else:
            self.clone = img.copy(); 
            cv2.line(self.clone, (Ref_x0[0],0), (Ref_x0[0],shp[0]), CVColor.GREEN, 1)
            cv2.line(self.clone, (Ref_x0[1],0), (Ref_x0[1],shp[0]), CVColor.GREEN, 1)
            self.Reference = Ref_x0[0:2].copy()

        Ref_x0.sort() # to make sure that the limits are properly assigned
        
        if scale_pixels:  self.pixelScale = self.D / abs(Ref_x0[1]-Ref_x0[0])
        print(f'Image scale: {self.pixelScale}')
        
        #----------------------------------------------------------
        
        # Alocate Horizontal reference
        if Ref_y0 == -1 and Ref_y1 == -1:
            self.LineDraw(self.clone, 'H', 2)  # to draw the reference line
            if len(self.Reference) < 3: print('Reference lines are not sufficient!'); sys.exit()
            Ref_y0 = self.Reference[-1]
            Ref_y1 = self.Reference[-1]-round(slice_loc/self.pixelScale)
        else:
            if   Ref_y0 != -1: Ref_y1 = Ref_y0-round(slice_loc/self.pixelScale)
            elif Ref_y1 != -1: Ref_y0 = Ref_y1+round(slice_loc/self.pixelScale)
            self.Reference.append(Ref_y0)
            cv2.line(self.clone, (0,Ref_y0),(shp[1],Ref_y0), CVColor.YELLOW, 1)
        return Ref_x0, Ref_y0, Ref_y1
    
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
        
        print('Improving image quality ...')
        
        for arg in args:
            if arg == 'SL_Average': CorrectedImg = SliceListAverage(CorrectedImg)
            if arg == 'SL_FFT': CorrectedImg = CleanIlluminationEffects(CorrectedImg, **kwargs)
            if arg == 'SL_Brightness/Contrast': CorrectedImg = BrightnessAndContrast(CorrectedImg, **kwargs)
        return CorrectedImg
    
    def ShockTrakingAutomation(self, img, method = 'integral', reviewInterval = [0,0], Signalfilter=None, CheckSolutionTime = True, **kwargs):
        return GenerateShockSignal(img, method, Signalfilter, reviewInterval, CheckSolutionTime, **kwargs)
    
    def VelocitySignal(self, Signal, TotalTime):
        n = len(Signal);  dx_dt = np.zeros(n) 
        dt = TotalTime/n;
        
        dx_dt[0] = (Signal[1] - Signal[0]) / 1000*dt            #forward difference for first point
        dx_dt[-1] = (Signal[-1] - Signal[-2]) / 1000*dt         #backward difference for last point
        
        for x in range(1, n - 1):
            dx_dt[x] = (Signal[x + 1] - Signal[x - 1]) / (2000 * dt)

        V_avg = np.mean(dx_dt) 
        V = dx_dt - V_avg
        return V
        