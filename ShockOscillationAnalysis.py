
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:32:30 2022

@author: Ahmed H. Hanfy
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys
import time
# import glob
# import random
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
from __linedrawingfunctions import InclinedLine
from __shocktracking import ShockTraking
 
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
        # Window titles
        self.LineName = ["First Reference Line (left)",
                         "Second Reference Line (right)",
                         "Horizontal Reference Line",
                         "estimated shock location"]     
    
    def TimeCalculation(self, timeInSec):
        """
        Convert the given time in seconds into a formatted string representation.
    
        Parameters:
        - timeInSec (float): The time duration in seconds.
    
        Returns:
        None
    
        Example:
        >>> instance = SOA()
        >>> instance.TimeCalculation(3665)
    
        Note:
        - The function converts the time duration into hours, minutes, and seconds.
        - It prints the total run time in a human-readable format.
    
        """
        if timeInSec > 3600:
            timeInHr = timeInSec // 3600
            timeInMin = (timeInSec % 3600) // 60
            sec = (timeInSec % 3600) % 60
            print("Total run time: %s Hr, %s Min, %s Sec" % (timeInHr, timeInMin, round(sec)))
        elif timeInSec > 60:
            timeInMin = timeInSec // 60
            sec = timeInSec % 60
            print("Total run time: %s Min, %s Sec" % (timeInMin, round(sec)))
        else:
            print("Total run time: %s Sec" % round(timeInSec))
                
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
                # print('Starting: {}, Ending: {}'.format(self.TempLine[0], self.TempLine[1]))
                
                # Draw temprary line
                cv2.line(self.Temp, self.TempLine[0], self.TempLine[1], (0,0,255), 2)
                if parameters[2] == 'V':
                    avg = int((self.TempLine[0][0]+self.TempLine[1][0])/2)
                    cv2.line(self.Temp, (avg,0), (avg,parameters[1]), (0,255,0), 1)
                elif parameters[2] == 'H':
                    avg = int((self.TempLine[0][1]+self.TempLine[1][1])/2)
                    cv2.line(self.Temp, (0,avg), (parameters[1],avg), (0,255,255), 1)
                elif parameters[2] == 'Inc':
                    P1,P2,m,a = InclinedLine(self.TempLine[0],self.TempLine[1],imgShape = parameters[1])
                    cv2.line(self.Temp, P1, P2, (0,255,0), 1)
                    
                cv2.imshow(parameters[0], self.Temp)
            elif self.ClickCount == 2:
                   
                self.Temp = self.clone.copy()
                cv2.imshow(parameters[0], self.clone)
                # storing the vertical line
                if parameters[2] == 'V':
                    avg = int((self.line_coordinates[0][0]+self.line_coordinates[1][0])/2)
                    cv2.line(self.Temp, (avg,0), (avg,parameters[1]), (0,255,0), 1)
                
                # storing the Horizontal line
                elif parameters[2] == 'H':
                    avg = int((self.line_coordinates[0][1]+self.line_coordinates[1][1])/2)
                    cv2.line(self.Temp, (0,avg), (parameters[1],avg), (0,255,255), 1)
                    
                elif parameters[2] == 'Inc':
                    P1,P2,m,a = InclinedLine(self.line_coordinates[0],self.line_coordinates[1],imgShape = parameters[1])
                    cv2.line(self.Temp, P1, P2, (0,255,0), 1)
                    avg = [P1, P2, m,a]
                
                    
                self.Reference.append(avg)
                self.clone = self.Temp.copy()
                cv2.imshow(parameters[0], self.clone)
                
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
        if Intialize:
            self.Reference = []
            self.line_coordinates = []
        shp = img.shape
        if   lineType == 'V':
            prams = [self.LineName[LineNameInd],shp[0],lineType]
        elif lineType == 'H':
            prams = [self.LineName[LineNameInd],shp[1],lineType]
        elif lineType == 'Inc':
            prams = [self.LineName[LineNameInd],shp,lineType]
            
        cv2.imshow(self.LineName[LineNameInd], self.clone)
        cv2.setMouseCallback(self.LineName[LineNameInd], self.extract_coordinates,prams)
        # Wait until user press some key
        cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
        return self.Reference
    
    def Average(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width = len(img[0])
        Avg = np.zeros(width)
        for i in img: Avg += i
        Avg /= img.shape[0]
        Newimg = np.zeros(img.shape)
        for i in range(img.shape[0]):  Newimg[i] = img[i] - Avg
        return Newimg
        
    def CleanIlluminationEffects(self, img, Spectlocation = [0, 233], D = 10, n=10, ShowIm = False ):
        if len(img.shape) > 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        magnitude_spectrum = np.fft.fftshift(dft)
        imShp = magnitude_spectrum.shape
        x = imShp[1];   y = imShp[0]
        
        if ShowIm:
            fig, ax = plt.subplots(figsize=(30,20))
            spectrum_im = 20*np.log(np.abs(magnitude_spectrum)+1)
            im = ax.imshow(spectrum_im[:,:,0])
            ax.set_ylim([int(y/2)-20,int(y/2)+Spectlocation[1]+147])
            # fig.colorbar(im)
            # ax.set_title('Row Image FFT')

        LowpassFilter = np.ones([imShp[0],imShp[1],2])
              
        for i in range(y):
            for j in range(x):
                if i > y/2:
                    y_shift = int(y/2)+Spectlocation[1]
                    x_shift = int(x/2)+Spectlocation[0]
                    denominator = np.sqrt((i-y_shift)**2+(j-x_shift)**2)
                    if denominator <= 0: LowpassFilter[i][j] = 0
                    else: LowpassFilter[i][j]= 1/(1+(D/denominator)**(n*2))
                else: LowpassFilter[i][j]= 0
        
        CleanFFT = magnitude_spectrum*LowpassFilter
        
        if ShowIm:
            fig, ax = plt.subplots(figsize=(30,20))
            CleanFFT_im = 20*np.log(np.abs(CleanFFT)+1)
            im = ax.imshow(CleanFFT_im[:,:,0])
            ax.set_ylim([int(y/2)-20,int(y/2)+Spectlocation[1]+147])
            # ax.set_title('Cleaned Image FFT')
            # fig.colorbar(im)
            
        f_ishift = np.fft.ifftshift(CleanFFT)
        img_back = cv2.idft(f_ishift)
        CleanedImage = img_back[:,:,0]/np.amax(img_back[:,:,0])
        return CleanedImage
    
    def FindTheShockwaveImproved(self, img, reviewInterval = [0,0], Signalfilter=None, CheckSolutionTime = True):
        if CheckSolutionTime: start_time = time.time()
        # Initiating Variables
        ShockLocation = [] # ........................... set of shock locations
        uncertain = [] # set of uncertain shock locations [snapshot value, uncertain location]
        count = 0 # ................................ Processed snapshot counter
        
        # check ploting conditions
        reviewInterval.sort(); start = reviewInterval[0]; end = reviewInterval[1]
        plotingInterval = abs(end-start)
        if plotingInterval > 0: ploting = True
        else: ploting= False
        
        # check if the image on grayscale or not and convert if not
        if len(img.shape) > 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        nShoots = img.shape[0] # .................... total number of snapshots
        print('Processing the shock location')
        # check ploting conditions
        
        
        for SnapshotSlice in img:
            if ploting and count >= start and count<end: Plot = True
            else: Plot = False
            if len(ShockLocation) > 0: LastShockLocation = ShockLocation[-1]
            else :
                LastShockLocation = -1
            minLoc, certain, reason = ShockTraking(SnapshotSlice, 
                                                   LastShockLoc = LastShockLocation, 
                                                   Plot = Plot,
                                                   count = count)
            ShockLocation.append(minLoc)
            if not certain: uncertain.append([count,minLoc,reason])
            count += 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(count/(nShoots/20)), int(5*count/(nShoots/20))))
        print('')
            
        # for pnt in uncertain:
        #     if ShockLocation[pnt[0]] == pnt[1]: print('Uncorrected point at',pnt[0])
        if Signalfilter == 'median':
            print('Appling median filter...')
            ShockLocation = signal.medfilt(ShockLocation)
        elif Signalfilter == 'Wiener':
            print('Appling Wiener filter...')
            ShockLocation = signal.wiener(ShockLocation)
        elif Signalfilter == 'med-Wiener':
            print('Appling med-Wiener filter...')
            ShockLocation = signal.medfilt(ShockLocation)
            ShockLocation = signal.wiener(ShockLocation)
        
        # Shock tracking time
        if CheckSolutionTime:
            timeInSec =  time.time() - start_time
            self.TimeCalculation(timeInSec)
            
        return ShockLocation, uncertain
    