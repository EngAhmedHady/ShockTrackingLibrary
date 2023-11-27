
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:32:30 2022

@author: Ahmed Hady
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import signal
import glob
import sys
import time
import random
from sklearn.linear_model import LinearRegression
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
    
    def XCheck(self,x,Shp,slope,a):
        if   x >= 0 and x <= Shp[1]:                           p2 = (x, Shp[0])
        elif x >= 0 and x >  Shp[1]: y2 = int(Shp[1]*slope+a); p2 = (Shp[1],y2)
        elif x <  0 and x <= Shp[1]: y2 = int(a);              p2 = (0,y2)
        return p2    
    
    def TimeCalculation(self, timeInSec):
        if timeInSec > 60: 
            timeInMin = timeInSec/60
            if timeInMin > 60:
                timeInHr = int(timeInMin/60)
                Min = int((timeInSec%3600)/60)
                sec = (timeInSec%3600)%60
                print("Total run time: %s Hr, %s Min, %s Sec" % (timeInHr,Min,round(sec)))
            else:
                Min = int(timeInSec/60)
                sec = timeInSec%60
                print("Total run time: %s Min, %s Sec" % (Min,round(sec)))
        else: print("Total run time:  %s Sec" % round(timeInSec))
        
    def InclinedLine(self,P1, P2 = (), slope = None, imgShape = ()):
        # Generates the inclind line equation from two points or one point,slope 
        # The image boundary/shape should be given
        # inputs : P1       => first point tuple (a1,b1)
        # ........ P2       => second point tuple (a2,b2)
        # ........ slope    => second point tuple (a2,b2)
        # ........ imgShape => image size (y-length 'Number of raws', x-length'Number of columns')
        # outputs: - first boundary point tuple
        # ........ - second boundary point tuple
        # ........ - line slope  (equal to zero in case of vertical or horizontal)
        # ........ - y-intersept (equal to zero in case of vertical or horizontal)
        if len(imgShape) < 1: 
            print('Image shape is not provided, program aborting ...')
            sys.exit()
            
        if len(P2) > 0 and slope is None:
            dx = P1[0]-P2[0];   dy = P1[1]-P2[1]
            if dx != 0: slope = dy/dx
        elif len(P2) == 0 and slope is np.inf: dx = 0;
        else: dx = -1 
        
            
        if slope != 0 and slope is not None and slope is not np.inf:
            a = P1[1] - slope*P1[0]
            Xmax = int((imgShape[0]-a)/slope)
            Xmin = int(-a/slope)
            if   Xmin >= 0 and Xmin <= imgShape[1]:
                p1 = (Xmin,0)
                p2 = self.XCheck(Xmax,imgShape,slope,a)
            elif Xmin >= 0 and Xmin >  imgShape[1]:
                y = int(imgShape[1]*slope+a)
                p1 = (imgShape[1],y)
                p2 = self.XCheck(Xmax,imgShape,slope,a)
            else:
                y1 = int(a);
                p1 = (0,y1)
                p2 = self.XCheck(Xmax,imgShape,slope,a)
            return p1, p2, slope, a
        elif dx == 0:
            return (P1[0],0), (P1[0],imgShape[0]), np.Inf, 0 
        else:
            return (0,P1[1]), (imgShape[1],P1[1]), 0, 0   
    
    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click and draw
        # line that cross allover the image and store it in a global variable in case
        # of Horizontal or Vertical lines it takes the average between points
        # Drawing steps: 1- push the left mouse on the first point
        # .............. 2- pull the mouse cursor to the second point
        # .............. 3- the software will draw a thick red line (indecating the mouse locations)
        # ................. and green line indecating the generated averaged line 
        # .............. 4- to confrim press left click anywhere on the image, or
        # ................. to delete the line press right click anywhere on the image
        # .............. 5- press anykey to proceed
        
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
                    P1,P2,m,a = self.InclinedLine(self.TempLine[0],self.TempLine[1],imgShape = parameters[1])
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
                    P1,P2,m,a = self.InclinedLine(self.line_coordinates[0],self.line_coordinates[1],imgShape = parameters[1])
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
        # This function drive the extract_coordinates fucntion to draw lines 
        # Inputs: img         => is a single openCV image
        # ....... lineType    => 'V'   -> Vertical line 'starts from top to bottom of the image'
        # ...................... 'H'   -> Horizontal line 'starts from the left to the right'
        # ...................... 'Inc' -> Inclind line 'not averaging take the exact selected points'
        # ....... LineNameInd =>  Window title from the list 
        # ....... Intialize   => to rest the values of Reference and line_coordinates for new line set
        # ...................... True or False (Default: False)
        # Outputs: croping limits or (line set) 
        
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
    
    def ShockTraking(self, SnapshotSlice, LastShockLoc = -1, Plot = False, count = -1):
        # Start processing the slice
        avg = np.mean(SnapshotSlice) # ...... Average illumination on the slice   
        MinimumPoint = min(SnapshotSlice) # ........... minimum (darkest) point
        
        if Plot: # to plot slice illumination values with location and Avg. line
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(SnapshotSlice); ax.axhline(avg,linestyle = ':');
            ax.plot(SnapshotSlice); 
            # ax.plot(AvgLocation,AvgIllumination,linestyle = '-.');
        
        # Initiating Variables 
        MinA = 0 # ............................................... Minimum Area
        Pixels = len(SnapshotSlice) # ............................. Slice width
        localmin = [] # .............. Local Minimum set of illumination values
        LocMinI = [] # ......................... Local Minimum set of Locations 
        AeraSet = [] # ............................... local minimum areas list
       
        # Loop through the slice illumination values
        for pixel in range(Pixels):
            if SnapshotSlice[pixel] < avg: 
            # find the local minimum and store illumination values and location
                localmin.append(SnapshotSlice[pixel]); LocMinI.append(pixel)
                
                # find open local minimum at the end of the slice
                if pixel == Pixels-1 and len(localmin) >1: 
                    A = abs(np.trapz(avg-localmin))
                    SetMinPoint = min(localmin)
                    AeraSet.append(A)
                    if Plot: ax.fill_between(LocMinI, localmin,avg , alpha=0.5)
                    if A > MinA and SetMinPoint/MinimumPoint > 0.3: 
                        MinA = A;   ShockRegion = [LocMinI,localmin]
                    localmin = []; LocMinI = []
            
            # bounded local minimum
            elif SnapshotSlice[pixel] >= avg and len(localmin) > 1: 
                A = abs(np.trapz(avg-localmin))
                SetMinPoint = min(localmin)
                AeraSet.append(A)
                if Plot:ax.fill_between(LocMinI, localmin,avg , alpha=0.5)                        
                if A > MinA and SetMinPoint/MinimumPoint > 0.3:
                    MinA = A;   ShockRegion = [LocMinI,localmin]
                localmin = []; LocMinI = []
                
            else: localmin = [];  LocMinI = []
        
        # check if there is more than one valley in the local minimum
        LocMinAvg = np.mean(ShockRegion[1])
        if Plot: ax.plot([ShockRegion[0][0]-5,ShockRegion[0][-1]+5],[LocMinAvg,LocMinAvg],'-.r')
        
        localmin2 = [] # .............................. sub-local minimum value
        LocMinI2 = [] # ............................ sub-local minimum location
        SubLocalMinSets = [] # ........ Sub-local minimum set [Location, Value]
        n = 0 # ................ number of valleys in the sub-local minimum set
        # if Plot: print(min(ShockRegion[1])/MinimumPoint)
        for k in range(len(ShockRegion[1])):
            # check all pixels lays under the valley average line
            if ShockRegion[1][k] < LocMinAvg:
                localmin2.append(ShockRegion[1][k]); LocMinI2.append(ShockRegion[0][k])
            elif ShockRegion[1][k] >= LocMinAvg and len(localmin2) > 1:
                SubLocalMinSets.append([LocMinI2,localmin2])
                n += 1; localmin2 = []; LocMinI2 = []
            else:
                localmin2 = []; LocMinI2 = []
        # uncertainity calculation
        certainLoc = True
        reason = ''
        
        # if there is more than one valley in the local minimum, 
        # the closest to the preivous location will be choosen
        if n > 1 and LastShockLoc > -1:
            # The minimum distance between the sub-valley and last shock location
            # initiated with the full lenght 
            MinDis = Pixels;  
            AreaSet2 = [] # ......................... Set of sub-local minimums
            MaxArea2 = 0 # ................. minimum area in sub-local minimums
            for SubLocalMinSet in SubLocalMinSets:
                # calculating the area of the sub-valley
                A2 = abs(np.trapz(LocMinAvg-SubLocalMinSet[1]))
                AreaSet2.append(A2) # ........................ storing the area
                if A2 > MaxArea2: MaxArea2 = A2 # ...... Check the minimum area
                
                # check the location of the minimum illumination point from last snapshot location and choose the closest
                minValue = min(SubLocalMinSet[1]) # ........ find minimam illumination in the sub-set
                minLoc = SubLocalMinSet[1].index(minValue) # find the location of the minimam illumination in the sub-set
                
                Distance = abs(LastShockLoc-SubLocalMinSet[0][minLoc])                
                if Distance < MinDis: 
                    MinDis = Distance;  ShockRegion = SubLocalMinSet
                if Plot: ax.fill_between(ShockRegion[0], ShockRegion[1],avg , hatch='\\')
        elif n > 1 and LastShockLoc == -1: 
            n = 1; 
            certainLoc = False
            reason = 'First pexil slice, No shock location history'
        
        
        # Find the middel of the shock wave as middle point of RMS
        LocMinRMS = avg-np.sqrt(np.mean(np.array(avg-ShockRegion[1])**2))
        if Plot: 
            ax.plot([ShockRegion[0][0]-5,ShockRegion[0][-1]+5],[LocMinRMS,LocMinRMS],'-.k') 
            ax.fill_between(ShockRegion[0], ShockRegion[1],avg , hatch='///') 
            
        shockLoc = [];
        for elment in range(len(ShockRegion[1])):
            if ShockRegion[1][elment] <= LocMinRMS: shockLoc.append(ShockRegion[0][elment])
        minLoc = np.mean(shockLoc) 
        
        if Plot:
            ax.axvline(minLoc, linestyle = '--', color = 'b')
            if count > -1: ax.set_title(count)
            if LastShockLoc > -1:
                ax.axvline(LastShockLoc,linestyle = '--',color = 'orange')  
        
        for Area in AeraSet:
            Ra = Area/MinA
            if Ra > 0.6 and Ra < 1 and certainLoc:
                certainLoc = False
                reason = 'Almost equal Valleys'
        
        if n > 1 and certainLoc:
            for Area in AreaSet2:
                if MaxArea2 > 0: Ra = Area/MaxArea2
                if Ra > 0.5 and Ra < 1 and certainLoc: certainLoc = False; reason = 'Almost equal sub-Valleys'   
                if MaxArea2 !=  abs(np.trapz(LocMinAvg-ShockRegion[1])) and certainLoc: 
                    certainLoc = False; reason = 'different sub-Valleys than smallest'
        
        if (not certainLoc) and Plot: 
            ax.text(Pixels-130,0.55, 'uncertain: '+ reason, color = 'red', fontsize=14)
            # ax.set_ylim([-1,1])
        return minLoc, certainLoc, reason                    
    
    def InclinedShockCheck(self, CheckingRange, H_line, SliceThickness, imgShape):
        print('Shock inclination test')
        Ht = int(SliceThickness/2)
        Ref = []; inclinationCheck = True
        
        # generat the points
        if SliceThickness > 10:              Pnts = np.linspace(0, SliceThickness, 10)
        elif SliceThickness > 2 and SliceThickness <= 10: Pnts = range(SliceThickness)
        else: 
            print('escaping the shock angle checking... \nSlice thickness is not sufficient for check the shock angle')
            return Ref, 0, False
        
        self.LineDraw(self.clone, 'Inc', 3)                
        HalfSliceWidth = int(CheckingRange/2) 
        if len(self.Reference) < 4: 
            print('Reference lines is not sufficient!')
            sys.exit()
        
        # Define the estimated shock line using 2 points P1, P2 --> User defined
        P1 = (int(self.Reference[3][0][0]), int(self.Reference[3][0][1]))            
        LineSlope = self.Reference[3][2]
        
        # Define the shock domain of tracking ** the line upstream
        P1Up = (P1[0] - HalfSliceWidth, P1[1])
        aUp = P1Up[1] - LineSlope*P1Up[0] # y-intercept
        P1Up,P2Up,m,a = self.InclinedLine(P1Up, slope = LineSlope, imgShape=imgShape)            
        cv2.line(self.clone, P1Up, P2Up, (0,0,255), 1)
        
        # Define the shock domain of tracking ** the line downstream
        P1Down = (P1[0] + HalfSliceWidth, P1[1])
        aDown = P1Down[1] - self.Reference[3][2]*P1Down[0] # y-intercept
        P1Down,P2Down,m,a = self.InclinedLine(P1Down, slope = LineSlope, imgShape=imgShape) 
        cv2.line(self.clone, P1Down, P2Down, (0,0,255), 1)
        
        pointTranslation = H_line-Ht
        for i in Pnts: 
            y_i = int(i+pointTranslation)
            
            if self.Reference[3][2] != 0 and self.Reference[3][2] != np.inf: 
                x_i1 = int((i+pointTranslation-aUp)/self.Reference[3][2])
                x_i2 = int((i+pointTranslation-aDown)/self.Reference[3][2])
            elif self.Reference[3][2] == np.inf: 
                x_i1 = P1[0] - HalfSliceWidth; x_i2 = P1[0] + HalfSliceWidth
            elif self.Reference[3][2] == 0: 
                print('Software is not supporting horizontal shock waves, aborting...')
                sys.exit()
                
            cv2.circle(self.clone, (x_i1,y_i), radius=3, color=(0, 0, 255), thickness=-1)
            
            cv2.circle(self.clone, (x_i2,y_i), radius=3, color=(0, 0, 255), thickness=-1)                
            Ref.append([[x_i1,x_i2],y_i])
        return Ref, len(Pnts), inclinationCheck 
    
    def InclinedShockTracking(self, imgSet, nSlices, Ref,
                              nReview = 0, CheckSolutionTime = False, OutputDirectory = ''):
        if CheckSolutionTime: start_time = time.time()
        # Ploting conditions
        # reviewInterval.sort(); start = reviewInterval[0]; end = reviewInterval[1]
        AvgAngleGlob= 0;   count = 0; xLoc = [];
        imgShp = imgSet[0].shape; AvgSlope = 0; AvgMidLoc = 0;
        for img in imgSet:
            if count > 1: xLocOld = xLoc.copy()
            xLoc = []; ColumnY = []; uncertain = [];uncertainY = []
            for i in range(nSlices):
                y_i = Ref[i][1]
                x_i1 = Ref[i][0][0];x_i2 = Ref[i][0][1]
                Slice = img[y_i-1:y_i,x_i1:x_i2]
                if count > 1: LastShockLoc = xLocOld[i]
                else: LastShockLoc = -1
                ShockLoc, certainLoc, reason  = self.ShockTraking(Slice[0], LastShockLoc = LastShockLoc)
                ColumnY.append(y_i)
                xLoc.append(ShockLoc + Ref[i][0][0])
                if not certainLoc:
                    uncertain.append(xLoc[-1])
                    uncertainY.append(y_i)
            
            # finding the middle point
            if nSlices%2 != 0:
                midIndx = int(nSlices/2) + 1
                midLoc = xLoc[midIndx]
                y = ColumnY[midIndx]
            else: 
                midIndx = int(nSlices/2)
                # The average location between two points
                midLoc = ((xLoc[midIndx-1]+xLoc[midIndx])/2)
                y = (ColumnY[midIndx-1]+ColumnY[midIndx])/2
                
            ColumnXLoc = np.array(xLoc).reshape((-1, 1))
            ColumnY = np.array(ColumnY).reshape((-1, 1))
            model = LinearRegression().fit(ColumnXLoc, ColumnY)
            m = model.coef_[0][0]
            if m > 0:   
                AngReg = 180 - np.arctan(m)*180/np.pi
            elif m < 0: 
                AngReg = abs(np.arctan(m)*180/np.pi)
            else:
                AngReg = 90
                
            AvgMidLoc += midLoc
            AvgAngleGlob += AngReg
            AvgSlope += m
            if nReview > 0 and count < nReview:                 
                fig, ax = plt.subplots(figsize=(int(imgShp[1]*1.5*px), int(imgShp[0]*1.5*px)))
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
                ax.plot(xLoc, ColumnY,'-o',color = 'yellow' ,ms = 10)
                ax.plot(uncertain, uncertainY,'o',color = 'r' ,ms = 10)
                ax.plot(midLoc, y,'*',color = 'g' ,ms = 10)
                
                if len(OutputDirectory)> 0:
                    fig.savefig(OutputDirectory +'\\ShockAngleReview_'+str(f"{count:04d}")+'.png')
            count += 1
        # Shock tracking time
        if CheckSolutionTime:
            timeInSec =  time.time() - start_time  
            self.TimeCalculation(timeInSec)
        return AvgAngleGlob/count, AvgSlope/count, AvgMidLoc/count
                           
    def ImportSchlierenImages(self, path, ScalePixels = True, HLP = 0, WorkingRange = [] , FullImWidth = False,
                              SliceThickness = 0, nt = -1, Mode = -1, ShockAngleSamples = 30, AngleSamplesReview = 10,
                              OutputDirectory = '',comment=''):
        # This function is importing a seuqnce of image to perform single horizontal line shock wave analysis
        # for efficient and optimizied analysis the function extract only one pixel slice from each image
        # defined by the user and append one to another and finally generates a single image where each raw 
        # represent a snap shoot
        # Importing steps: 1- define the reference vertical boundaries which can be used for scaling as well
        # ................ 2- define the reference horizontal line [the slice is shifted by HLP from the reference]
        # ................ 3- The function will import all files, slice them and store the generated slices list into image
        # ...............................................................................................................
        # Inputs: path        => image path for sequence 'Ex: "Directory/FileName*.img"' of any image extensions 
        #....................... * referes to any
        # ....... ScalePixels => wheater scale the pixels with vertical limits by refrence distance (D/pixel)
        # ...................... in that case user should define D otherwise D will be 1 mm - (Default: True)
        # ....... HLP         => Horizontal line shift from the reference horizontal line in [mm] if ScalePixels is true
        # ...................... or pixels if ScalePixels is false (Default: 0)
        # ....... OutputDirectory => output directory to store slices list as png image but if it is empaty quetes 
        # .......................... slices list will not be stored  (Default: '')
        # Outputs: openCV image slices list, number of slices, horizontal slice location on the image [pixels]
        img_list=[]
        n = 0; o = 0; inclinationCheck = False
        # Find all files in the directory with the sequence and sorth them by name
        files = sorted(glob.glob(path))
        n1 = len(files)
        WorkingRangeLen = len(WorkingRange)
        
        if n1 > 1:
            img = cv2.imread(files[0])
            # Open first file and set the limits and scale
            self.Reference = []
            shp = img.shape
            print('Img Shape is:', shp)
            # Defining the working range
            if WorkingRangeLen < 2:
                # Vertical limits and scale 
                self.LineDraw(img, 'V', 0)
                self.LineDraw(self.clone, 'V', 1)
                if len(self.Reference) < 2: 
                    print('Reference lines is not sufficient!')
                    sys.exit()
            elif WorkingRangeLen > 1:
                self.clone = img.copy(); 
                cv2.line(self.clone, (WorkingRange[0],0), (WorkingRange[0],shp[0]), (0,255,0), 1)
                cv2.line(self.clone, (WorkingRange[1],0), (WorkingRange[1],shp[0]), (0,255,0), 1)
                self.Reference = WorkingRange[0:2].copy()
                
            self.Reference.sort() # to make sure that the limits are properly assigned
            if ScalePixels:  self.pixelScale = self.D / abs(self.Reference[1]-self.Reference[0])
            #----------------------------------------------------------
            # Alocate Horizontal reference
            if WorkingRangeLen < 3:
                self.LineDraw(self.clone, 'H', 2)
                if len(self.Reference) < 3: 
                    print('Reference lines is not sufficient!')
                    sys.exit()
                H_line = self.Reference[2]-round(HLP/self.pixelScale)
            else:
                self.Reference = WorkingRange
                H_line = WorkingRange[2]
                cv2.line(self.clone, (0     ,H_line+round(HLP/self.pixelScale)), 
                                     (shp[1],H_line+round(HLP/self.pixelScale)), 
                                     (0,255,255), 1)
             
            cv2.line(self.clone, (0,H_line), (shp[1],H_line), (0,0,255), 1)
            
            if SliceThickness > 0:
                Ht = int(SliceThickness/2)  # Half Thickness
                cv2.line(self.clone, (0,H_line+Ht), (shp[1],H_line+Ht), (0, 128, 255), 1)
                cv2.line(self.clone, (0,H_line-Ht), (shp[1],H_line-Ht), (0, 128, 255), 1)
                
            if len(WorkingRange) == 1:
                Ref, nSlices, inclinationCheck = self.InclinedShockCheck(WorkingRange[0], H_line, SliceThickness, shp)
            elif len(WorkingRange) == 4:
                Ref, nSlices, inclinationCheck = self.InclinedShockCheck(WorkingRange[4], H_line, SliceThickness, shp)
                

            if len(WorkingRange) < 3:    
                cv2.imshow(self.LineName[2], self.clone)
                cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
                
                        
            if  nt == -1 and Mode == -1: n1 = len(files)    
            elif Mode > 0 and nt > 0: n1 = int(nt/Mode)
            elif Mode > 0 and nt < 0: n1 = int(len(files)/Mode)
            else: n1 = nt
            if inclinationCheck:
                print('Shock inclination estimate ... ')
                randomIndx=[]
                if n1 < ShockAngleSamples: 
                    ShockAngleSamples = n1
                    print('ShockAngleSamples should not be more than number of files, number of files will be only considered')

                k = 0
                while k < ShockAngleSamples:
                   r =random.randint(0,n1) # ....................................... generating a random number in the range 1 to 100
                   # checking whether the generated random number is not in the randomList
                   if r not in randomIndx: 
                       randomIndx.append(r) # ................. appending the random number to the resultant list, if the condition is true
                       k += 1
                
                samplesList = []; k = 0
                for indx in randomIndx:
                    with open(files[indx]):
                        Sample = cv2.imread(files[indx])
                        # check if the image on grayscale or not and convert if not
                        if len(Sample.shape) > 2: Sample = cv2.cvtColor(Sample, cv2.COLOR_BGR2GRAY)
                        samplesList.append(Sample)
                    k += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-20s] %d%%" % ('='*int(k/(ShockAngleSamples/20)), int(5*k/(ShockAngleSamples/20))))
                print('')   
            

                if AngleSamplesReview < ShockAngleSamples: NSamplingReview = AngleSamplesReview
                else:
                    NSamplingReview = ShockAngleSamples
                    print('Warning: Number of samples is larger than requested to review, all samples will be reviewed')
            
                
                AvgAngleGlob, AvgSlopeGlob, AvgShockLocGlob = self.InclinedShockTracking(samplesList, nSlices, Ref, 
                                                                                         nReview = NSamplingReview, 
                                                                                         OutputDirectory = OutputDirectory)
                print('Average inclination angle {:.2f} deg'.format(AvgAngleGlob))
                
                M = cv2.getRotationMatrix2D((AvgShockLocGlob, H_line), 90-AvgAngleGlob, 1.0)
                NewImg = cv2.warpAffine(img, M, (shp[1],shp[0]))
                
                cv2.line(NewImg, (0,H_line-Ht), (shp[1],H_line-Ht), (255, 128, 0), 1)
                cv2.line(NewImg, (0,H_line+Ht), (shp[1],H_line+Ht), (255, 128, 0), 1)
                cv2.line(NewImg, (0,H_line), (shp[1],H_line), (255, 128, 255), 1)
                cv2.line(NewImg, (round(AvgShockLocGlob),0), (round(AvgShockLocGlob),shp[0]), (255,255,0), 1)
                cv2.line(NewImg, (self.Reference[0],0), (self.Reference[0],shp[0]), (0,255,0), 1)
                cv2.line(NewImg, (self.Reference[1],0), (self.Reference[1],shp[0]), (0,255,0), 1)                
                cv2.imshow(self.LineName[2], NewImg)
                cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
            elif WorkingRangeLen > 4:
                inclinationCheck = True
                print('Average inclination angle {:.2f} deg'.format(90-WorkingRange[3]))
                M = cv2.getRotationMatrix2D(WorkingRange[4], WorkingRange[3], 1.0)
                NewImg = cv2.warpAffine(img, M, (shp[1],shp[0]))
                cv2.line(NewImg, (0,H_line-Ht), (shp[1],H_line-Ht), (255, 128, 0), 1)
                cv2.line(NewImg, (0,H_line+Ht), (shp[1],H_line+Ht), (255, 128, 0), 1)
                cv2.line(NewImg, (0,H_line), (shp[1],H_line), (255, 128, 255), 1)
                cv2.line(NewImg, (round(90-WorkingRange[4][0]),0), (round(90-WorkingRange[4][0]),shp[0]), (255,255,0), 1)
                cv2.line(NewImg, (WorkingRange[0],0), (WorkingRange[0],shp[0]), (0,255,0), 1)
                cv2.line(NewImg, (WorkingRange[1],0), (WorkingRange[1],shp[0]), (0,255,0), 1)
                
            if len(OutputDirectory) > 0:
                if len(comment) > 0:
                    self.outputPath = OutputDirectory+'\\RefDomain-'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+ str(SliceThickness) +'_slice'+comment+'.png'
                else:
                    now = datetime.now()
                    now = now.strftime("%d%m%Y%H%M")
                    self.outputPath = OutputDirectory+'\\RefDomain'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+str(SliceThickness)+'_slice'+now+'.png'
                if inclinationCheck: cv2.imwrite(self.outputPath, NewImg)
                else: cv2.imwrite(self.outputPath, self.clone)
            
            if FullImWidth: 
                if WorkingRangeLen == 1: 
                    WorkingRange = [0,shp[1],H_line, 90-AvgAngleGlob, (round(AvgShockLocGlob), H_line)]
                elif WorkingRangeLen < 4:   
                    WorkingRange = [0,shp[1],H_line]
                else:
                    WorkingRange = [0,shp[1],H_line, WorkingRange[3], WorkingRange[4]]
                print ('scaling lines:', [self.Reference[0],self.Reference[1],H_line])
            elif WorkingRangeLen < 2: 
                if WorkingRangeLen == 1: 
                    WorkingRange = [self.Reference[0],self.Reference[1],H_line, 90-AvgAngleGlob, (round(AvgShockLocGlob), H_line)]
                else: WorkingRange = [self.Reference[0],self.Reference[1],H_line]
            
            print('working range is: ', WorkingRange)
            
            print('Importing images ... ')
            for name in files:
                if o%Mode == 0 and n < n1:
                    with open(name):
                        img = cv2.imread(name)
                        if SliceThickness > 0:
                            if inclinationCheck: 
                                img = cv2.warpAffine(img, M, (shp[1],shp[0]))
                            cropped_image = np.zeros([1,WorkingRange[1]-WorkingRange[0],3])
                            for i in range(SliceThickness): cropped_image += img[WorkingRange[2]-(Ht+1)+i:WorkingRange[2]-Ht+i,WorkingRange[0]:WorkingRange[1]]
                            cropped_image /= SliceThickness
                        else:
                            cropped_image = img[WorkingRange[2]-1:WorkingRange[2],
                                                WorkingRange[0]  :WorkingRange[1]]
                        img_list.append(cropped_image.astype('float32'))
                    n += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n1/20)), int(5*n/(n1/20))))
                o += 1
            print('')
            ImgList = cv2.vconcat(img_list)
            if len(OutputDirectory) > 0:
                if len(comment) > 0:
                    self.outputPath = OutputDirectory+'\\'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+ str(SliceThickness) +'_slice'+comment+'.png'
                else:
                    now = datetime.now()
                    now = now.strftime("%d%m%Y%H%M")
                    self.outputPath = OutputDirectory+'\\'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+ str(SliceThickness) +'_slice'+now+'.png'
                cv2.imwrite(self.outputPath, ImgList)
                print('File was stored:', self.outputPath)
        else:
            # In case no file found end the progress and eleminate the program
            print('No files found!')
            sys.exit()
        return ImgList,n,H_line,self.pixelScale
    
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
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        magnitude_spectrum = np.fft.fftshift(dft)
        imShp = magnitude_spectrum.shape
        x = imShp[1];   y = imShp[0]
        
        if ShowIm:
            fig, ax = plt.subplots(figsize=(30,20))
            spectrum_im = 20*np.log(np.abs(magnitude_spectrum)+1)
            im = ax.imshow(spectrum_im[:,:,0])
            ax.set_ylim([int(y/2)-20,int(y/2)+Spectlocation[1]+50])
            fig.colorbar(im)
            ax.set_title('Row Image FFT')

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
            ax.set_ylim([int(y/2)-20,int(y/2)+Spectlocation[1]+50])
            ax.set_title('Cleaned Image FFT')
            fig.colorbar(im)
            
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
            minLoc, certain, reason = self.ShockTraking(SnapshotSlice, 
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
    