# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:45:35 2023

@author: Ahmed H. Hanfy
"""
import sys
import cv2
import time
import glob
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from ShockOscillationAnalysis import SOA
from __shocktracking import ShockTraking
from __linedrawingfunctions import InclinedLine
from sklearn.linear_model import LinearRegression
px = 1/plt.rcParams['figure.dpi']

class importSchlierenImages(SOA):
    def __init__(self, f, D=1, pixelScale = 1, Type='single pixel raw'):
        super().__init__(f, D, pixelScale, Type)

    
    def shockDomain(self, Loc, P1, HalfSliceWidth, LineSlope, imgShape):
        """
        Generate and visualize a shock domain based on the slice width and 
        the drawn line parameters (one point and slope).
    
        Parameters:
        - Loc (str): The shock direction, either 'up' or 'down'.
        - P1 (tuple): The starting point of the shock domain.
        - HalfSliceWidth (int): Half the width of the slice.
        - LineSlope (float): Slope of the inclined line.
        - imgShape (tuple): Image size (y-length, x-length).
    
        Returns:
        float: The y-intercept of the inclined line.
    
        Example:
        >>> from __importImages import importSchlierenImages
        >>> instance = importSchlierenImages(f)
        >>> slope_intercept = instance.shockDomain('up', (10, 20), 5, 0.5, (100, 200))
        >>> print(slope_intercept)
    
        Note:
        - The function generates and visualizes a shock domain line based on the specified parameters.
        - It returns the y-intercept of the inclined line.
    
        """
        if Loc =='up': P1new = (P1[0] - HalfSliceWidth, P1[1])
        else: P1new = (P1[0] + HalfSliceWidth, P1[1])  
        anew = P1new[1] - LineSlope*P1new[0] # y-intercept
        P1new,P2new,m,a = InclinedLine(P1new, slope = LineSlope, imgShape=imgShape)            
        cv2.line(self.clone, P1new, P2new, (0,0,255), 1)
        return anew

    def InclinedShockDomainSetup(self, CheckingWidth, CheckingHieght, imgShape, VMidPnt = 0, nPnts = 0):
        """
        Set up shock inclination test using inclined shock lines.
    
        Parameters:
        - CheckingWidth (int): Width for shock domain checking (sliceWidth).
        - CheckingHeight (int or list): Height for shock domain checking. If a list is provided,
          it represents a range of heights for generating points.
        - imgShape (tuple): Shape of the image (y-length, x-length).
        - VMidPnt (int, optional): Vertical midpoint. Default is 0.
        - nPnts (int, optional): Number of points to generate for inclined shock lines. Default is 0.
    
        Returns:
        tuple: A tuple containing:
        - SlicesInfo (list): List of shock domain slices, [[x-domainStrt,x-domainEnd],y-sliceLoc].
        - nPnts (int): Number of slices generated for inclined shock.
        - inclinationCheck (bool): Boolean indicating whether the shock inclination test is applicable.
    
        Example:
        >>> from __importImages import importSchlierenImages
        >>> instance = importSchlierenImages(f)
        >>> width = 20
        >>> height = [10, 20]
        >>> shape = (100, 200)
        >>> points = 5
        >>> slices, nPnts, success = instance.InclinedShockDomainSetup(width, height, shape, nPnts=points)
        >>> print(slices, nPnts, success)
    
        Note:
       - The function sets up shock inclination testing by visualizing the shock domain.
       - It returns a list of Slices location and range, the number of slices, and the inclination applicability.
    
        """
        print('Shock inclination test ...')
        Ht = int(CheckingHieght/2)
        SlicesInfo = []; inclinationCheck = True
        
        # generat the points
        if hasattr(CheckingHieght, "__len__"):
            CheckingHieght.sort()
            Pnts = np.linspace(CheckingHieght[0], CheckingHieght[1], nPnts)
            DatumY = CheckingHieght[0]
        else:
            DatumY = VMidPnt-Ht
            if CheckingHieght > 10:             Pnts = np.linspace(0, CheckingHieght, 10); nPnts = 10
            elif CheckingHieght > 2 and CheckingHieght <= 10: Pnts = range(CheckingHieght); nPnts = CheckingHieght
            else: 
                print('escaping the shock angle checking... \nSlice thickness is not sufficient for check the shock angle')
                return SlicesInfo, 0, False
        
        self.LineDraw(self.clone, 'Inc', 3)   
        IncInfoIndx = len(self.Reference) - 1           
        HalfSliceWidth = int(CheckingWidth/2) 
        if len(self.Reference) < 4: 
            print('Reference lines is not sufficient!')
            sys.exit()
        
        # Define the estimated shock line using 2 points P1, P2 --> User defined
        P1 = (int(self.Reference[IncInfoIndx][0][0]), int(self.Reference[IncInfoIndx][0][1]))            
        LineSlope = self.Reference[IncInfoIndx][2]

        aUp = self.shockDomain('up', P1, HalfSliceWidth, LineSlope, imgShape)
        aDown = self.shockDomain('down', P1, HalfSliceWidth, LineSlope, imgShape)

        for i in Pnts: 
            y_i = int(i+DatumY)
            if self.Reference[IncInfoIndx][2] != 0 and self.Reference[IncInfoIndx][2] != np.inf: 
                x_i1 = int((i+DatumY-aUp)/self.Reference[IncInfoIndx][2])
                x_i2 = int((i+DatumY-aDown)/self.Reference[IncInfoIndx][2])
            elif self.Reference[IncInfoIndx][2] == np.inf: 
                x_i1 = P1[0] - HalfSliceWidth; x_i2 = P1[0] + HalfSliceWidth
            elif self.Reference[IncInfoIndx][2] == 0: 
                print('Software is not supporting horizontal shock waves, aborting...')
                sys.exit()
                
            cv2.circle(self.clone, (x_i1,y_i), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.circle(self.clone, (x_i2,y_i), radius=3, color=(0, 0, 255), thickness=-1)                
            SlicesInfo.append([[x_i1,x_i2],y_i])
        return SlicesInfo, nPnts, inclinationCheck 
    
    def IntersectionPoint(self, M , A, Ref):
        """
        Calculate the intersection point between two lines.
    
        Parameters:
        - M (list): List containing slopes of the two lines.
        - A (list): List containing y-intercepts of the two lines.
        - Ref (list): List containing reference points for each line.
    
        Returns:
        tuple: A tuple containing:
        - Pint (tuple): Intersection point coordinates (x, y).
    
        Example:
        >>> from __importImages import importSchlierenImages
        >>> instance = importSchlierenImages(f)
        >>> slopes = [0.5, -2]
        >>> intercepts = [2, 5]
        >>> references = [(0, 2), (0, 5)]
        >>> intersection, angles = instance.IntersectionPoint(slopes, intercepts, references)
        >>> print(intersection, angles)
    
        Note:
        - The function calculates the intersection point and angles between two lines specified by their slopes and y-intercepts.
        - Returns the intersection point coordinates and angles of the lines in degrees.
    
        """
        theta1 = math.degrees(np.arctan(M[0]))
        theta2 = math.degrees(np.arctan(M[1]))
         
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
            print('Lines are parallel')
         
        Pint = (round(Xint), round(Yint))
        return Pint
        
    def plot_review(self, ax, img, x_loc, column_y, uncertain, uncertain_y, mid_loc, y):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.plot(x_loc, column_y, '-o', color='yellow', ms=10)
        ax.plot(uncertain, uncertain_y, 'o', color='r', ms=10)
        ax.plot(mid_loc, y, '*', color='g', ms=10)
    
    def InclinedShockTracking(self, imgSet, nSlices, Ref,
                              nReview = 0, CheckSolutionTime = False, OutputDirectory = ''):
        if CheckSolutionTime: start_time = time.time()
        
        AvgAngleGlob= 0;   count = 0; xLoc = [];
        AvgSlope = 0; AvgMidLoc = 0;
        
        for img in imgSet:
            if count > 1: xLocOld = xLoc.copy()
            
            xLoc = []; ColumnY = []; uncertain = [];uncertainY = []
            for i in range(nSlices):
                y_i = Ref[i][1]
                x_i1 = Ref[i][0][0];x_i2 = Ref[i][0][1]
                Slice = img[y_i-1:y_i,x_i1:x_i2]
                if count > 1: LastShockLoc = xLocOld[i]
                else: LastShockLoc = -1
                ShockLoc, certainLoc, reason  = ShockTraking(Slice[0], LastShockLoc = LastShockLoc)
                ColumnY.append(y_i)
                xLoc.append(ShockLoc + Ref[i][0][0])
                if not certainLoc:
                    uncertain.append(xLoc[-1])
                    uncertainY.append(y_i)
            
            # finding the middle point
            midIndx = nSlices // 2 + 1 if nSlices % 2 != 0 else nSlices // 2
            midLoc = xLoc[midIndx] if nSlices % 2 != 0 else (xLoc[midIndx - 1] + xLoc[midIndx]) / 2
            y = ColumnY[midIndx] if nSlices % 2 != 0 else (ColumnY[midIndx - 1] + ColumnY[midIndx]) / 2
            
            # if nSlices%2 != 0:
            #     midIndx = int(nSlices/2) + 1
            #     midLoc = xLoc[midIndx]
            #     y = ColumnY[midIndx]
            # else: 
            #     midIndx = int(nSlices/2)
            #     # The average location between two points
            #     midLoc = ((xLoc[midIndx-1]+xLoc[midIndx])/2)
            #     y = (ColumnY[midIndx-1]+ColumnY[midIndx])/2
                
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
                fig, ax = plt.subplots(figsize=(int(img.shape[1] * 1.5*px), int(img.shape[0] * 1.5*px)))
                self.plot_review(ax, img, xLoc, ColumnY, uncertain, uncertainY, midLoc, y)
                
                if len(OutputDirectory)> 0:
                    fig.savefig(OutputDirectory +'\\ShockAngleReview_'+str(f"{count:04d}")+'.png')
            count += 1
        # Shock tracking time
        if CheckSolutionTime:
            timeInSec =  time.time() - start_time  
            self.TimeCalculation(timeInSec)
        return AvgAngleGlob/count, AvgSlope/count, AvgMidLoc/count  
    
    
    def GenerateSlicesArray(self, path, ScalePixels = True, HLP = 0, WorkingRange = [] , FullImWidth = False,
                              SliceThickness = 0, nt = -1, Mode = -1, ShockAngleSamples = 30, AngleSamplesReview = 10,
                              OutputDirectory = '',comment=''):
        # This function is importing a seuqnce of image to perform single horizontal line shock wave analysis
        # for efficient and optimizied analysis the function extract only one pixel slice from each image
        # defined by the user and append one to another and finally generates a single image where each raw 
        # represent a snap shoot
        # Importing steps: 1- define the reference vertical boundaries which can be used for scaling as well
        # ................ 2- define the reference horizontal line [the slice is shifted by HLP from the reference]
        #................. 3- define the reference horizontal line [the slice is shifted by HLP from the reference]
        # ................ 4- The function will import all files, slice them and store the generated slices list into image
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
            shp = img.shape
            print('Img Shape is:', shp)
            # Defining the working range
            if WorkingRangeLen < 2:
                # Vertical limits and scale 
                self.LineDraw(img, 'V', 0,  Intialize = True)
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
                self.Reference.append(WorkingRange[2])
                H_line = WorkingRange[2]
                cv2.line(self.clone, (0     ,H_line+round(HLP/self.pixelScale)), 
                                     (shp[1],H_line+round(HLP/self.pixelScale)), 
                                     (0,255,255), 1)
             
            cv2.line(self.clone, (0,H_line), (shp[1],H_line), (0,0,255), 1)
            if SliceThickness > 0:
                Ht = int(SliceThickness/2)  # Half Thickness
                cv2.line(self.clone, (0,H_line+Ht), (shp[1],H_line+Ht), (0, 128, 255), 1)
                cv2.line(self.clone, (0,H_line-Ht), (shp[1],H_line-Ht), (0, 128, 255), 1)
                
            if WorkingRangeLen == 1:
                Ref, nSlices, inclinationCheck = self.InclinedShockDomainSetup(WorkingRange[0], SliceThickness, shp, H_line)
            elif WorkingRangeLen == 4:
                self.LineDraw(self.clone, 'Inc', 3)
                AvgShockLocGlob = self.IntersectionPoint([0, self.Reference[3][2]], 
                                                         [H_line, self.Reference[3][3]], 
                                                         [(0,H_line),self.Reference[3][0]])
                WorkingRange.append(AvgShockLocGlob)
                

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
            elif WorkingRangeLen > 3:
            
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
                cv2.circle(NewImg,WorkingRange[4], 5, (0,255,0), (2))
                
            if len(OutputDirectory) > 0:
                if len(comment) > 0:
                    self.outputPath = OutputDirectory+'\\RefDomain-'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+ str(SliceThickness) +'_slice'+comment
                else:
                    now = dt.now()
                    now = now.strftime("%d%m%Y%H%M")
                    self.outputPath = OutputDirectory+'\\RefDomain'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+str(SliceThickness)+'_slice'+now
                if inclinationCheck: 
                    cv2.imwrite(self.outputPath+'-Final.png', NewImg)
                    cv2.imwrite(self.outputPath+'.png', self.clone)
                else: cv2.imwrite(self.outputPath+'.png', self.clone)
            
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
                    now = dt.now()
                    now = now.strftime("%d%m%Y%H%M")
                    self.outputPath = OutputDirectory+'\\'+str(self.f/1000)+'kHz_'+str(HLP)+'mm_'+str(self.pixelScale)+'mm-px_ts_'+ str(SliceThickness) +'_slice'+now+'.png'
                cv2.imwrite(self.outputPath, ImgList)
                print('File was stored:', self.outputPath)
        else:
            # In case no file found end the progress and eleminate the program
            print('No files found!')
            sys.exit()
        return ImgList,n,H_line,self.pixelScale