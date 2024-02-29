# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:47:27 2024

@author: admin
"""
import cv2
import sys
from .. import SOA
import numpy as np
import matplotlib.pyplot as plt
from ..__preview import plot_review
from ..__shocktracking import ShockTraking
from sklearn.linear_model import LinearRegression
from ..__linedrawingfunctions import InclinedLine
px = 1/plt.rcParams['figure.dpi']

class inclinedShockTracking(SOA):
    def __init__(self, f, D=1, pixelScale = 1, Type='single pixel raw'):
        super().__init__(f, D, pixelScale, Type)
        
    def shockDomain(self, Loc, P1, HalfSliceWidth, LineSlope, imgShape, preview_img = []):
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
        cv2.line(preview_img, P1new, P2new, (0,0,255), 1)
        return anew 
        
    def InclinedShockDomainSetup(self, CheckingWidth, CheckingHieght, imgShape, 
                                 VMidPnt = 0, nPnts = 0, inclined_ref_line = [], preview_img = []):
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
        print('Shock inclination test and setup ...', end=" ")
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
                print(u'\u2717')
                print('Escaping the shock angle checking... \nSlice thickness is not sufficient for check the shock angle')
                return SlicesInfo, 0, False
        
        # IncInfoIndx = len(self.Reference) - 1           
        HalfSliceWidth = round(CheckingWidth/2) 
        
        # Define the estimated shock line using 2 points P1, P2 --> User defined
        P1 = (round(inclined_ref_line[0][0]), round(inclined_ref_line[0][1]))            
        LineSlope = inclined_ref_line[2]

        aUp = self.shockDomain('up', P1, HalfSliceWidth, LineSlope, imgShape, preview_img)
        aDown = self.shockDomain('down', P1, HalfSliceWidth, LineSlope, imgShape, preview_img)

        for i in Pnts: 
            y_i = int(i+DatumY)
            if inclined_ref_line[2] != 0 and inclined_ref_line[2] != np.inf: 
                x_i1 = round((i+DatumY-aUp)/inclined_ref_line[2])
                x_i2 = round((i+DatumY-aDown)/inclined_ref_line[2])
            elif inclined_ref_line[2] == np.inf: 
                x_i1 = P1[0] - HalfSliceWidth; x_i2 = P1[0] + HalfSliceWidth
            elif inclined_ref_line[2] == 0: 
                print('Software is not supporting horizontal shock waves, aborting...')
                sys.exit()
                
            cv2.circle(preview_img, (x_i1,y_i), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.circle(preview_img, (x_i2,y_i), radius=3, color=(0, 0, 255), thickness=-1)                
            SlicesInfo.append([[x_i1,x_i2],y_i])
        print(u'\u2713')
        return SlicesInfo, nPnts, inclinationCheck
    
    def InclinedShockTracking(self, imgSet, nSlices, Ref, nReview = 0, OutputDirectory = ''):        
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
                
            ColumnXLoc = np.array(xLoc).reshape((-1, 1))
            ColumnY = np.array(ColumnY).reshape((-1, 1))
            model = LinearRegression().fit(ColumnXLoc, ColumnY)
            m = model.coef_[0][0]
            
            if   m > 0:AngReg = 180 - np.arctan(m)*180/np.pi
            elif m < 0:AngReg = abs(np.arctan(m)*180/np.pi)
            else:      AngReg = 90
                
            AvgMidLoc += midLoc
            AvgAngleGlob += AngReg
            AvgSlope += m
            if nReview > 0 and count < nReview:                 
                fig, ax = plt.subplots(figsize=(int(img.shape[1] * 1.5*px), int(img.shape[0] * 1.5*px)))
                plot_review(ax, img, xLoc, ColumnY, uncertain, uncertainY, midLoc, y)
                
                if len(OutputDirectory)> 0:
                    fig.savefig(OutputDirectory +f'\\ShockAngleReview_{count:04d}_Ang{AngReg:.2f}.png')
            count += 1
        return AvgAngleGlob/count, AvgSlope/count, AvgMidLoc/count
    
    def ImportingFiles(self, pathlist, indices_list, n_images, imgs_shp, x_range, tk , M):
        img_list=[]; n = 0;
        slice_thickness =  tk[1]-tk[0]
        for i in indices_list:
            img = cv2.imread(pathlist[i])
            img = cv2.warpAffine(img, M, (imgs_shp[1],imgs_shp[0]))
            cropped_image = np.zeros([1,x_range[1]-x_range[0],3])
            
            for j in range(tk[0],tk[1]): 
                cropped_image += img[j-1 : j,
                                     x_range[0]: x_range[1]]
            cropped_image /= slice_thickness
            
            img_list.append(cropped_image.astype('float32'))
            n += 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n_images/20)), int(5*n/(n_images/20))))
        print('')
        img_list = cv2.vconcat(img_list)
        return img_list, n
    
    def ShockPointsTracking(self, path, tracking_H_range, scale_pixels = True, slice_loc = 0, preview = True,
                            OutputDirectory = '',comment='', inclination_est_info = [],**kwargs):
        pass
        
        