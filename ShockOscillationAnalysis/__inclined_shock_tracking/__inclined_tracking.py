# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:47:27 2024

@author: admin
"""
import cv2
import sys
import glob
import screeninfo # ............................... To find the  monitor resolution
from .. import SOA
import numpy as np
import matplotlib.pyplot as plt
from ..__preview import plot_review
from ..ShockOscillationAnalysis import CVColor
from ..__shocktracking import ShockTraking
from sklearn.linear_model import LinearRegression
from ..__linedrawingfunctions import InclinedLine
from ..__imgcleaningfunctions import ImgListAverage
from ..__slice_list_generator.__list_generation_tools import GenerateIndicesList

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
        cv2.line(preview_img, P1new, P2new, CVColor.RED, 1)
        return anew 
        
    def InclinedShockDomainSetup(self, CheckingWidth, CheckingHieght, inclined_ref_line, imgShape,
                                 VMidPnt = 0, nPnts = 0, preview_img = []):
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
        SlicesInfo = []; inclinationCheck = True
        
        # generat the points
        if hasattr(CheckingHieght, "__len__"):
            Pnts = np.linspace(0, abs(CheckingHieght[1]- CheckingHieght[0]), nPnts)
            DatumY = CheckingHieght[0]
        else:
            Ht = int(CheckingHieght/2)
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
            if LineSlope != 0 and LineSlope != np.inf: 
                x_i1 = round((i+DatumY-aUp)/LineSlope)
                x_i2 = round((i+DatumY-aDown)/LineSlope)
            elif LineSlope == np.inf: 
                x_i1 = P1[0] - HalfSliceWidth; x_i2 = P1[0] + HalfSliceWidth
            elif LineSlope == 0: 
                print('Software is not supporting horizontal shock waves, aborting...')
                sys.exit()
                
            cv2.circle(preview_img, (x_i1,y_i), radius=3, color=CVColor.RED, thickness=-1)
            cv2.circle(preview_img, (x_i2,y_i), radius=3, color=CVColor.RED, thickness=-1)                
            SlicesInfo.append([[x_i1,x_i2],y_i])
        print(u'\u2713')
        return SlicesInfo, nPnts, inclinationCheck
    
    def InclinedShockTracking(self, imgSet, nSlices, Ref, nReview = 0, slice_thickness = 1, OutputDirectory = ''):        
        AvgAngleGlob= 0;   count = 0; xLoc = [];
        AvgSlope = 0; AvgMidLoc = 0;
        shp = imgSet[0].shape; 
        slice_width = Ref[0][0][1]-Ref[0][0][0]
        if len(shp) > 2: zero_slice = np.zeros([1,slice_width,3])
        else: zero_slice = np.zeros([1,slice_width])
        
        if slice_thickness > 1: Ht = int(slice_thickness/2)  # Half Thickness
        else: Ht = 1; slice_thickness = 2;
        
        upper_bounds = np.zeros(nSlices, dtype = int); lower_bounds = np.zeros(nSlices, dtype = int)
        for i in range(nSlices): 
            upper_bounds[i] = Ref[i][1] - Ht
            lower_bounds[i] = Ref[i][1] + Ht if slice_thickness%2 == 0 else Ref[i][1] + Ht + 1
            
        for img in imgSet:
            if count > 1: xLocOld = xLoc.copy()
            
            xLoc = []; ColumnY = []; uncertain = [];uncertainY = []
            for i in range(nSlices):
                y_i = Ref[i][1]
                x_i1 = Ref[i][0][0];x_i2 = Ref[i][0][1]

                Slice = zero_slice.copy()
                for sl in range(upper_bounds[i],lower_bounds[i]): 
                    Slice += img[sl-1 : sl, x_i1: x_i2]
                Slice /= slice_thickness
                
                if count > 1: LastShockLoc = xLocOld[i]
                else: LastShockLoc = -1
                
                # print(Slice)
                
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
    
    def ImportingFiles(self, pathlist, indices_list, n_images, imgs_shp, **kwargs):
        img_list=[]; n = 0; original_img_list=[]
        BG_path = kwargs.get('BG_path', '')
        resize_img = kwargs.get('resize_img', (imgs_shp[1],imgs_shp[0]))
        for i in indices_list:
            img = cv2.imread(pathlist[i])
            original_img_list.append(cv2.resize(img.astype('float32'), resize_img))
            n += 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n_images/20)), int(5*n/(n_images/20))))
        print('')
        
        if len(BG_path) > 0:
            print('Removing background image ...', end=" ")
            BG = cv2.imread(BG_path)
            BG = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            BG_len , BG_Wid  = BG.shape
            img_len, img_wid = imgs_shp
            
            if resize_img != imgs_shp: 
                BG_len = resize_img[0]; r = BG_Wid / BG_len
                BG = BG[0:BG_len,0:r*BG_len]
                
            if BG_len < img_len: img_len = BG_len; img_wid = BG_Wid
            else: BG = BG[0:img_len,0:img_wid]
            
            for img in original_img_list:
                New_img = cv2.subtract(img[0:img_len,0:img_wid],BG)
                img_list.append(New_img)
            print(u'\u2713')
        return original_img_list, img_list
    
    def ShockPointsTracking(self, path, tracking_V_range = [0,0],inclination_info = 0, nPnts = 0, scale_pixels = True, 
                            preview = True, OutputDirectory = '',comment='', **kwargs):
        
        files = sorted(glob.glob(path))
        n1 = len(files)
        # In case no file found end the progress and eleminate the program
        if n1 < 1: print('No files found!'); sys.exit();
        # Open first file and set the limits and scale
        Refimg = cv2.imread(files[0])
        shp = Refimg.shape; print('Img Shape is:', shp)
        Ref_x0 = kwargs.get('Ref_x0', [0,0])
        Ref_y0 = kwargs.get('Ref_y0', -1)
        resize_img = kwargs.get('resize_img', (shp[1],shp[0]))
        Refimg = cv2.resize(Refimg, resize_img)
        
        if scale_pixels: Ref_x0, Ref_y0, Ref_y1 = self.DefineReferences(Refimg, shp, Ref_x0, scale_pixels, Ref_y0)
        else: self.clone = Refimg.copy()
        
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        print(f'Screen resolution: {screen_width}, {screen_height}')
        
        if shp[0] >= screen_height*0.85:
            r = shp[0]/shp[1] # ---------- Image aspect ratio
            NewImgSize = (round(screen_height*0.85/r),round(screen_height*0.85))
            Refimg = cv2.resize(Refimg, NewImgSize)
            reductionRatio = NewImgSize[0]/shp[0]
            shp = NewImgSize
            print('Warning: Image hieght is larger than your monitor hieght')
            print(f'Only reference image will be adjusted to {shp}')
        
        tracking_V_range.sort(); start, end = tracking_V_range
        y_diff = abs(end-start);  draw_y = y_diff == 0       
        
        if draw_y:
            tracking_V_range = []
            # Vertical limits and scale 
            
            try:
                Ref_y1 = self.LineDraw(self.clone, 'H', 2)[-1]
            except Exception:
                Ref_y1 = Ref_y0;
                print(f'Nothing was drawn! Ref_y1 value is {Ref_y1}')
            tracking_V_range.append((Ref_y0 - Ref_y1)* self.pixelScale)
            try:
                Ref_y2 = self.LineDraw(self.clone, 'H', 2)[-1]
            except Exception:
                Ref_y2 = Ref_y1;
                print(f'Nothing was drawn! Ref_y1 value is {Ref_y2}')
                
            tracking_V_range.append((Ref_y0 - Ref_y2)* self.pixelScale)
            if Ref_y1 == Ref_y2: print('Vertical range of tracking is not sufficient!'); sys.exit()
            tracking_V_range.sort()
            if Ref_y1 > Ref_y2: Ref_y11 = Ref_y2; Ref_y2 = Ref_y1; Ref_y1 = Ref_y11;
        else:
            tracking_V_range.sort()
            Ref_y2, Ref_y1  = [round(Ref_y0 - (x / self.pixelScale)) for x in tracking_V_range] if Ref_y0 > -1 else tracking_V_range
            if Ref_y1< 0 or Ref_y2 > shp[0]: print('Vertical range of tracking is not sufficient!'); sys.exit()
            cv2.line(self.clone, (0,Ref_y1), (shp[1],Ref_y1), CVColor.YELLOW, 1)
            cv2.line(self.clone, (0,Ref_y2), (shp[1],Ref_y2), CVColor.YELLOW, 1)
            
        print(f'Vertical range of tracking points starts from {tracking_V_range[0]:0.2f}mm to {tracking_V_range[1]:0.2f}mm')
        print(f'But In pixels from {Ref_y1}px to {Ref_y2}px')
        
        # estemat shock domain
        if not hasattr(inclination_info, "__len__"):
            CheckingWidth = inclination_info
            if CheckingWidth < 10: 
                print('Reference width is not sufficient!'); 
                CheckingWidth = int(input("Please provide reference width >10px: "))
            inclined_ref_line = []
            try:
                inclined_ref_line = self.LineDraw(self.clone, 'Inc', 3)[-1] 
            except Exception:
                print(f'Nothing was drawn! inclined_ref_line value is {inclined_ref_line}')
            
            if not hasattr(inclined_ref_line, "__len__") or len(inclined_ref_line) < 4: 
                print('Reference lines are not sufficient!'); sys.exit()
           
        elif len(inclination_info) > 2:
            P1,P2,m,a = InclinedLine(inclination_info[1],inclination_info[2],imgShape = shp)
            cv2.line(self.clone, P1, P2, (0,255,0), 1)
            inclined_ref_line = [P1,P2,m,a]
            CheckingWidth = inclination_info
        
        if nPnts == 0: 
            while nPnts == 0:
                nPnts = int(input("Please provide number of points to be tracked: "))
                if nPnts > abs(Ref_y1-Ref_y2): print('insufficient number of points'); nPnts = 0

        Ref, nSlices, inclinationCheck = self.InclinedShockDomainSetup(CheckingWidth, 
                                                                       [Ref_y1,Ref_y2],
                                                                       inclined_ref_line,
                                                                       shp, nPnts = nPnts,
                                                                       preview_img = self.clone)
       
        if preview:
            cv2.imshow('investigation domain before rotating', self.clone)
            cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
            
            cv2.imwrite(f'{OutputDirectory}\\AnalysisDomain-Points.jpg', self.clone)

        import_n_files = kwargs.get('n_files', 0);
        if import_n_files == 0: import_n_files = kwargs.get('within_range', [0,0])
        import_step = kwargs.get('every_n_files', 1)
        indices_list, n_images = GenerateIndicesList(n1, import_n_files, import_step)
        
        if inclinationCheck:
            original_img_list, img_list = self.ImportingFiles(files, indices_list, n_images, shp, **kwargs)
            
        slice_thickness = kwargs.get('slice_thickness', 1)
        avg_shock_angle, avg_slope, avg_shock_loc = self.InclinedShockTracking(original_img_list, 
                                                                               nSlices, Ref,  
                                                                               slice_thickness = slice_thickness,
                                                                               nReview = n_images,
                                                                               OutputDirectory = OutputDirectory)
        
        
        
            
        
        
        