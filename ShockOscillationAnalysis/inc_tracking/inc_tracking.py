# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:47:27 2024

@author: admin
"""
import cv2
import sys
import glob
import screeninfo # ............................... To find the  monitor resolution
from ..ShockOscillationAnalysis import SOA
import numpy as np
import matplotlib.pyplot as plt
from ..preview import plot_review
from ..shocktracking import ShockTraking
from ..ShockOscillationAnalysis import CVColor
from ..linedrawingfunctions import InclinedLine, AngleFromSlope
# from ..imgcleaningfunctions import ImgListAverage
from ..slice_list_generator.list_generation_tools import GenerateIndicesList

px = 1/plt.rcParams['figure.dpi']
plt.rcParams.update({'font.size': 25})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"

class InclinedShockTracking(SOA):
    def __init__(self, f: int = 1, D: float = 1, pixelScale: float = 1):
        self.f = f # ----------------------- sampling rate (fps)
        self.D = D # ----------------------- refrence length for scaling (mm)
        self.pixelScale = pixelScale # ----- initialize scale of the pixels
        super().__init__(f, D, pixelScale)
        
    def shockDomain(self, Loc: str, P1: tuple[int], HalfSliceWidth: int, LineSlope: float, 
                    imgShape: tuple[int], preview_img = None) -> float:
        """
        Generate and visualize a shock domain based on the slice width and 
        the drawn line parameters (one point and slope).
    
        Parameters:
        - Loc (str): The shock direction, either 'up' or 'down'.
        - P1 (tuple): The starting point of the shock domain.
        - HalfSliceWidth (int): Half the width of the slice.
        - LineSlope (float): Slope of the inclined line.
        - imgShape (tuple): Image size (y-length, x-length).
        - preview_img (optional): Image for previewing the shock domain. Default is None.
    
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
        if preview_img is not None: cv2.line(preview_img, P1new, P2new, CVColor.RED, 1)
        return anew 

    def anglesInterpolation(self, pnts_y_list: list[int], 
                            flow_dir: list[float] = [], flow_Vxy:list[tuple] = [], 
                            **kwargs) -> list[float]:
        """
        Interpolate angles based on given y-coordinates and corresponding angles.
    
        Parameters:
            pnts_y_list (list): List of y-coordinates to interpolate angles for.
            angles_list (list): List of tubles contains the measured y-coordinates and the corresponding angles [(y_loc, angle),...].
    
        Returns:
            list: Interpolated angles for each y-coordinate in `pnts_y_list`. If the y-domain is out of valid range, returns an empty list.
    
        Example:
            >>> instance = YourClass()
            >>> pnts_y = [5, 15, 25]
            >>> flow_dir = [0, 10, 20, 30]
            >>> interpolated_angles = instance.anglesInterpolation(pnts_y, flow_dir)
            >>> print(interpolated_angles)
    
        Notes:
            - The function performs linear interpolation to determine the angles at specified y-coordinates.
            - If a y-coordinate in `pnts_y_list` is out of the range defined by `input_locs`, the function will abort and return an empty list.
        """
        
        if len(flow_dir) > 0:
            # Unzip the angles_list into separate locs and angles lists
            locs, angles = zip(*flow_dir)
        elif len(flow_Vxy):
            # Unzip the Vxy into separate locs, Vx, Vy lists
            locs, Vx, Vy = zip(*flow_Vxy)
            angles = np.arctan(np.array(Vy)/np.array(Vx))*180/np.pi
        
        if min(locs) > min(pnts_y_list) or max(locs) < max(pnts_y_list):
            print('provided y-domain is out of valid range, only boundary angles will considered ...')
        
        intr_flow_dir = [];
        for yi in pnts_y_list:
            # Perform binary search to find the correct interval for yi
            l, r = 0, len(angles)-1;
            while r > l and r-l > 1:
                mid = (r + l) // 2
                if locs[mid] <= yi: r = mid
                elif locs[mid] > yi: l = mid
            if locs[r] >= yi: intr_flow_dir.append(angles[locs.index(min(locs))])
            elif locs[l] <= yi: intr_flow_dir.append(angles[locs.index(max(locs))])
            else:
                intr_flow_dir.append(angles[r]+(yi-locs[r])*(angles[l]-angles[r])/(locs[l]-locs[r]))
        preview_angle_interpolation = kwargs.get('preview_angle_interpolation', False)
        if preview_angle_interpolation:
            fig, ax = plt.subplots(figsize=(10,20))
            ax.plot(angles, locs, '-o', ms = 5)
            ax.plot(intr_flow_dir, pnts_y_list, 'x', ms = 10)
        return intr_flow_dir


    def InclinedShockDomainSetup(self, CheckingWidth, CheckingHieght, inclined_ref_line, imgShape,
                                 VMidPnt = 0, nPnts = 0, preview_img = None):
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
        slices_info = []; inclinationCheck = True
        
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
                return slices_info, 0, False
        
        # IncInfoIndx = len(self.Reference) - 1
        HalfSliceWidth = round(CheckingWidth/2) 

        # Define the estimated shock line using 2 points P1, P2 --> User defined
        P1 = (round(inclined_ref_line[0][0]), round(inclined_ref_line[0][1]))            
        LineSlope = inclined_ref_line[2]
        
        aUp = self.shockDomain('up', P1, HalfSliceWidth, LineSlope, imgShape, preview_img)
        aDown = self.shockDomain('down', P1, HalfSliceWidth, LineSlope, imgShape, preview_img)
        
        y_i = np.array(Pnts + DatumY).astype(int)
        if LineSlope != 0 and LineSlope != np.inf:
            x_i1 = np.array((y_i - aUp) / LineSlope).astype(int)
            x_i2 =  np.array((y_i - aDown) / LineSlope).astype(int)
        elif LineSlope == np.inf:
            x_i1 = np.full(nPnts, P1[0] - HalfSliceWidth)
            x_i2 = np.full(nPnts, P1[0] + HalfSliceWidth)
        elif LineSlope == 0:
            print('Software is not supporting horizontal shock waves, aborting...')
            sys.exit()
            
        if preview_img is not None:
            for pnt in range(len(Pnts)):
                cv2.circle(preview_img, (x_i1[pnt],y_i[pnt]), radius=3, color=CVColor.RED, thickness=-1)
                cv2.circle(preview_img, (x_i2[pnt],y_i[pnt]), radius=3, color=CVColor.RED, thickness=-1)
        slices_info = x_i1,x_i2,y_i
        print(u'\u2713')
        # print(slices_info)
        return slices_info, nPnts, inclinationCheck

    def InclinedShockTracking(self, imgSet, nSlices, Ref, nReview = 0, slice_thickness = 1, 
                              output_directory = '', **kwargs):
        
        avg_ang_glob= 0;   count = 0; midLocs =[] ; xLocs = [];
        avg_slope = 0; AvgMidLoc = 0; columnY = []; m = []
        uncertain_list = []; uncertainY_list = []
        shp = imgSet[0].shape; 

        avg_preview_mode = kwargs.get('avg_preview_mode', None)
        review_slice_tracking = kwargs.get('review_slice_tracking', -1)
        slice_ploting_array = np.zeros(nReview)
        if hasattr(review_slice_tracking, "__len__") and len(review_slice_tracking) == 2:
            review_slice_tracking.sort(); start, end = review_slice_tracking
            for i in range(start, end): slice_ploting_array[i] = 1
        elif not hasattr(review_slice_tracking, "__len__") and review_slice_tracking > -1:
            slice_ploting_array[review_slice_tracking] = 1
        

        if slice_thickness > 1: Ht = int(slice_thickness/2)  # Ht -> Half Thickness
        else: Ht = 1; slice_thickness = 2;
        
        
        upper_bounds = np.zeros(nSlices, dtype = int); lower_bounds = np.zeros(nSlices, dtype = int)
        
        for i in range(nSlices): 
            upper_bounds[i] = Ref[2][i] - Ht
            lower_bounds[i] = Ref[2][i] + Ht if slice_thickness%2 == 0 else Ref[2][i] + Ht + 1
            columnY.append(Ref[2][i]) 
        columnY = np.array(columnY)

        midIndx = nSlices // 2
        midIndx2 = midIndx if nSlices % 2 != 0 else midIndx - 1
        y = (columnY[midIndx2] + columnY[midIndx]) / 2
        LastShockLoc = -1

        xLoc = -1*np.ones(nSlices)
        AngReg = []
        for count, img in enumerate(imgSet):
            xLocOld = xLoc.copy()
            xLoc = [];  uncertain = [];uncertainY = []
            for i in range(nSlices):
                x_i1, x_i2 = Ref[0][i], Ref[1][i]
                Slice = np.sum(img[upper_bounds[i]-1:lower_bounds[i], x_i1:x_i2], axis=0) / slice_thickness

                LastShockLoc = xLocOld[i]-Ref[0][i]
                ShockLoc, certainLoc, _  = ShockTraking(Slice, LastShockLoc = LastShockLoc, count = count, Plot = slice_ploting_array[count])
                # ShockLoc, certainLoc, _  = ShockTraking(Slice, LastShockLoc = LastShockLoc, count = count)
                xLoc.append(ShockLoc + Ref[0][i])
                if not certainLoc: uncertain.append(xLoc[-1]); uncertainY.append(Ref[2][i])

            # finding the middle point
            midLocs.append(np.mean([xLoc[midIndx], xLoc[midIndx2]]))

            xy = np.array(xLoc)*columnY; yy = columnY**2
            x_sum = np.sum(xLoc)       ; y_sum = np.sum(columnY)
            xy_sum = np.sum(xy)        ; yy_sum = np.sum(yy)

            m.append(1/((nSlices*xy_sum - x_sum * y_sum)/(nSlices*yy_sum - y_sum**2)))

            AngReg.append(AngleFromSlope(m[-1]))
            xLocs.append(xLoc)
            uncertain_list.append(uncertain); uncertainY_list.append(uncertainY)
            
        AvgMidLoc= np.mean(midLocs);  avg_ang_glob = np.mean(AngReg);
        if avg_preview_mode != 'avg_ang':
            avg_slope = np.mean(m)*np.ones(nReview)
            avg_midLoc = AvgMidLoc*np.ones(nReview)
            avg_ang = avg_ang_glob*np.ones(nReview)
        else:
            avg_slope = m; avg_midLoc = midLocs; avg_ang = AngReg

        print('Plotting tracked data ...')
        if nReview > 0:
            if nReview > 20: 
                print('For memory reasons, only 20 imgs will be displayed.')
                print('note: this will not be applied on imgs storing')
            for i in range(nReview):
                fig, ax = plt.subplots(figsize=(int(shp[1]*1.75*px), int(shp[0]*1.75*px)))
                ax.set_ylim([shp[0],0]); ax.set_xlim([0,shp[1]])
                plot_review(ax, imgSet[i], shp, xLocs[i], columnY, 
                            uncertain_list[i], uncertainY_list[i], 
                            avg_slope[i], avg_ang[i], avg_midLoc[i] , y, **kwargs)
                if len(output_directory) > 0: 
                    fig.savefig(fr'{output_directory}\ShockAngleReview_{i:04d}_Ang{avg_ang_glob:.2f}.png', bbox_inches='tight', pad_inches=0.1)

                if i > 18:
                    if len(output_directory) == 0: 
                        plt.close(fig); i = nReview
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-20s] %d%%" % ('='*int((i+1)/(nReview/20)), int(5*(i+1)/(nReview/20))))
                        break;
                    else: plt.close(fig)

                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int((i+1)/(nReview/20)), int(5*(i+1)/(nReview/20))))
            print()

        print(f'Angle range variation: [{min(AngReg):0.2f},{max(AngReg):0.2f}], \u03C3 = {np.std(AngReg):0.2f}')
        return avg_ang_glob, AvgMidLoc
    
    def ImportingFiles(self, pathlist, indices_list, n_images, imgs_shp, **kwargs):
        print(f'Importing {n_images} images ...')
        img_list=[]; n = 0; original_img_list=[]
        BG_path = kwargs.get('BG_path', '')
        resize_img = kwargs.get('resize_img', (imgs_shp[1],imgs_shp[0]))
        for i in indices_list:
            img = cv2.imread(pathlist[i])
            # original_img_list.append(cv2.resize(img.astype('float32'), resize_img))
            img_list.append(cv2.cvtColor(cv2.resize(img.astype('float32'), resize_img), cv2.COLOR_BGR2GRAY))
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
            
            for img in img_list:
                img = cv2.subtract(img[0:img_len,0:img_wid],BG)
            print(u'\u2713')
        return original_img_list, img_list
    
    def ShockPointsTracking(self, path, tracking_V_range = [0,0],inclination_info = 0, nPnts = 0, scale_pixels = True, 
                            preview = True, output_directory = '',comment='', **kwargs):
        
        files = sorted(glob.glob(path))
        n1 = len(files)
        # In case no file found end the progress and eleminate the program
        if n1 < 1: print('No files found!'); sys.exit();
        # Open first file and set the limits and scale
        Refimg = cv2.imread(files[0])
        Refimg = cv2.cvtColor(Refimg, cv2.COLOR_BGR2GRAY)
        Refimg = cv2.cvtColor(Refimg, cv2.COLOR_GRAY2BGR)
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
        
        # if shp[0] >= screen_height*0.85:
        #     r = shp[0]/shp[1] # ---------- Image aspect ratio
        #     NewImgSize = (round(screen_height*0.85/r),round(screen_height*0.85))
        #     Refimg = cv2.resize(Refimg, NewImgSize)
        #     reductionRatio = NewImgSize[0]/shp[0]
        #     shp = NewImgSize
        #     print('Warning: Image hieght is larger than your monitor hieght')
        #     print(f'Only reference image will be adjusted to {shp}')
        
        tracking_V_range.sort(); start, end = tracking_V_range
        y_diff = abs(end-start);  draw_y = y_diff == 0       
        
        if draw_y:
            tracking_V_range = []
            # Vertical limits and scale 
            try:
                Ref_y1 = self.LineDraw(self.clone, 'H', 2, line_color = CVColor.ORANGE)[-1]
            except Exception:
                Ref_y1 = Ref_y0;
                print(f'Nothing was drawn! Ref_y1 value is {Ref_y1}')
            tracking_V_range.append((Ref_y0 - Ref_y1)* self.pixelScale)
            try:
                Ref_y2 = self.LineDraw(self.clone, 'H', 2, line_color = CVColor.ORANGE)[-1]
            except Exception:
                Ref_y2 = Ref_y1;
                print(f'Nothing was drawn! Ref_y1 value is {Ref_y2}')
                
            tracking_V_range.append((Ref_y0 - Ref_y2)* self.pixelScale)
            if Ref_y1 == Ref_y2: print('Vertical range of tracking is not sufficient!'); sys.exit()
            tracking_V_range.sort()
            if Ref_y1 > Ref_y2: Ref_y11 = Ref_y2; Ref_y2 = Ref_y1; Ref_y1 = Ref_y11;
        else:
            tracking_V_range.sort() if Ref_y0 > -1 else tracking_V_range.sort(reverse=True)
            Ref_y2, Ref_y1  = [round(Ref_y0 - (x / self.pixelScale)) for x in tracking_V_range] if Ref_y0 > -1 else tracking_V_range
            if Ref_y1< 0 or Ref_y2 > shp[0]: print('Vertical range of tracking is not sufficient!'); sys.exit()
            cv2.line(self.clone, (0,Ref_y1), (shp[1],Ref_y1), CVColor.ORANGE, 1)
            cv2.line(self.clone, (0,Ref_y2), (shp[1],Ref_y2), CVColor.ORANGE, 1)
            
        print(f'Vertical range of tracking points starts from {tracking_V_range[0]:0.2f}mm to {tracking_V_range[1]:0.2f}mm')
        print(f'in pixels from {Ref_y1}px to {Ref_y2}px')
        
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
            CheckingWidth = inclination_info[0]
        
        if nPnts == 0: 
            while nPnts == 0:
                nPnts = int(input("Please provide number of points to be tracked: "))
                if nPnts > abs(Ref_y1-Ref_y2): print('insufficient number of points'); nPnts = 0

        Ref, nSlices, inclinationCheck = self.InclinedShockDomainSetup(CheckingWidth, 
                                                                       [Ref_y1,Ref_y2],
                                                                       inclined_ref_line,
                                                                       shp, nPnts = nPnts,
                                                                       preview_img = self.clone)
        
        pnts_y_list = []; 
        for i in range(nSlices): pnts_y_list.append((Ref_y0-Ref[2][i])*self.pixelScale)
        # input_locs = kwargs.get('input_locs', [])
        flow_dir = kwargs.get('flow_dir', [])
        flow_Vxy = kwargs.get('flow_Vxy', [])
        Mach_ang_mode = kwargs.get('Mach_ang_mode', None)
        if (len(flow_dir) > 0 or len(flow_Vxy) > 0) and Mach_ang_mode != None:
            kwargs['inflow_dir_deg'] = self.anglesInterpolation(pnts_y_list, **kwargs)
            kwargs['inflow_dir_rad'] = np.array(kwargs['inflow_dir_deg'])*np.pi/180

            
        # if len(inflow_dir_deg) > 0 and Mach_ang_mode != None:
        #      = inflow_dir_deg
        #      = inflow_dir_rad

        if preview:
            cv2.imshow('investigation domain before rotating', self.clone)
            cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
            
            cv2.imwrite(fr'{output_directory}\AnalysisDomain-Points.jpg', self.clone)

        import_n_files = kwargs.get('n_files', 0);
        if import_n_files == 0: import_n_files = kwargs.get('within_range', [0,0])
        import_step = kwargs.get('every_n_files', 1)
        indices_list, n_images = GenerateIndicesList(n1, import_n_files, import_step)
        
        if inclinationCheck:
            original_img_list, img_list = self.ImportingFiles(files, indices_list, n_images, shp, **kwargs)
            
        print('Shock tracking started ...')
        avg_shock_angle, avg_shock_loc = self.InclinedShockTracking(img_list, 
                                                                    nSlices, Ref,  
                                                                    nReview = n_images,
                                                                    output_directory = output_directory,
                                                                    **kwargs)
        print('Average inclination angle {:.2f} deg'.format(avg_shock_angle))
        
        return avg_shock_angle, avg_shock_loc