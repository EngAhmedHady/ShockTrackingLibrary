# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:45:35 2023

@author: Ahmed H. Hanfy
"""
import sys
import cv2
import glob
import math
import numpy as np
from .. import SOA
from datetime import datetime as dt
from ..__preview import PreviewCVPlots
from ..ShockOscillationAnalysis import CVColor
from ..__linedrawingfunctions import InclinedLine
from ..__inclined_shock_tracking.__inclined_tracking import inclinedShockTracking
from .__list_generation_tools import genratingRandomNumberList, GenerateIndicesList


class sliceListGenerator(SOA):
    def __init__(self, f, D=1, pixelScale = 1, Type='single pixel raw'):
        # self.f = f # ----------------------- sampling rate (fps)
        # self.D = D # ----------------------- refrence distance (mm)
        # self.pixelScale = pixelScale # ----- initialize scale of the pixels
        self.inc_trac = inclinedShockTracking(f,D)
        super().__init__(f, D, pixelScale)

    
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
            
    def ImportingFiles(self, pathlist, indices_list, n_images, imgs_shp, x_range, tk , M):
        img_list=[]; n = 0;
        slice_thickness =  tk[1]-tk[0]
        for i in indices_list:
            img = cv2.imread(pathlist[i])
            img = cv2.warpAffine(img, M, (imgs_shp[1],imgs_shp[0]))
            cropped_image = np.zeros([1,x_range[1]-x_range[0],3])
            
            # cropped_image = np.sum(img[tk[0]-1:tk[1], x_range[0]:x_range[1]], axis=0) / slice_thickness
            
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
        
    def GenerateSlicesArray(self, path, scale_pixels = True, slice_loc = 0, full_img_width = False, preview = True,
                            slice_thickness = 0, shock_angle_samples = 30, angle_samples_review = 10,
                            OutputDirectory = '',comment='', inclination_est_info = [],**kwargs):
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
        inclinationCheck = False
        # Find all files in the directory with the sequence and sorth them by name
        files = sorted(glob.glob(path))
        n1 = len(files)
        
        # In case no file found end the progress and eleminate the program
        if n1 < 1: print('No files found!'); sys.exit();
        
        # Open first file and set the limits and scale
        img = cv2.imread(files[0])
        
        shp = img.shape; print('Img Shape is:', shp)        
        Ref_x0 = kwargs.get('Ref_x0', [0,0])
        Ref_y0 = kwargs.get('Ref_y0', -1);    Ref_y1 = kwargs.get('Ref_y1', -1)
        
        Ref_x0, Ref_y0, Ref_y1 = self.DefineReferences(img, shp, 
                                                                Ref_x0, scale_pixels, 
                                                                Ref_y0, Ref_y1, slice_loc)
        print(f'Slice is located at: {Ref_y1}px')
        if Ref_y1 > 0 and Ref_y1 != Ref_y0: cv2.line(self.clone, (0,Ref_y1), (shp[1],Ref_y1), CVColor.RED, 1)

        if slice_thickness > 0: Ht = int(slice_thickness/2)  # Half Thickness
        else: Ht = 1; 
        
        upper_bounds =  Ref_y1 - Ht; 
        lower_bounds =  Ref_y1 + Ht if slice_thickness%2 == 0 else  Ref_y1 + Ht + 1
        cv2.line(self.clone, (0,lower_bounds), (shp[1],lower_bounds), CVColor.ORANGE, 1)
        cv2.line(self.clone, (0,upper_bounds), (shp[1],upper_bounds), CVColor.ORANGE, 1)
            
        avg_shock_angle = kwargs.get('avg_shock_angle', 90)
        avg_shock_loc = kwargs.get('avg_shock_loc', 0)
        if not hasattr(inclination_est_info, "__len__"):
            self.LineDraw(self.clone, 'Inc', 3)
            if len(self.Reference) < 4: print('Reference lines are not sufficient!'); sys.exit()
            P1,P2,m,a = self.Reference[3]
            Ref, nSlices, inclinationCheck = self.inc_trac.InclinedShockDomainSetup(inclination_est_info,
                                                                                    slice_thickness, [P1,P2,m,a], 
                                                                                    shp, VMidPnt = Ref_y1, 
                                                                                    preview_img = self.clone)
        elif len(inclination_est_info) > 2:
            P1,P2,m,a = InclinedLine(inclination_est_info[1],inclination_est_info[2],imgShape = shp)
            cv2.line(self.clone, P1, P2, CVColor.GREEN, 1)
            self.Reference.append([P1, P2, m,a])
            Ref, nSlices, inclinationCheck = self.inc_trac.InclinedShockDomainSetup(inclination_est_info[0],
                                                                                    slice_thickness, [P1,P2,m,a], 
                                                                                    shp, VMidPnt = Ref_y1, 
                                                                                    preview_img = self.clone)
        elif avg_shock_angle != 90 and avg_shock_loc == 0: # in case the rotation angle only is provieded in working _range
            print('Please, provide the rotation center...')
            self.LineDraw(self.clone, 'Inc', 3)
            # find the rotation center
            avg_shock_loc = self.IntersectionPoint([0,         self.Reference[-1][2]], 
                                                   [Ref_y1,    self.Reference[-1][3]], 
                                                   [(0,Ref_y1),self.Reference[-1][0]])
            
        if preview:
            cv2.imshow('investigation domain before rotating', self.clone)
            cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
        
        # number of files to be imported 
        import_n_files = kwargs.get('n_files', 0);
        if import_n_files == 0: import_n_files = kwargs.get('within_range', [0,0])
        import_step = kwargs.get('every_n_files', 1)
        indices_list, n_images = GenerateIndicesList(n1, import_n_files, import_step)
        
        if inclinationCheck:
            print('Shock inclination estimation ... ')
            
            randomIndx = genratingRandomNumberList(shock_angle_samples, n1)

            samplesList = []; k = 0
            for indx in randomIndx:
                Sample = cv2.imread(files[indx])
                # check if the image on grayscale or not and convert if not
                if len(Sample.shape) > 2: Sample = cv2.cvtColor(Sample, cv2.COLOR_BGR2GRAY)
                samplesList.append(Sample)
                k += 1
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int(k/(shock_angle_samples/20)), int(5*k/(shock_angle_samples/20))))
            print('')

            if angle_samples_review < shock_angle_samples: NSamplingReview = angle_samples_review
            else:
                NSamplingReview = shock_angle_samples
                print('Warning: Number of samples is larger than requested to review!, all samples will be reviewed')

            avg_shock_angle, avg_shock_loc = self.inc_trac.InclinedShockTracking(samplesList, nSlices, Ref,
                                                                                            nReview = NSamplingReview, 
                                                                                            OutputDirectory = OutputDirectory)
        print('Average inclination angle {:.2f} deg'.format(avg_shock_angle))
            
        M = cv2.getRotationMatrix2D((avg_shock_angle, Ref_y1), 90-avg_shock_angle, 1.0)
        new_img = cv2.warpAffine(img, M, (shp[1],shp[0]))
        
        new_img = PreviewCVPlots(new_img, Ref_x0, Ref_y = Ref_y1, 
                                 tk = [lower_bounds,upper_bounds], 
                                 avg_shock_loc = avg_shock_loc)            
        
        if avg_shock_angle != 90 and preview:
            cv2.imshow('Final investigation domain', new_img)
            cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
            
        if len(OutputDirectory) > 0:
            if len(comment) > 0:
                outputPath = f'{OutputDirectory}\\{self.f/1000:.1f}kHz_{slice_loc}mm_{self.pixelScale}mm-px_tk_{slice_thickness}px_{comment}'
            else:
                now = dt.now()
                now = now.strftime("%d%m%Y%H%M")
                outputPath =f'{OutputDirectory}\\{self.f/1000:.1f}kHz_{slice_loc}mm_{self.pixelScale}mm-px_tk_{slice_thickness}px_{now}'
            if avg_shock_angle != 90:
                print('RotatedImage:', u"stored \u2713" if cv2.imwrite(outputPath+f'-RefD{round(avg_shock_angle,2)}deg.png', new_img) else "Failed !")
                print('DomainImage:' , u"stored \u2713" if cv2.imwrite(outputPath+'-RefD.png', self.clone)   else "Failed !")
            else: print('DomainImage:',u"stored \u2713" if cv2.imwrite(outputPath+'-RefD.png', self.clone)   else "Failed !") 
                
        if full_img_width:
            x_range = [0, shp[1]]
            working_range = {'Ref_x0': [0, shp[1]], 'Ref_y1': Ref_y1, 
                            'avg_shock_angle': avg_shock_angle, 'avg_shock_loc': avg_shock_loc}
            print (f'scaling lines: Ref_x0 = {Ref_x0}, Ref_y1 = {Ref_y1}')

        else:
            x_range = Ref_x0
            working_range = {'Ref_x0': Ref_x0, 'Ref_y1': Ref_y1, 
                            'avg_shock_angle': avg_shock_angle, 'avg_shock_loc': avg_shock_loc}
            
        print ('working range is: ', working_range)
        print(f'Importing {n_images} images ...')
        img_list, n = self.ImportingFiles(files, indices_list, n_images, shp, x_range, [upper_bounds,lower_bounds], M)

        if len(OutputDirectory) > 0:
            print('ImageList write:', f"File was stored: {outputPath}.png" if cv2.imwrite(f'{outputPath}.png', img_list) else "Failed !")
                
        return img_list,n,working_range,self.pixelScale