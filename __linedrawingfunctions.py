# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:38:37 2023

@author: Ahmed H. Hanfy
"""
import sys
import cv2
import numpy as np

def XCheck(self,x,Shp,slope,a):
    """
    Check and calculate the image boundary y-coordinate based on the given x-coordinate and slope.

    This function takes an x-coordinate, image shape parameters (Shp), slope, and intercept (a) as inputs,
    and calculates the corresponding y-coordinate (p2) based on the specified conditions.

    Parameters:
    - x (float): The x-coordinate to be checked.
    - Shp (tuple): A tuple containing shape parameters (Shp[0] for y-axis limit, Shp[1] for x-axis limit).
    - slope (float): The slope of the line.
    - a (float): The y-intercept of the line.

    Returns:
    tuple: A tuple (p2) representing the calculated point (x, y).

    Example:
    >>> instance = YourClass()
    >>> result = instance.XCheck(2.5, (10, 5), 2, 3)
    >>> print(result)
    (2.5, 8)

    Note:
    - If x is within the range [0, Shp[1]], the y-coordinate is calculated based on the line equation.
    - If x is greater than Shp[1], the y-coordinate is calculated at the point (Shp[1], y2), where y2 is determined by the line equation.
    - If x is less than 0, the y-coordinate is calculated at the point (0, y2), where y2 is determined by the line equation.

    """
    if   x >= 0 and x <= Shp[1]:                           p2 = (x, Shp[0])
    elif x >= 0 and x >  Shp[1]: y2 = int(Shp[1]*slope+a); p2 = (Shp[1],y2)
    elif x <  0 and x <= Shp[1]: y2 = int(a);              p2 = (0,y2)
    return p2

def InclinedLine(self,P1, P2 = (), slope = None, imgShape = ()):
    """
    Generates the inclined line equation from two points or one point and slope.

    The image boundary/shape should be given.

    Parameters:
    - P1 (tuple): First point tuple (a1, b1).
    - P2 (tuple, optional): Second point tuple (a2, b2). Defaults to ().
    - slope (float, optional): Slope of the line. Defaults to None.
    - imgShape (tuple): Image size (y-length, x-length).

    Returns:
    tuple: A tuple containing:
    - first boundary point tuple.
    - second boundary point tuple.
    - line slope.
    - y-intercept.

    Example:
    >>> instance = YourClass()
    >>> result = instance.InclinedLine((0, 0), (2, 4), imgShape=(5, 5))
    >>> print(result)
    ((0, 0), (5, 5), 1.0, 0)

    Note:
    - If `imgShape` is not provided, the function prints an error message and aborts the program.
    - If only one point (`P1`) and slope (`slope`) are provided, the function calculates the second point.
    - If the line is not vertical or horizontal, it calculates the boundary points based on the image shape.
    - If the line is vertical, the slope is `np.inf`, and the function returns vertical boundary points.
    - If the line is horizontal, the slope is 0, and the function returns horizontal boundary points.

    """
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
        return (0,P1[1]), (imgShape[1],P1[1]), 0, P1[1]  

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
    >>> instance = YourClass()
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
    cv2.setMouseCallback(self.LineName[LineNameInd], extract_coordinates,prams)
    # Wait until user press some key
    cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
    return self.Reference