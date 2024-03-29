# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:51:47 2022

@author: Hady
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
# import gc # ............................................. To clean the memory
import os
from matplotlib.patches import Arc
from sklearn.linear_model import LinearRegression
import imageio.v2 as imageio #.......................... gif generator liberary
import ctypes # ............................... To find the  monitor resolution 

plt.rcParams.update({'font.size': 30})
plt.rcParams["text.usetex"] =  False
plt.rcParams["font.family"] = "Times New Roman"
px = 1/plt.rcParams['figure.dpi']

LineName = ["First Reference Line (left)",
            "Second Reference Line (right)",
            "Horizontal Measuring Line",
            "Inclined Line"]

def XCheck(x,Shp,slope,a):
    if   x >= 0 and x <= Shp[1]:                           p2 = (x, Shp[0])
    elif x >= 0 and x >  Shp[1]: y2 = int(Shp[1]*slope+a); p2 = (Shp[1],y2)
    elif x <  0 and x <= Shp[1]: y2 = int(a);              p2 = (0,y2)
    return p2
        
def InclinedLine(P1,P2,imgShape):
  dx = P1[0]-P2[0]
  dy = P1[1]-P2[1]
  if  dy != 0 and  dx !=0:
      slope = dy/dx
      a = P1[1] - slope*P1[0]
      Xmax = int((imgShape[0]-a)/slope)
      Xmin = int(-a/slope)
      if   Xmin >= 0 and Xmin <= imgShape[1]:
          p1 = (Xmin,0)
          p2 = XCheck(Xmax,imgShape,slope,a)
      elif Xmin >= 0 and Xmin >  imgShape[1]:
          y = int(imgShape[1]*slope+a)
          p1 = (imgShape[1],y)
          p2 = XCheck(Xmax,imgShape,slope,a)
      else:
          y1 = int(a);
          p1 = (0,y1)
          p2 = XCheck(Xmax,imgShape,slope,a)
      return p1, p2, slope, a
  elif dx == 0:
      return (P1[0],0), (P1[0],imgShape[0]), 0, np.Inf
  else:
      return (0,P1[1]), (imgShape[1],P1[1]), 0, 0   

def extract_coordinates(event, x, y, flags, parameters): 
    global line_coordinates;
    global TempLine; global Reference
    global Temp;   global clone;
    global ClickCount
    # Record starting (x,y) coordinates on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        ClickCount += 1
        if len(TempLine) == 2: 
            line_coordinates = TempLine;
        elif len(TempLine) == 0: TempLine = [(x,y)]
        
    # Record ending (x,y) coordintes on left mouse bottom release
    elif event == cv2.EVENT_LBUTTONUP: 
        if len(TempLine) < 2:
            TempLine.append((x,y))
            print('Starting: {}, Ending: {}'.format(TempLine[0], TempLine[1]))
            
            # Draw temprary line
            cv2.line(Temp, TempLine[0], TempLine[1], (0,0,255), 2)
            if parameters[2] == 'V':
                avg = int((TempLine[0][0]+TempLine[1][0])/2)
                cv2.line(Temp, (avg,0), (avg,parameters[1]), (0,255,0), 1)
            elif parameters[2] == 'H':
                avg = int((TempLine[0][1]+TempLine[1][1])/2)
                cv2.line(Temp, (0,avg), (parameters[1],avg), (0,255,255), 1)
            elif parameters[2] == 'Inc':
                P1,P2,m,a = InclinedLine(TempLine[0],TempLine[1],parameters[1])
                cv2.line(Temp, P1, P2, (0,255,0), 1)

            cv2.imshow(parameters[0], Temp)
        
        elif ClickCount == 2:
            # storing the vertical line
            Temp = clone.copy()
            cv2.imshow(parameters[0], clone)
            if parameters[2] == 'V':
                avg = int((line_coordinates[0][0]+line_coordinates[1][0])/2)
                cv2.line(Temp, (avg,0), (avg,parameters[1]), (0,255,0), 1)
            elif parameters[2] == 'H':
                avg = int((line_coordinates[0][1]+line_coordinates[1][1])/2)
                cv2.line(Temp, (0,avg), (parameters[1],avg), (0,255,255), 1)
            elif parameters[2] == 'Inc':
                P1,P2,m,a = InclinedLine(line_coordinates[0],line_coordinates[1],parameters[1])
                cv2.line(Temp, P1, P2, (0,255,0), 1)
                avg = [P1, P2, m,a]
                
            Reference.append(avg)
            clone = Temp.copy()
            cv2.imshow(parameters[0], clone)
            
    # Delete draw line before storing    
    elif event == cv2.EVENT_RBUTTONDOWN:
        TempLine = []
        if ClickCount>0: ClickCount -= 1
        Temp = clone.copy()
        cv2.imshow(parameters[0], Temp)
        
                
def LineDraw(img, lineType, LineNameInd, Intialize = False):
    global line_coordinates;
    global TempLine; global Reference
    global Temp;   global clone;
    global ClickCount
    clone = img.copy(); 
    Temp = clone.copy();
    TempLine = [];
    ClickCount = 0
    if Intialize:
        Reference = []
        line_coordinates = []
    shp = img.shape
    if lineType == 'V':
        prams = [LineName[LineNameInd],shp[0],lineType]
    elif lineType == 'H':
        prams = [LineName[LineNameInd],shp[1],lineType]
    elif lineType == 'Inc':
        prams = [LineName[LineNameInd],shp,lineType]
        
    cv2.imshow(LineName[LineNameInd], clone)
    cv2.setMouseCallback(LineName[LineNameInd], extract_coordinates,prams)
    # Wait until user press some key
    cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
    return clone

def ImportSchlierenImages(path, ResizeImg = (), BGPath = '', nt = -1, Mode = -1):
# This function is importing a seuqnce of image to perform shock wave analysis
# after defining the analysis region, also clean the working images by extracting 
# the background image
# Importing steps: 1- define the horizontal upper and lower limits
# ................ 2- define the estimated shock line (inclined line)
# ................ 3- The function will import all files, clean them and prepare 
# ................... them for next step
# ..............................................................................
# Inputs: path      => image path for sequence 'Ex: "Directory/FileName*.img"'  
# ...................... of any image extensions, note: '*' referes to any
# ....... ResizeImg => in case it is require to resize the image in pexels,
# .................... integre tuple (width, hight), (Default: ()) means no changes
# ....... BGPath    => background image path that used for images cleaning
# ....... nt        => total number of images, means to take firest n images
# .................... in the path folder (Default: -1) means import all images
# ....... Mode      => take images that dividable on the 'Mode' good for sampling
# .................... from propsed period (Default: -1) means import all images
# Outputs: openCV image list, number of imported images, horizontal slice location on the image [pixels]
    
        img_list=[]
        originalImg_list=[]
        n = 0; o = 0
        global Reference; global clone;
        reductionRatio = 1
        
        files = sorted(glob.glob(path))            
        
        if len(files) > 1:   
            Refimg = cv2.imread(files[0])
            if len(ResizeImg) > 0: Refimg = cv2.resize(Refimg, ResizeImg)
            
            imgSize = Refimg.shape
            print(f'Images Dimensions: {imgSize}')
            
            user32 = ctypes.windll.user32
            screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
            print(f'Screen resolution: {screensize}')
            # if imgSize[0] >= screensize[1]*0.85:
            #     r = imgSize[0]/imgSize[1] # ---------- Image aspect ratio
            #     NewImgSize = (int(screensize[1]*0.85/r),int(screensize[1]*0.85))
            #     Refimg = cv2.resize(Refimg, NewImgSize)
            #     reductionRatio = NewImgSize[0]/imgSize[0]
            #     print(f'Image hieght is larger than your monitor hieght,\n Only reference image will be adjusted to {NewImgSize}')
                
            # img = cv2.resize(img, (1620,1080))
            # Drawing Horizontal limits of Analysis
            LineDraw(Refimg, 'H', 2, Intialize = True)
            LineDraw(clone, 'H', 2)
            # Drawing estimated shock inclination
            LineDraw(clone, 'Inc', 3)

            if len(Reference) < 3: 
                print('Insufficient Input lines!! Aborting ...')
                sys.exit() 
            
            if  nt == -1 and Mode == -1: n1 = len(files)    
            elif Mode > 0 and nt > 0: n1 = int(nt/Mode)
            elif Mode > 0 and nt < 0: n1 = int(len(files)/Mode)
            else: n1 = nt
            
            if len(BGPath) > 0:
                BG = cv2.imread(BGPath)
                BG = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY).astype(np.float32)
                BGlen = BG.shape[0]
                BGWid = BG.shape[1]
            
            for name in files:
                if o%Mode == 0 and n < n1:
                    with open(name):
                        img = cv2.imread(name)                            
                        if len(BGPath) > 0:
                            originalImg = img.copy()
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                            imglen = img.shape[0]; imgWid = img.shape[1]
                            
                            if BGlen < imglen: 
                                img = img[0:BGlen,0:BGWid]
                                originalImg = originalImg[0:BGlen,0:BGWid]
                            else: BG = BG[0:imglen,0:imgWid]
                            
                            
                            img = cv2.subtract(img,BG);
                            if len(ResizeImg) > 0: originalImg = cv2.resize(originalImg, ResizeImg)
                            originalImg_list.append(originalImg)
                            
                        if len(ResizeImg) > 0: img = cv2.resize(img, ResizeImg)
                        img_list.append(img)
                        
                        
                    n += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n1/20)), int(5*n/(n1/20))))
                o += 1
            print('')
        else:
            print('No files found!')
            sys.exit()
        return img_list,n,imgSize,reductionRatio,originalImg_list
    
def Average(imgList, n):
    NewImg = []
    shp = imgList[0].shape
    AvgImg = np.zeros(shp)
    for i in imgList:  
        GrayImg = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        AvgImg += GrayImg
    AvgImg /= n
    for i in imgList: NewImg.append(i - AvgImg)
    return NewImg

def ShockLoc(ImgSlice, plot = False):
    if plot:  fig, ax = plt.subplots(figsize=(10,5))
    xPixls = len(ImgSlice)
    avg = np.mean(ImgSlice)
    ImgSlice = np.array(ImgSlice)-avg
    if plot:
        ax.plot(ImgSlice);
        ax.plot([0,xPixls],[0,0],'-');
    # l = 0; 
    MinA = 0
    localmin = []; LocMinI =  []
    ShockRegion=[]
    for j in range(xPixls):
        if ImgSlice[j] < 0: 
            localmin.append(ImgSlice[j])
            LocMinI.append(j)
            if j == xPixls-1 and len(localmin) > 1: 
                if plot:  ax.fill_between(LocMinI, localmin,0 , alpha=0.5)
                A = np.trapz(localmin)
                
                if A < MinA: 
                    MinA = A;   ShockRegion = [LocMinI,localmin]
                localmin = []; LocMinI = []

        elif ImgSlice[j] >= 0 and len(localmin) > 1:
            if plot: ax.fill_between(LocMinI, localmin,0 , alpha=0.5)
            A = np.trapz(localmin)
            if A < MinA:
                MinA = A;   ShockRegion = [LocMinI,localmin]
            localmin = [];    LocMinI = []
        else: localmin = [];  LocMinI = []
    LocMinAvg = np.mean(ShockRegion[1])
    if plot: ax.plot([ShockRegion[0][0]-5,ShockRegion[0][-1]+5],[LocMinAvg,LocMinAvg],'-.r')
    localmin2 = []; LocMinI2 = []
    for j in range(xPixls):
         if ImgSlice[j] < LocMinAvg: 
            localmin2.append(ImgSlice[j])
            LocMinI2.append(j)
    ShockRegion = [LocMinI2,localmin2]
    if plot: ax.fill_between(ShockRegion[0], ShockRegion[1],0 , hatch='///') 
    
    shockLoc = [];
    for elment in range(len(ShockRegion[1])):
        if ShockRegion[1][elment] < LocMinAvg: shockLoc.append(ShockRegion[0][elment])
    
                
    minValue = min(ShockRegion[1])  
    minLoc = ShockRegion[1].index(minValue)
    minLoc = np.mean(shockLoc)
    if plot:
        ax.plot(ShockRegion[0][minLoc],minValue,'rx')
        ax.plot([minLoc,minLoc],[-125,125],'--')
   
    return minLoc
      
    
# imgPath = '2022_06_07\\Test 1\\*.JPG'
# imgPath ='E:\\Test\\Test 6 - 90deg from the ref\\*.png'

# Input Parameters ================================================================
# Snapshots -----------------------------------------------------------------------
# imgPath ='D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Fully Open\\2023-03-29_Sync_Oil_sch-fully open\\Oil-sch test 7 - half covered 25fps - 5%\\Oil-Sch test7\\*.png'
# imgPath ='D:\\TFAST\\TEAMAero experiments\\Philipp Cases, tripping\\2023_04_11\Fast Schlieren\\Test 7 - 180deg from the ref\\*.png'
imgPath ='D:\\TFAST\\TEAMAero experiments\\Roughness study\\Smooth profile (P1)\\2023_05_25\\2kHz_smooth P1 (Test 5)\\*.png'
# BGPath ='D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\Half Open\\2023_01_19\\Test 4-5\\Avg\\Avg4.JPG'
# imgPath ='D:\\TFAST\\TEAMAero experiments\\Inlet valve (Second campaign)\\2022_11_30\\*.jpg'
# imgPath ='C:\\Users\\Hady-PC\\Desktop\\PhD\\TFAST\\TEAMAero Experiments\\2023_04_24\\Rough-2kHz-4sec (test 5)\\*.png'
NSnapshots = 2000  # ---------------------- SnapShopts to import (-1 to import all)
# ResizeImg = (1620,1080)  # ----------- In case of need to change image resolution  
useOrgImg = False # ----------- Use Original image in case of subtracting Avg image
# Traking points ------------------------------------------------------------------
N_Slices = 4 # --------------------------------------------------- Number of points
SliceWidth = 120 # --------------------------------------- Shock Oscillation domain
SliceThickness = 9 # -------------------- Slice Thickness to average befor tracking
# Flow Parameters -----------------------------------------------------------------
InletM = 1.26 # -------------- Avg. measured inlflow mach number upstream the shock
# Plots Configuration ------------------------------------------------------------- 
AvgAngleYloc = 350 # Average Angle values Y-position in plot (est. from img. hight)
ArrowLen = 50 # -------------- Inflow Mach number in x direction Arrow length in px 
TextSize = 26 # ------------- Inflow Mach number and angle in x direction text size
M1Color  = 'orange' #  Inflow Mach number and angle in x direction text/lines color
M1dColor = 'maroon' # ----- Inflow Mach number and angle estimated from shock angle
# =================================================================================


# ----------------- Generate storing data folder ----------------------------------
Folders = imgPath.split("\\")
FileDirectory = ''
for i in range(len(Folders)-1): FileDirectory += (Folders[i]+'\\')
NewFileDirectory = os.path.join(FileDirectory, "Tracked_shock")
if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)
# ---------------------------------------------------------------------------------

# ----------------------- Import working images -----------------------------------
# ImgList, n, imgSize, reductionRatio, originalImg_list = ImportSchlierenImages(imgPath, ResizeImg)
ImgList, n, imgSize, reductionRatio, originalImg_list = ImportSchlierenImages(imgPath, nt = -1, Mode = 3)

if len(originalImg_list) > 0: useOrgImg = True 
print('Number of imported snapshoots: %s' %n)
   
P1 = (int(Reference[2][0][0]/reductionRatio), int(Reference[2][0][1]/reductionRatio))
P2 = (int(Reference[2][1][0]/reductionRatio), int(Reference[2][1][1]/reductionRatio))


print(Reference)
HalfSliceWidth = int(SliceWidth/2)
NewP1 = (P1[0]-HalfSliceWidth, P1[1])
NewP2 = (P2[0]-HalfSliceWidth, P2[1])
a2 = NewP1[1] - Reference[2][2]*NewP1[0]/reductionRatio
cv2.line(clone, NewP1, NewP2, (0,0,255), 1)
NewP3 = (P1[0]+HalfSliceWidth, P1[1])
NewP4 = (P2[0]+HalfSliceWidth, P2[1])
a3 = NewP3[1] - Reference[2][2]*NewP3[0]/reductionRatio
cv2.line(clone, NewP3, NewP4, (0,0,255), 1)
Pnts = np.linspace(Reference[0], Reference[1], N_Slices)
AvgAngleYloc = imgSize[0] - AvgAngleYloc
Ref = []
for i in Pnts: 
    y_i = int(i)
    x_i1 = int((i-a2)*reductionRatio/Reference[2][2])
    cv2.circle(clone, (x_i1,y_i), radius=3, color=(0, 0, 255), thickness=-1)
    x_i2 = int((i-a3)*reductionRatio/Reference[2][2])
    cv2.circle(clone, (x_i2,y_i), radius=3, color=(0, 0, 255), thickness=-1)
    Ref.append([x_i1,x_i2])

cv2.imwrite(NewFileDirectory+'\\AnalysisDomain-Points.jpg', clone)
cv2.imshow('Measuring Domain', clone)
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);

imList = []
l = 0
Alfa = np.arcsin(1/InletM)*180/np.pi

AvgAngleGlob= 0
for j in ImgList:
    fig, ax = plt.subplots(figsize=(int(imgSize[1]*1.5*px), int(imgSize[0]*1.5*px)))
    xLoc = []
    for i in range(N_Slices): 
        y_i = int(Pnts[i])
        x_i1 = Ref[i][0];x_i2 = Ref[i][1]
        if len(j.shape) > 2: Slice = np.zeros([1,SliceWidth,3])
        else: Slice = np.zeros([1,SliceWidth])
        # Slice = j[y_i-1:y_i, x_i1:x_i2]
        # 
        
        Ht = int(SliceThickness/2)  # Half Thickness
        for k in range(SliceThickness): Slice += j[y_i+k-(Ht+1):y_i-Ht+k,x_i1:x_i2]
        Slice /= SliceThickness
        if len(j.shape) > 2: Slice = np.array(cv2.cvtColor(Slice.astype('float32'), cv2.COLOR_BGR2GRAY))

        # print(Slice)
        # fig1, ax1 = plt.subplots(figsize=(10, 5))
        # ax1.plot(Slice[0])
        # xLoc.append(int(ShockLoc(Slice[0]))+Ref[i][0])
        # minLoc, certain, reason = SOA.ShockTraking(Slice, 
        #                                            LastShockLoc = -1, 
        #                                            Plot = False, 
        #                                            count = -1)
        
        xLoc.append(int(ShockLoc(Slice[0]))+Ref[i][0])
        
        if i > 0 and i < N_Slices-1:
            dx = xLoc[i]-xLoc[i-1]
            dy = Pnts[i]-Pnts[i-1]
            if  dy != 0 and  dx !=0:
                slope = dy/dx
                AngRad = np.arctan(slope)
                if AngRad < 0: AngDeg = AngRad*180/np.pi
                else: AngDeg = 180-(AngRad*180/np.pi)
                
                    
            # ShockTraking(SnapshotSlice, LastShockLoc = -1, Plot = False, count = -1)    
            DAlfa = (AngDeg + Alfa)*np.pi/180
            aOfAlfa = y_i - DAlfa*xLoc[i]
            y2 = DAlfa*(xLoc[i]+80)+aOfAlfa
            y3 = DAlfa*(xLoc[i]-120)+aOfAlfa
            # ax.plot([xLoc[i],xLoc[i]+80], [y_i,y2],'--', color = M1dColor ,lw = 1)
            
            if AngDeg > 90:
                arc1 = Arc((xLoc[i], Pnts[i]),50, 50, theta1=-abs(AngDeg), theta2=0, color = M1Color)
            else:
                arc1 = Arc((xLoc[i], Pnts[i]),50, 50, theta1=AngDeg, theta2=0, color = M1Color)   
                
            # ax.add_patch(arc1);            
            # ax.text(xLoc[i]+40 ,Pnts[i]-3 , str(abs(round(AngDeg,2)))+'$^o$',size = TextSize, color = M1Color);
            
            # arc1 = Arc((xLoc[i], Pnts[i]),100, 100, theta1=AngDeg, theta2=(AngDeg + Alfa), color = M1dColor)
            # ax.add_patch(arc1);
            # ax.text(xLoc[i]+60 ,Pnts[i]-(y3-Pnts[i])-5 , str(abs(round(Alfa,2)))+'$^o$',size = 25, color = M1dColor);
            
            # ax.plot([xLoc[i]-10,xLoc[i]+60], [y_i,y_i],'--',color = M1Color,ms = 15)
            # AngRad = abs(AngDeg)*np.pi/180
            # M1 = round(1/np.sin(AngRad),2)
            
            # ax.annotate(r'M$_1$ = '+str(M1), xy=(xLoc[i], Pnts[i]),  xycoords='data',color=M1Color, size = TextSize,
            #         xytext=(xLoc[i] - ArrowLen, Pnts[i]), arrowprops=dict(headwidth = 5, color=M1Color, width=2),
            #         horizontalalignment='right', verticalalignment='center')
            
            # ax.annotate(r'M$_1$ = '+str(M1), xy=(xLoc[i], Pnts[i]),  xycoords='data',color=M1Color, size = TextSize,
            #         xytext=(xLoc[i] - 100*(M1/InletM), Pnts[i]), arrowprops=dict(headwidth = 5, color=M1Color, width=2),
            #         horizontalalignment='right', verticalalignment='center')
            
            # M1d
            # ax.annotate(r'M$_{1d}$ = '+str(InletM), xy=(xLoc[i], y_i),  xycoords='data',color=M1dColor, size = TextSize,
            #         xytext=(xLoc[i] - 100, y3), arrowprops=dict(headwidth = 10, color=M1dColor, width=2),
            #         horizontalalignment='right', verticalalignment='center')
            
            # arcontalalignment='right', verticalalignment='center')
    
    
    # ax.imshow(cv2.cvtColor(j, cv2.COLOR_BGR2GRAY),cmap='gray');
    if useOrgImg: 
        if len(originalImg_list[l].shape) < 3: ax.imshow(cv2.cvtColor(originalImg_list[l]));
        else: ax.imshow(cv2.cvtColor(originalImg_list[l], cv2.COLOR_BGR2RGB));
    else: ax.imshow(cv2.cvtColor(j, cv2.COLOR_BGR2RGB).astype(np.uint8));
    ColumnXLoc = np.array(xLoc).reshape((-1, 1))
    model = LinearRegression().fit(ColumnXLoc, Pnts)
    r_sq = model.score(ColumnXLoc, Pnts)
    # print(r_sq)
    a = model.intercept_
    m = model.coef_[0]
    
    # AngReg = np.arctan(m)*180/np.pi
    
    # if m < 0:   AngReg = np.arctan(m)*180/np.pi
    # else: AngReg = 180 - np.arctan(m)*180/np.pi
    if m > 0:   
        AngReg = 180 - np.arctan(m)*180/np.pi
    elif m <= 0: 
        AngReg = abs(np.arctan(m)*180/np.pi)
    else:
        AngReg = 90
    
    AvgAngleGlob += AngReg
    
    ax.plot(xLoc, Pnts,'-o',color = M1Color ,ms = 10)
    
    
    if m != 0:
        Xmin = int((-a)/m); 
        Xmax = int((imgSize[1]-a)/m)
        X = int((AvgAngleYloc-a)/m);
        
        # ax.plot([Xmin,Xmax], [0,imgSize[1]],'-.w', lw = 3)
    
        # arc1 = Arc((X, AvgAngleYloc),80, 80, theta1=AngReg, theta2=0, color = 'w')
        # ax.add_patch(arc1);
        # ax.text(X+45 ,AvgAngleYloc-10 , str(abs(round(AngReg,2)))+'$^o$',size = 50, color = 'w');
        # ax.plot([X-10,X+200], [AvgAngleYloc,AvgAngleYloc],'--w',ms = 10, lw = 3)
        # ax.set_xlim(0,imgSize[1])
        # ax.set_ylim(imgSize[0],0)

    
    plt.close(fig)
    fig.savefig(NewFileDirectory +'\\'+str(f"{l:04d}")+'.png')
    
    imList.append(imageio.imread(NewFileDirectory+'\\'+str(f"{l:04d}")+'.png'))
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(l/(n/20)), int(5*l/(n/20))))
    sys.stdout.flush()
    l += 1

imageio.mimsave(NewFileDirectory +'\\movie.avi', imList, fps=60)
print('\n',abs(round(AvgAngleGlob/n,2)))