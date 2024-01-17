# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 00:20:49 2023

@author: Ahmed H. Hanfy
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def plotting(FFT,y,Spectlocation):
    """
    Plot the magnitude spectrum of a Fourier-transformed image.

    Parameters:
    - FFT (numpy.ndarray): 2D NumPy array representing the Fourier-transformed image.
    - y (int): Height of the original image.
    - Spectlocation (list): List containing the location information for spectral analysis.

    Returns:
    - None

    Example:
    >>> plotting(FFT_image, 512, [0, 233])

    This function plots the magnitude spectrum of the Fourier-transformed image.
    The input 'FFT' is a 2D NumPy array representing the Fourier-transformed image.
    The 'y' parameter is the height of the original image.
    The 'Spectlocation' parameter is a list containing the location information for spectral analysis.

    The plot displays the magnitude spectrum using a logarithmic scale.
    The vertical axis is limited based on the height of the original image and the spectral location.

    .. note::
       This function assumes that the FFT input is a complex-valued array with shape (height, width, 2),
       where the last dimension represents the real and imaginary parts of the Fourier-transformed image.

    """
    MaxY = 0;
    for yloc in Spectlocation:
        if yloc[1] > MaxY: MaxY = yloc[1]
    
    fig, ax = plt.subplots(figsize=(30,20))
    FFT_im = 20*np.log(np.abs(FFT)+1)
    ax.imshow(FFT_im[:,:,0])
    ax.set_ylim([int(y/2)-20,int(y/2)+MaxY+147])


def Average(img):
    """
    Compute the average intensity profile across the width of an image and subtract it from each row.

    Parameters:
    - img (numpy.ndarray): Input image (grayscale or BGR).

    Returns:
    numpy.ndarray: New image with the average intensity profile subtracted from each row.

    Example:
    >>> import cv2
    >>> import numpy as np
    >>> image = cv2.imread('your_image.jpg')
    >>> result = Average(image)
    >>> cv2.imshow('Original Image', image)
    >>> cv2.imshow('Result Image', result)
    >>> cv2.waitKey(0)
    >>> cv2.destroyAllWindows()

    Note:
    - If the input image is in color (BGR), it will be converted to grayscale before processing.
    - The function computes the average intensity profile across the width of the image.
    - It then subtracts this average from each row to obtain a new image.
    """    
    # Convert image to grayscale if it is in color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    print('\t - subtracting Averaging ...', end=" ")
    width = len(img[0])
    Avg = np.zeros(width)
    for i in img: Avg += i
    Avg /= img.shape[0]
    Newimg = np.zeros(img.shape)
    for i in range(img.shape[0]):  Newimg[i] = img[i] - Avg
    maxValue = np.amax(Newimg); minValue = np.amin(Newimg)
    Newimg = np.around(((Newimg-minValue)/(maxValue-minValue))*255).astype(np.uint8)
    print(u'\u2713')
    return Newimg
    

def CleanIlluminationEffects(img, filterCenter = [0, 233], D = 10, n=10, ShowIm = False ,**kwargs):
    
    """
    Clean illumination effects from an image using a frequency domain approach.

    Parameters:
    - img (numpy.ndarray): Input image (grayscale or BGR).
    - filterCenter (list): Coordinates [x, y] of the filter center. Default: [0, 233].
    - D (float): Cut-off frequency for the low-pass filter. Default: 10.
    - n (int): Filter order. Default: 10.
    - ShowIm (bool): Whether to display intermediate images during processing. Default: False.
    - **kwargs: Additional keyword arguments for optional parameters.

    Returns:
    numpy.ndarray: Cleaned image.
    
    Example:
    >>> import cv2
    >>> import numpy as np
    >>> image = cv2.imread('your_image.jpg')
    >>> result = CleanIlluminationEffects(image, filterCenter=[0, 233], D=10, n=10, ShowIm=True)
    >>> cv2.imshow('Original Image', image)
    >>> cv2.imshow('Cleaned Image', result)
    >>> cv2.waitKey(0)
    >>> cv2.destroyAllWindows()
    
    Note:
    - If the input image is in color (BGR), it will be converted to grayscale before processing.
    - The function uses a frequency domain approach with a low-pass filter to remove illumination effects.
    - The filter parameters (filterCenter, D, n) can be adjusted to control the cleaning process.
    - If ShowIm is set to True, intermediate images will be displayed during processing.
    """
    
    # Handle optional parameter values from **kwargs
    filterCenter = kwargs.get('filterCenter', filterCenter)
    D = kwargs.get('D', D)
    n = kwargs.get('n', n)
    ShowIm = kwargs.get('ShowIm', ShowIm)
    
    # Convert image to grayscale if it is in color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    
    print('\t - Removing illumination instability', end=" ")
    
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    magnitude_spectrum = np.fft.fftshift(dft)
    y, x = magnitude_spectrum.shape[:2]

    if ShowIm: plotting(magnitude_spectrum,y,filterCenter)
    
    # Create a low-pass filter in the frequency domain
    LowpassFilter = np.ones((y, x, 2), dtype=np.float32)
    Filter = np.ones((y, x, 2), dtype=np.float32)
    for Center in filterCenter:       
        for i in range(y):
            for j in range(x):
                if i > y/2:
                        y_shift = int(y/2)+Center[1]
                        x_shift = int(x/2)+Center[0]
                        denominator = np.sqrt((i-y_shift)**2+(j-x_shift)**2)
                        if denominator <= 0: LowpassFilter[i][j] = 0
                        else: LowpassFilter[i][j]= 1/(1+(D/denominator)**(n*2))
                else: LowpassFilter[i][j]= 0
        Filter *= LowpassFilter
        print('.', end="")
    
        # Apply the low-pass filter to the magnitude spectrum
    CleanFFT = magnitude_spectrum*Filter
    
    # Display the cleaned spectrum if required
    if ShowIm: plotting(CleanFFT,y,filterCenter)
        
    # Compute the Inverse Discrete Fourier Transform (IDFT)
    f_ishift = np.fft.ifftshift(CleanFFT)
    img_back = cv2.idft(f_ishift)
    maxValue = np.amax(img_back[:,:,0]); minValue = np.amin(img_back[:,:,0])

    CleanedImage = np.around(((img_back[:,:,0]-minValue)/(maxValue-minValue))*255).astype(np.uint8)
    print(u' \u2713')
    return CleanedImage

def BrightnessAndContrast(img, Brightness = 1, Contrast = 1, Sharpness = 1, ShowIm =False, **kwargs):
    """
    Adjusts the brightness, contrast, and sharpness of an image.

    Parameters:
    - img (numpy.ndarray): NumPy array representing the image.
    - Brightness (float, optional): Brightness adjustment factor (default: 1).
      Valid range: 0 (min) to 2 (max).
    - Contrast (float, optional): Contrast adjustment factor (default: 1).
      Valid range: 0 (min) to 2 (max).
    - Sharpness (float, optional): Sharpness adjustment factor (default: 1).
      Valid range: 0 (min) to 3 (max).
    - **kwargs: Additional keyword arguments.

    Returns:
    - numpy.ndarray: NumPy array representing the adjusted image.

    Example:
    >>> adjusted_image = BrightnessAndContrast(image, Brightness=1.5, Contrast=1.2, Sharpness=2)

    This function adjusts the brightness, contrast, and sharpness of the input image.
    The `Brightness`, `Contrast`, and `Sharpness` parameters control the degree of adjustment.
    The image is converted to grayscale if it is in color. The adjusted image is returned as a NumPy array.

    If optional parameters are not provided, default values are used.

    The valid range for `Brightness` and `Contrast` is from 0 to 2, and for `Sharpness` is from 0 to 3.

    .. note::
       This function uses the Python Imaging Library (PIL) to perform the adjustments.

    .. seealso::
       For more information on PIL: https://pillow.readthedocs.io/en/stable/

    .. important::
       Ensure that the PIL library is installed in your Python environment.

    """
    
    # Handle optional parameter values from **kwargs
    Brightness = kwargs.get('Brightness', Brightness)
    Contrast = kwargs.get('Contrast', Contrast)
    Sharpness = kwargs.get('Sharpness', Sharpness)
    ShowIm = kwargs.get('ShowIm', ShowIm)

    # Validate parameter values
    Brightness = max(0, min(2, Brightness))
    Contrast = max(0, min(2, Contrast))
    Sharpness = max(0, min(3, Sharpness))
    
    # Convert image to grayscale if it is in color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    
    img = Image.fromarray(img, mode='L')    
    
    print('\t - Enhancing Image visability ...', end=" ")
    
    CorrectedImage = img.copy()
    if Brightness != 1:
        enhancer = ImageEnhance.Brightness(img)
        CorrectedImage = enhancer.enhance(Brightness)
        
    if Contrast != 1:
        enhancer = ImageEnhance.Contrast(CorrectedImage)
        CorrectedImage = enhancer.enhance(Contrast)
        
    if Sharpness != 1:
        enhancer = ImageEnhance.Sharpness(CorrectedImage)
        CorrectedImage = enhancer.enhance(Sharpness)
    print(u'\u2713')
    return np.array(CorrectedImage)
        
        
