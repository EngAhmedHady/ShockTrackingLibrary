# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:51:35 2024

@author: Ahmed H. Hanfy
"""
from ShockOscillationAnalysis import SliceListGenerator

if __name__ == '__main__':
    # Define the snapshots path with glob[note the extention of imported files]
    imgPath = r'test_files\raw_images\*.png'
    
    f = 2000    # images sampling rate
    D = 80      # distance in mm
    
    output_directory = r'results'
    
    # iniate the SliceListGenerator module
    SA = SliceListGenerator(f, D)
    
    # use GenerateSlicesArray function
    ShockwaveRegion ,n ,WR, Scale = SA.GenerateSlicesArray(imgPath,
                                                           scale_pixels=True,
                                                           # as scaled tracking reference values in mm
                                                           slice_loc=10,
                                                           # to crop the slices by vertical reference line
                                                           full_img_width=False,
                                                           # in pixels
                                                           slice_thickness=60,
                                                           # number of samples to determine the average inclination
                                                           shock_angle_samples=33,
                                                           # to preview the tracked points during angle determination
                                                           angle_samples_review=3,
                                                           # information for angle determination
                                                           inclination_est_info=[110, (474, 591), (463, 482)],
                                                           # to preview the final setup before proceeding
                                                           preview=True,
                                                           # the directory where the slice list will be stored
                                                           output_directory=output_directory,
                                                           # additional comments to the stored slice list file name
                                                           comment='-SliceList',
                                                           )