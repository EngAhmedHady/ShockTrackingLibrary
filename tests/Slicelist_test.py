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
    SA = SliceListGenerator(f,D)
    
    # use GenerateSlicesArray function
    ShockwaveRegion ,n ,WR, Scale = SA.GenerateSlicesArray(imgPath,
                                                           scale_pixels = True,
                                                           slice_loc = 10,            # as scaled tracking reference values in mm
                                                           full_img_width = False,    # to crop the slices by vertical reference line 
                                                           slice_thickness = 60,      # always in pixels
                                                           shock_angle_samples = 33,  # number of samples to determine the average inclination
                                                           angle_samples_review = 3, # to preview the tracked points during angle determination
                                                           inclination_est_info =  [110, (474, 591),(463, 482)], # information for angle determination
                                                           preview =  True,           # to preview the final setup before proceeding
                                                           output_directory = output_directory, # the directory where the slice list will be stored
                                                           comment='-SliceList',      # additional comments to the stored slice list file name
                                                           )