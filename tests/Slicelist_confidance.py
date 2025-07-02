# -*- coding: utf-8 -*-
"""
Created on Tue Jul 01 09:51:35 2025

@author: Ahmed H. Hanfy
"""
from ShockOscillationAnalysis import SliceListGenerator

if __name__ == '__main__':
    # Define the snapshots path with glob[note the extention of imported files]
    imgPath = r'test_files\raw_images\*.png'
    
    f = 2000    # images sampling rate
    D = 80      # distance in mm
    
    output_directory = r'results\Slicelist_confidance-results'
    
    # iniate the SliceListGenerator module
    SA = SliceListGenerator(f, D)
    
    # use GenerateSlicesArray function
    ShockwaveRegion ,n ,WR, Scale = SA.GenerateSlicesArray(imgPath,
                                                           # Define the reference line locations
                                                           Ref_x0=[109, 726], Ref_y0=617,
                                                           # Define the range of images to be only imported 
                                                           within_range = [50,200],
                                                           scale_pixels=True,
                                                           # as scaled tracking reference values in mm
                                                           slice_loc=10,
                                                           # to crop the slices by vertical reference line
                                                           full_img_width=False,
                                                           # in mm
                                                           slice_thickness=[5, 'mm'],
                                                           # Define shock angle tracking vertical range in mm
                                                           sat_vr=[-5, 8, 'mm'],
                                                           # Define confidence interval with 98% accuracy
                                                           conf_interval=0.98,
                                                           # number of samples to determine the average inclination
                                                           shock_angle_samples=33,
                                                           # Define the number of points to track for angle estimation
                                                           nPnts = 15,
                                                           # to preview the tracked points during angle determination
                                                           angle_samples_review=3,
                                                           # information for angle determination
                                                           inclination_est_info=[110, (474, 591), (463, 482)],
                                                           # to preview the final setup before proceeding
                                                           preview=True,
                                                           # display properties
                                                           avg_preview_mode='avg_ang',
                                                           points_opacity=0.5,
                                                           points_size=5,
                                                           # the directory where the slice list will be stored
                                                           output_directory=output_directory,
                                                           # additional comments to the stored slice list file name
                                                           comment='-SliceList',
                                                           )