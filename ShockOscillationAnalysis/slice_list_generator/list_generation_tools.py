# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:28:27 2024

@author: Ahmed H. Hanfy
"""
import random
from ..ShockOscillationAnalysis import BCOLOR

def genratingRandomNumberList(ShockAngleSamples, n1):
    if n1 < ShockAngleSamples:
        ShockAngleSamples = n1
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC} ShockAngleSamples should not be more than the number of files; the number of files will be the only considered.{BCOLOR.ENDC}')

    randomIndx = random.sample(range(n1), min(ShockAngleSamples, n1))
    return randomIndx


def GenerateIndicesList(total_n_files, files = [0,0], every_n_files = 1):
    start_file = 0; end_file = total_n_files
    if hasattr(files, "__len__"):
        files.sort(); start, end = files
        if abs(end-start) > 0: start_file = start; end_file = end;
    elif files > 0: 
        if files <= total_n_files:  end_file = files
        else: print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC}Requested files are more than available files, only available files will be imported{BCOLOR.ENDC}')

    n_images = int((end_file-start_file)/every_n_files)
    return range(start_file,end_file,every_n_files), n_images