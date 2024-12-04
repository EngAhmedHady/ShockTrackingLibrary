# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:28:27 2024

@author: Ahmed H. Hanfy
"""
import random
from ..ShockOscillationAnalysis import BCOLOR

def genratingRandomNumberList(ShockAngleSamples: int, n1: int) -> list[int]:
    """
    Generate a list of random indices based on the given sample size.

    Parameters:
        - **ShockAngleSamples (int)**: The desired number of shock angle samples.
        - **n1 (int)**: The total number of files available.

    Returns:
        list:  A list of randomly selected indices.

    .. note ::
        - If `ShockAngleSamples` is greater than `n1`, it will be set to `n1`.
        - Prints a warning if `ShockAngleSamples` is adjusted to match `n1`.

    Example:
        >>> random_indices = genratingRandomNumberList(10, 15)
        >>> print(random_indices)
        [3, 7, 12, 1, 9, 4, 0, 8, 5, 2]

    """
    if n1 < ShockAngleSamples:
        ShockAngleSamples = n1
        warning = 'ShockAngleSamples should not be more than the number of files;'
        action = 'the number of files will be the only considered.'
        print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
        print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')
        

    randomIndx = random.sample(range(n1), min(ShockAngleSamples, n1))
    return randomIndx


def GenerateIndicesList(total_n_files:int, files:list[int,int]=[0,0], 
                        every_n_files:int=1) -> tuple[range, int]:
    """
    Generate a list of indices based on the specified range and step.

    Parameters:
        - **total_n_files (int)**: The total number of available files.
        - **files (list[int, int], optional)**: A list specifying the start and end files (default is [0, 0]).
        - **every_n_files (int, optional)**: Step value to determine the frequency of indices (default is 1).

    Returns:
        tuple[range, int]:  A tuple containing a range object of the indices and the total number of images.

    .. note ::
        - If `files` list is empty or contains values beyond the available range, defaults will be used.
        - If `files` is a list with two integers, it determines the start and end of the range.
        - If `files` is a single integer greater than zero, it determines the end of the range.
        - If `files` is greater than the total number of files, a warning is printed and `total_n_files` is used as the end.
        - The function calculates the number of images based on the specified range and step.

    Example:
        >>> indices, num_images = GenerateIndicesList(100, [10, 50], 5)
        >>> print(list(indices))
        [10, 15, 20, 25, 30, 35, 40, 45]
        >>> print(num_images)
        8
    """
    start_file = 0; end_file = total_n_files
    if hasattr(files, "__len__"):
        files.sort(); start, end = files
        if abs(end-start) > 0: 
            start_file = start 
            end_file = end
    elif files > 0: 
        if files <= total_n_files:  
            end_file = files
        else: 
            warning = 'Requested files are more than available files;'
            action = 'Only available files will be imported'
            print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}', end=' ')
            print(f'{BCOLOR.ITALIC}{warning} {action}{BCOLOR.ENDC}')

    n_images = int((end_file-start_file)/every_n_files)
    return range(start_file,end_file,every_n_files), n_images