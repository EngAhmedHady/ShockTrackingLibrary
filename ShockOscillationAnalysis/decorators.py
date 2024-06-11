# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:49:58 2024

@author: Ahmed H. Hanfy
"""
import time

def TimeCalculation(timeInSec: float):
        """
        Convert the given time in seconds into a formatted string representation.

        Parameters:
            - **timeInSec (float)**: The time duration in seconds.

        Returns:
            None

        Example:
            >>> instance = SOA()
            >>> instance.TimeCalculation(3665)

        .. note ::
            - The function converts the time duration into hours, minutes, and seconds.
            - It prints the total run time in a human-readable format.

        """
        if timeInSec > 3600:
            timeInHr = timeInSec // 3600
            timeInMin = (timeInSec % 3600) // 60
            sec = (timeInSec % 3600) % 60
            print("Processing time: %s Hr, %s Min, %s Sec" % (round(timeInHr), round(timeInMin), round(sec)))
        elif timeInSec > 60:
            timeInMin = timeInSec // 60
            sec = timeInSec % 60
            print("Processing time: %s Min, %s Sec" % (round(timeInMin), round(sec)))
        else:
            print("Processing time: %s Sec" % round(timeInSec))
    
def calculate_running_time(func):
    """
    Decorator to calculate the running time of a function.

    This decorator calculates the running time of a function by measuring the time taken to execute it.
    It prints the running time to the console using the TimeCalculation function.

    Parameters:
         **func (function)**: The function to be decorated.

    Returns:
        function: The wrapped function.

    Examples:
        >>> @calculate_running_time
        ... def my_function():
        ...     # function body
        ...     pass
    
        In this example, the running time of the `my_function` will be calculated and printed to the console.

    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        TimeCalculation(end_time - start_time)
        return result
    return wrapper