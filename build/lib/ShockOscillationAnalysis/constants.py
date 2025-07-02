# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 21:50:45 2025

author: Ahmed H. Hanfy
"""

class CVColor:
    """
    A class to represent common colors used in OpenCV.
    This class provides RGB tuples for a variety of commonly used colors.

    Supported colors:
       BLACK, WHITE, RED, GREEN, BLUE, GREENBLUE, YELLOW, CYAN, MAGENTA,
       FUCHSIPINK, GRAY, ORANGE

   """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    GREENBLUE = (255, 128, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    FUCHSIPINK = (255, 128, 255)
    GRAY = (128, 128, 128)
    ORANGE = (0, 128, 255)


class BCOLOR:  # For coloring the text in terminal
    """
    A class to represent ANSI escape sequences for coloring terminal text.
    This class provides various ANSI escape codes to color and style text in
    terminal output.

    Supported formats:
        - BGOKBLUE: background blue
        - BGOKCYAN: background cyan
        - OKCYAN: cyan text
        - BGOKGREEN: background green
        - OKGREEN: green text
        - WARNING: yellow background (warning)
        - FAIL: red text (fail)
        - ITALIC: italic text
        - UNDERLINE: underlined text
        - ENDC: reset all attributes.

    """
    BGOKBLUE = '\033[44m'
    BGOKCYAN = '\033[46m'
    OKCYAN = '\033[36m'
    BGOKGREEN = '\033[42m'
    OKGREEN = '\033[32m'
    WARNING = '\033[43m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


