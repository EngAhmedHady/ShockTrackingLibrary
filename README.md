[![DOI](https://zenodo.org/badge/711672788.svg)](https://zenodo.org/doi/10.5281/zenodo.11197727) [![Documentation Status](https://readthedocs.org/projects/shocktrackinglibrary/badge/?version=latest)](https://shocktrackinglibrary.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/EngAhmedHady/ShockTrackingLibrary/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/ShockOscillationAnalysis)](https://pypi.org/project/ShockOscillationAnalysis)



# Shock Tracking Library

## Overview

The instability of shock waves due to induced separation presents a significant challenge in aerodynamics. Accurately predicting shock wave instability is crucial for reducing vibrations and noise generation. The high-speed schlieren technique, valued for its simplicity, affordability, and non-intrusiveness, is crucial for understanding flow patterns in linear cascades.

This Python package introduces an advanced method that employs **line-scanning** to detect and track shock waves from large series of schlieren images. It includes an **adaptive feedback system** to handle uncertainties in shock detection and is compatible with supervised learning and AI workflows. The method is capable of identifying and analyzing different types of shocks, even in low-resolution or visually degraded images.

The method's performance has been validated in a **transonic fan passage test section** and a **supercritical A320 wing profile** under varying Reynolds numbers and oscillation conditions.

> 📄 For scientific details and benchmarking, please refer to the article:  
> **"Advancements in Shock-Wave Analysis and Tracking from Schlieren Imaging"**  
> DOI: [10.2139/ssrn.4797840](https://dx.doi.org/10.2139/ssrn.4797840) and [10.1007/s00348-021-03145-3](https://doi.org/10.1007/s00348-021-03145-3)

---
<img src="https://github.com/user-attachments/assets/0a4532af-44c4-4bd1-bac2-01925d841ce4" alt="drawing" style="width:360px;"/> <img src="https://github.com/user-attachments/assets/66e4c5a9-24ac-4e09-b2e6-ee5f222929b0" alt="drawing" style="width:640px;"/>

## 🚀 Key Features

- **Three robust shock tracking methods**:  
  - `integral`: Based on maximum blocked light intensity.  
  - `darkest_spot`: Tracks absolute minimum intensity.  
  - `maxGrad`: Uses Sobel gradient to locate shock edge.

- **Adaptive outlier detection** using RANSAC and Tukey's fences.

- **Confidence estimation** via t-distribution and standard error for shock angle.

- **Weighted vs. arithmetic averaging** for better estimation accuracy.

- **Automatic feedback system** for detecting uncertain shock positions.

- **Visual tools** for tracking and comparing shock signals.

---

## 🔄 What's New in This Release

### 🔧 Code Enhancements
- Improved code style according to **PEP 8**
- Defined universal units:  
  ```python
  SOA.univ_unit = {'freq': 'fps', 'dis': 'mm', 'angle': 'deg'}

- Abort preview with `ESC` or continue with any key.

- Logging all tracking activities using:

    ```python 
    SOA.log(log_message: str, directory_path: str)

### Improved generating slice list array ``sliceListGenerator.GenerateSliceArray``:

- Define number of tracking points: ``npnts=n``

- Slice thickness in pixels or universal units: ``slice_thickness =[5.5, 'mm']``

- Custom vertical range: ``sat_vr = [-5.5, 3, 'mm']``

- avg_shock_loc now uses coordinate tuple (x, y)

### Improved shock angle estimation ``inc_tracking.InclinedShockTracking``

- RANSAC for better fitting

- Enhanced cached metadata and filename comments

- Confidence estimation: ``conf_interval=0.95``

- Histogram plotting with confidence stats

- Weighted average using slope std and error

### Visualization Upgrades

- Custom point size: ``points_size=12``

- Confidence/prediction bands:

    - ``conf_color``, ``conf_range_opacity``

    - ``pred_color``, ``pred_range_opacity``

- Custom background image:

    ``op_bg_path``, ``bg_y_crop``, ``bg_x_crop``, ``bg_resize``, ``bg_90rotate``

- Rotate output: ``op_90rotate=True``
- Use ``nReview``as an integer or tuple: (start, end, step) - applicable only with ``inc_tracking.ShockPointsTracking``

### 📦 Output Format Improvements

- *avg_shock_angle*:

    [arith_avg, arith_conf, weight_avg, weight_conf, std_dev]

- *avg_shock_loc*:

    [loc_avg, loc_conf, std_dev]


### 🛠️ Function Optimizations

Improved: ``SOA.extract_coordinates``, ``v_least_squares``

### Robust File Handling

Crop X: ``crop_x_img``, Crop Y: ``crop_y_img``, Resize: ``resize_img``

### 🐞 Bug Fixes

- Handle images without scale in ``sliceListGenerator.GenerateSliceArray``

- Fix corner cases in ``v_least_squares``

- Robust angle estimation even with missing slices ``ShockTraking``

- Fixed circular import issue using `constent.py`

- Also, ``InclinedLine``, ``AvgAnglePlot``, ``InclinedShockDomainSetup``


## Installation

To install **Shock Tracking Liberary** from pip you can use: <br>
``pip install ShockOscillationAnalysis``

Alternatively, you can also clone the repository manually by running: <br>
``git clone https://github.com/EngAhmedHady/ShockTrackingLibrary.git`` 

Then install the package using: <br>
``pip3 install dist\ShockOscillationAnalysis-2.15.8-py3-none-any.whl``

