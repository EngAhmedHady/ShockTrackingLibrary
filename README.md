[![DOI](https://zenodo.org/badge/711672788.svg)](https://zenodo.org/doi/10.5281/zenodo.11197727) [![Documentation Status](https://readthedocs.org/projects/shocktrackinglibrary/badge/?version=latest)](https://shocktrackinglibrary.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/EngAhmedHady/ShockTrackingLibrary/blob/main/LICENSE)


# Shock Tracking Library
The instability of shock waves due to induced separation presents a significant challenge in aerodynamics. Accurately predicting shock wave instability is crucial for reducing vibrations and noise generation. The high-speed schlieren technique, valued for its simplicity, affordability, and non-intrusiveness, is crucial for understanding the flow patterns in linear cascades. This Python package introduces an advanced method that employs line-scanning to detect and track shock waves from a large series of schlieren images. This method provides a feedback system to handle uncertainties in shock detection, with potential applications in supervised learning and AI. It proves effective in identifying and analyzing different types of shocks, even in images with low resolution or visibility. The method's performance was tested on a transonic fan passage test section in two case studies: one involving various Reynolds number conditions, and the other focusing on low Reynolds numbers, where shock instability is more prominent. The shock testing details can be found in this publication **Hanfy, A. H., Flaszyński, P., Kaczyński, P., & Doerffer, P., Advancements in Shock-Wave Analysis and Tracking from Schlieren Imaging**. DOI: [10.2139/ssrn.4797840 ](https://dx.doi.org/10.2139/ssrn.4797840)
![SnapShotsLE](https://github.com/EngAhmedHady/ShockTrackingLibrary/assets/49737863/875e0b51-e5dd-4c3e-8716-c2a92d39ce3b) This library employes OpenCV, scipy, glob, sys, numpy and matplotlib libraries.

## Installation

To install **Shock Tracking Liberary** from pip you can use: <br>
``pip install ShockOscillationAnalysis``

Alternatively, you can also clone the repository manually by running: <br>
``git clone https://github.com/EngAhmedHady/ShockTrackingLibrary.git`` 

Then install the package using: <br>
``pip3 install dist\ShockOscillationAnalysis-2.0.0-py3-none-any.whl``

