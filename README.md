# STEPUP_image_analysis_II 

<img src="https://github.com/helenarichie/helenarichie/blob/master/images/STEPUP_logo.png" alt="logo" width="200"/>


STEPUP Image Analysis (SIA) is a pipeline written for easy photometric analysis to extact light curves from image data using differential aperture photometry. SIA runs in three main (independent) steps: instrument signature removal (**ISR**), generating plate solutions (**ASTROM**), and differential aperture photometry (**PHOT**).

This respository contains the SIA pipeline and a user manual written explicity for use by University of Pittsburgh students that are either members of the STEPUP team or students in ASTRON 1263. The guide can also be applied for general use, omitting location-specific steps.

## Installation
To run SIA_II, user must have Python 3 with standard libraries and the following Python packages: 
- Matplotlib
- NumPy
- AstroPy
- PhotUtils

These packages are all available to install using `pip`. Additionally, the user must install and compile the [WCSTools software package](http://tdc-www.harvard.edu/wcstools/) and add the executables to their device's `PATH`.

## Outside Resources
SIA_II uses [Astrometry.net](http://astrometry.net/) and [WCSTools software package](http://tdc-www.harvard.edu/wcstools/) to plate solve inputted datasets.
