import numpy as np
from pydmd import BOPDMD
from pydmd.plotter import plot_summary
from pydmd.preprocessing import zero_mean_preprocessing
import netCDF4 as nc

import matplotlib.pyplot as plt
from scipy.linalg import eig

data_filename = "C:/Users/ASUS/OneDrive - Indian Institute of Technology Bombay/Machine Learning/ML projects/SLP/data/3D_temp_data/ind_1991_temp_3D_250hPa.nc"
dataset = nc.Dataset(data_filename, 'r')
print(dataset.variables)