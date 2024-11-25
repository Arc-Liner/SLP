import netCDF4 as nc
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.signal import hilbert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Open the .nc file
dataset = nc.Dataset('C:/Users/ASUS/OneDrive - Indian Institute of Technology Bombay/Machine Learning/ML projects/SLP/data/3D_temp_data/ind_1991_temp_3D_250hPa.nc', 'r')

# Extract the temperature data (TMP_prl), latitude, longitude, and pressure levels
temperature_data = (dataset.variables['TMP_prl'][:])[... , 75:85, 50:60]
data_tensor = torch.tensor(np.ma.filled(temperature_data, np.nan))

"""
boxed_temp = (dataset.variables['TMP_prl'][:])[... , 75:85, 50:60]
for i in range(10):
    if i==0 or i==9:
        continue

    else:
        boxed_temp[... , i, 1:9] = 0


data_tensor = torch.tensor(boxed_temp,device=device)
train_mean = torch.nanmean(data_tensor[:100], dim = 1, keepdim=True)
train_std = torch.std(data_tensor[:100], dim = 1, keepdim=True)
train_norm = (data_tensor[:100] - train_mean)/train_std
print(train_norm)
"""

"""
def hilbert_2d(matrix):
    # Apply the Hilbert transform along each axis
    hilbert_on_x = hilbert(matrix, axis=0)  # Hilbert transform along axis 0 (rows)
    hilbert_2d_result = hilbert(np.real(hilbert_on_x), axis=1)  # Hilbert transform along axis 1 (columns)
    return hilbert_2d_result

def plot_hilbert_2d(original, hilbert_transformed):
    # Create a 3D figure with three subplots
    fig = plt.figure(figsize=(18, 6))

    x = np.arange(original.shape[2])
    y = np.arange(original.shape[3])
    X, Y = np.meshgrid(x, y)

    # Plot the original signal (matrix)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, original, cmap='viridis')
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')
    ax1.set_zlabel('Amplitude')

    # Plot the real part of the Hilbert transform
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, np.real(hilbert_transformed), cmap='plasma')
    ax2.set_title('Hilbert Transform (Real Part)')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Rows')
    ax2.set_zlabel('Amplitude')

    # Plot the imaginary part of the Hilbert transform
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, np.imag(hilbert_transformed), cmap='inferno')
    ax3.set_title('Hilbert Transform (Imaginary Part)')
    ax3.set_xlabel('Columns')
    ax3.set_ylabel('Rows')
    ax3.set_zlabel('Amplitude')

    # Display the plots
    plt.tight_layout()
    plt.show()

hilbert_transformed = hilbert_2d(temperature_data[0, 0, ...])
print(np.real(hilbert_transformed.size()))
plot_hilbert_2d(temperature_data[0, 0, ...], hilbert_transformed)
"""

# Compute mean and std over the plevel, lat, lon dimensions (ignoring time)
temp_mean = torch.nanmean(data_tensor, dim = (0,1), keepdim=True)
#print(temperature_data)
#print(temp_mean)
# Shape: [time, plevel, latitude, longitude]
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]
# plevel = dataset.variables['plevel'][:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(latitude[75:85], longitude[50:60])
ax.plot_surface(X, Y, temperature_data[0,10,...], cmap='viridis')
plt.show()
#print("The temperature data shape:", temperature_data.shape)

# Initialize the movie
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(latitude[75:85], longitude[50:60])

for i in range(100):
    ax.plot_surface(X, Y, temperature_data[i, 15,...], cmap='viridis')
    plt.pause(1)
    plt.show()
    