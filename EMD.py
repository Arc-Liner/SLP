import numpy  as np
import pylab as plt
import netCDF4 as nc

from dtaidistance import dtw

from PyEMD import EMD

s1 = np.array([0.0, 0, 1, 2, 1, 0, 1, 0, 0])
s2 = np.array([0.0, 1, 2,1, 0, 1, 0, 0, 0])
d = dtw.distance_fast(s1, s2)
print(f"dtw:{d}")



data_filename = "C:/Users/ASUS/OneDrive - Indian Institute of Technology Bombay/Machine Learning/ML projects/SLP/data/3D_temp_data/ind_1991_temp_3D_250hPa.nc"
dataset = nc.Dataset(data_filename, 'r')
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]

# Extract temperature data, the shape will be [time, plevel, latitude, longitude]
data = (dataset.variables['TMP_prl'][:])[:100, :, 75:85, 50:60]

# Example 4D data with shape (time, channels, x, y)
time_dim, channels, x_dim, y_dim = data.shape

# Initialize EMD
emd = EMD()

# Placeholder for storing Intrinsic Mode Functions (IMFs)
imfs_data = np.zeros((100, time_dim, channels, x_dim, y_dim))  # 8 is arbitrary, can adjust based on # of modes found

for i in range(channels):
    for j in range(x_dim):
        for k in range(y_dim):
            signal = data[:, i, j, k]
            imfs = emd(signal)
            num_imfs = imfs.shape[0]
            imfs_data[:num_imfs:, :, i, j, k] = imfs  

# Check the shape of the resulting IMFs data (time, channels, x, y, modes)
print("IMFs shape:", imfs_data.shape)

# Set parameters for visualization
channel = 0  # Select the desired channel
x, y = 9, 9  # Specify a spatial location (example: x=10, y=10)

# Retrieve IMFs for the selected point
imfs_at_point = imfs_data[:10, : , channel, x, y]

# Define signal
t = np.linspace(0, 1, 200)
s = np.cos(11*2*np.pi*t*t) + 6*t*t

# Execute EMD on signal
IMF = EMD().emd(s,t)
print("IMF shape:",IMF.shape)
N = IMF.shape[0]+1
#N = imfs_at_point.shape[0]+1

# Plot results
plt.subplot(N,1,1)
plt.plot( t, s, 'r')
# plt.plot(data[:, channel, x, y], 'r')
#plt.title("Input signal")
plt.title("Input signal: $S(t)=cos(10\pi t) + 6t^2$")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.savefig('simple function 2 EMD')
plt.show()

time_step = 0

# Extract spatial data for the IMFs at the specified time step and channel
imfs_at_time = imfs_data[:, time_step, channel, :, :]  # Shape: (modes, x_dim, y_dim)
print("imfs_at_time:", imfs_at_time.shape)
num_modes = imfs_at_time.shape[0]
x_dim, y_dim = imfs_at_time.shape[1], imfs_at_time.shape[2]

# Generate X, Y grid for plotting
X, Y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))

# Plot each mode in 3D
"""
fig = plt.figure(figsize=(14, 3 * num_modes))
for mode in range(num_modes):
    ax = fig.add_subplot(num_modes, 1, mode + 1, projection='3d')
    ax.plot_surface(X, Y, imfs_at_time[mode, ...], cmap='viridis')
    ax.set_title(f"Mode {mode + 1} at Time {time_step}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Amplitude")

plt.suptitle(f"3D Plot of IMFs for Channel {channel} at Time {time_step}")
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust title position
plt.show()
"""
