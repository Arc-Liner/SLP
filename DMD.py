import numpy as np
from pydmd import DMD
from pydmd.preprocessing import zero_mean_preprocessing
import netCDF4 as nc

from scipy.linalg import eig

data_filename = "C:/Users/ASUS/OneDrive - Indian Institute of Technology Bombay/Machine Learning/ML projects/SLP/data/3D_temp_data/ind_1991_temp_3D_250hPa.nc"
dataset = nc.Dataset(data_filename, 'r')
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]

# Extract temperature data, the shape will be [time, plevel, latitude, longitude]
temp_data = (dataset.variables['TMP_prl'][:])[... , 75:85, 50:60]

time_dim, channels, x_dim, y_dim = temp_data.shape

# Reshape the data for DMD: combine spatial dimensions into a single feature dimension
X = temp_data.reshape(time_dim, channels * x_dim * y_dim).T  # Shape: (features, time)
X = X.filled(280)

t = np.linspace(0, 2920*3, 3)

# Assuming X is the spatiotemporal matrix, and dt is the time step
X1 = X[:, :-1]  # All columns except the last
X2 = X[:, 1:]   # All columns except the first
dt = 3

# Step 1: SVD decomposition
r = 2

U, S, Vh = np.linalg.svd(X1, full_matrices=False)
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vh.T[:, :r]

# Step 2: Compute A_tilde
Atilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)

# Step 3: Eigen decomposition of A_tilde
D, W = np.linalg.eig(Atilde)
Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W  # DMD modes

# Step 4: Compute lambda and omega
lambda_vals = np.diag(D)
omega = np.log(lambda_vals) / dt

# Step 5: Time dynamics reconstruction
print("Phi Shape:", Phi.shape)
print("Initial X Shape:", X[:, 0].shape)
X_0 = X[:, 0]
b = np.linalg.pinv(Phi) @ X_0.reshape(-1, 1)  # Initial condition projection
time_dynamics = np.zeros((r, len(t)), dtype=complex)
print("omega shape:", omega.shape)
print("b shape:", b.shape)
c = np.exp(omega * (t[0].reshape(-1,1)))
for iter in range(len(t)):
    time_dynamics[:, iter] = b * (np.exp(omega * (t[iter].reshape(-1,1))))

X_dmd = Phi @ time_dynamics

# Visualization 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# T, Xgrid = np.meshgrid(t, xgrid)  # Assuming t and xgrid are defined
# ax.plot_surface(Xgrid, T, np.real(X_dmd).T, cmap='viridis', edgecolor='none')
# plt.show()

