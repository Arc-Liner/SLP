import numpy as np
from pydmd import BOPDMD
from pydmd.plotter import plot_summary
from pydmd.preprocessing import zero_mean_preprocessing
import netCDF4 as nc

import matplotlib.pyplot as plt
from scipy.linalg import eig

data_filename = "C:/Users/ASUS/OneDrive - Indian Institute of Technology Bombay/Machine Learning/ML projects/SLP/data/3D_temp_data/ind_1991_temp_3D_250hPa.nc"
dataset = nc.Dataset(data_filename, 'r')
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]

# Extract temperature data, the shape will be [time, plevel, latitude, longitude]
temp_data = (dataset.variables['TMP_prl'][:])[:2000, :, 75:85, 50:60]
print(type(temp_data))
time_dim, channels, x_dim, y_dim = temp_data.shape

# Reshape the data for DMD: combine spatial dimensions into a single feature dimension
X = temp_data[:1000, ...].reshape(1000, channels * x_dim * y_dim).T  # Shape: (features, time)
X = X.filled(280)

t = np.linspace(0, 1,1000)

bopdmd = BOPDMD(
    svd_rank=6,                                  # Rank of the DMD fit.
    num_trials=100,                               # Number of bagging trials to perform.
    trial_size=0.5,                               # Use 50% of the total number of snapshots per trial.
    eig_constraints={"imag"},  # Eigenvalues must be imaginary and conjugate pairs.
    varpro_opts_dict={"tol":0.2, "verbose":True}, # Set convergence tolerance and use verbose updates.
)

# Fit the BOP-DMD model.
# X = (n, m) numpy array of time-varying snapshot data
# t = (m,) numpy array of times of data collection
bopdmd.fit(X, t)
plot_summary(bopdmd)

# Assuming X is the spatiotemporal matrix, and dt is the time step
X1 = X[:, :-1]  # All columns except the last
X2 = X[:, 1:]   # All columns except the first
dt = t[2] - t[1]

# Step 1: SVD decomposition
r = 1

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
t2 = np.linspace(0, 1, 1300)
# Step 5: Time dynamics reconstruction
print("Phi Shape:", Phi.shape)
print("Initial X Shape:", X[:, 0].shape)

### Compute the amplitudes

alpha1 = np.linalg.lstsq(Phi, X1[:, 0].reshape(-1,1), rcond=None)[0] ### DMD mode amplitudes

b = np.linalg.lstsq(Phi, X2[:, 0].reshape(-1,1), rcond=None)[0] ### DMD mode amplitudes

### DMD reconstruction
time_dynamics = None
for i in range(len(t2)):
    v = np.array(alpha1)[:,0]*np.exp(omega*(i+1)*dt)
    if time_dynamics is None:
        time_dynamics = v
    else:
        time_dynamics = np.vstack((time_dynamics, v))

X_dmd = np.dot( np.array(Phi), time_dynamics.T)

# X_0 = X[:, 0]
# b = (np.linalg.pinv(Phi) @ X_0.reshape(-1, 1))  # Initial condition projection
# time_dynamics = np.zeros((r, len(t2)), dtype=complex)
# print("omega shape:", omega.shape)
# print("b shape:", b.shape)

# for iter in range(len(t2)):
#     c = np.exp(omega * t2[iter])
#     time_dynamics[:, iter] = (b * c).reshape(-1, )

# X_dmd = Phi @ time_dynamics
print("X_dmd shape:", X_dmd.shape)
# Visualization 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
xgrid = np.arange(0, 1600, 1)
T, Xgrid = np.meshgrid(t2, xgrid)  # Assuming t and xgrid are defined
ax.plot_surface(Xgrid, T, np.real(X_dmd), cmap='viridis', edgecolor='none')
ax.set_xlabel("coordinates")
ax.set_ylabel("Time")
plt.title("Prediction using DMD")

X_actual = temp_data[:1300, ...].reshape(1300, channels * x_dim * y_dim).T
ax = fig.add_subplot(122, projection='3d')
# xgrid = np.arange(0, 1600, 1)
T2, Xgrid2 = np.meshgrid(t2, xgrid)  # Assuming t and xgrid are defined
ax.plot_surface(Xgrid2, T2, X_actual, cmap='viridis', edgecolor='none')
ax.set_xlabel("coordinates")
ax.set_ylabel("Time")
plt.title("Actual data")

mse = np.mean(np.sqrt((np.real(X_dmd) - X_actual) ** 2))
plt.suptitle(f"Actual vs Predicted Output Near to surface level(MSE:{mse})")
plt.show()



