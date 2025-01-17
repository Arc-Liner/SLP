import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import netCDF4 as nc
import matplotlib.pyplot as plt
from utilities3 import nanstd
from utilities3 import imf_gen

from graph_comp import compute_mutual_information

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################################
# 3d fourier layers
################################################################

def compl_mul3d(a, b):
    # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    assert a.device == b.device, "Tensors must be on the same device!"

    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

def hilbert_transform_3d(x):
    """Applies the Hilbert transform to the 3D Fourier coefficients."""
    # Get the size of the Fourier coefficients
    x_ft = torch.fft.rfft2(x, norm='ortho')

    batchsize, in_channels, dim_x, dim_y = x_ft.shape

    # Create a Hilbert mask to zero out the negative frequencies
    hilbert_mask = torch.zeros_like(x_ft)
    
    # In 3D, we need to handle multiple dimensions, so we zero out appropriate negative frequencies
    hilbert_mask[:, :, 0:dim_x//2+1, :] = 1  # Keep positive frequencies in the x-direction
    hilbert_mask[:, :, :, 0:dim_y//2+1] = 1  # Keep positive frequencies in the y-direction
    
    # Apply the mask to the Fourier coefficients
    x_hilbert_ft = x_ft * hilbert_mask
    
    # Return to physical space using irfft2
    x_processed = torch.fft.irfft2(x_hilbert_ft, s=(x.size(-2), x.size(-1)), norm='ortho').to(device)
    return x_processed

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2).to(device))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2).to(device))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2).to(device))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2).to(device))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients using rfft2 (for real-valued input)
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Convert the complex tensor to real-valued representation for compl_mul2d, basically separates real and imaginary part
        x_ft_real = torch.view_as_real(x_ft).to(device)

        # print(f"x_ft_real shape: {x_ft_real.shape}")  # It should have an extra dimension for real/imaginary parts

        # Allocate memory for the output Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=device)

        # Multiply relevant Fourier modes using real and imaginary parts
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul3d(x_ft_real[:, :, :self.modes1, :self.modes2], self.weights1.to(device))
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul3d(x_ft_real[:, :, -self.modes1:, :self.modes2], self.weights2.to(device))
        out_ft[:, :, :self.modes1, -self.modes2:] = compl_mul3d(x_ft_real[:, :, :self.modes1, :self.modes2], self.weights3.to(device))
        out_ft[:, :, -self.modes1:, -self.modes2:] = compl_mul3d(x_ft_real[:, :, -self.modes1:, :self.modes2], self.weights4.to(device))
        
        # Convert the result back to complex using view_as_complex for irfft2
        out_ft_complex = torch.view_as_complex(out_ft).to(device)

        # Return to physical space using irfft2
        x = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho').to(device)

        return x

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, in_ch, out_ch, width):
        super(SimpleBlock3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.fc0 = nn.Linear(self.in_ch, self.width).to(device)  #acts only on last dimension

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1).to(device)  #in_channels, out_channels, kernel
        self.w1 = nn.Conv2d(self.width, self.width, 1).to(device)
        self.w2 = nn.Conv2d(self.width, self.width, 1).to(device)
        self.w3 = nn.Conv2d(self.width, self.width, 1).to(device)
        
        self.bn = torch.nn.BatchNorm2d(self.in_ch).to(device)
        self.bn0 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn1 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn2 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn3 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn4 = torch.nn.BatchNorm2d(128).to(device)

        self.fc1 = nn.Linear(self.width, 128).to(device)
        self.fc2 = nn.Linear(128, out_ch).to(device)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        #x = self.bn(x)
        x = x.permute(0, 2, 3, 1) #since linear layer requires channels in the last dimension
        x = self.fc0(x)
        x = F.tanh(x)

        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        # x2 = self.w0(hilbert_transform_3d(x))
        x = x1
        x = self.bn0(x)
        x = F.tanh(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        #x = self.bn1(x1 + x2)
        x = F.tanh(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        #x = self.bn2(x1 + x2)
        x = F.tanh(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        #x = self.bn3(x1 + x2)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.bn4(x)
        # x = x.permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Net3d(nn.Module):
    def __init__(self, modes, width):
        super(Net3d, self).__init__()

        self.conv1 = SimpleBlock3d(modes, modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

################################################################
# Data loading
################################################################
data_filename = "C:/Users/ASUS/OneDrive - Indian Institute of Technology Bombay/Machine Learning/ML projects/SLP/data/3D_temp_data/ind_1991_temp_3D_250hPa.nc"
dataset = nc.Dataset(data_filename, 'r')
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]
# Extract temperature data, the shape will be [time, plevel, latitude, longitude]
# For near ocean 
temperature_data = (dataset.variables['TMP_prl'][:])[... , 75:85, 50:60]

# Creating boxed dataset for temperature
# boxed_temp = (dataset.variables['TMP_prl'][:])[... , 75:85, 50:60]
# for i in range(10):
#     if i==0 or i==9:
#         continue

#     else:
#         boxed_temp[... , i, 1:9] = 0

# Preparing dataset as pairs of consecutive time steps (t -> t+1)
class TemperatureDataset(Dataset):
    def __init__(self, data, eps = 1e-5, time_step=1):
        self.data = data # Data is a 4D tensor [time, plevel, lat, lon]
        self.time_step = time_step
        self.eps = eps
        # data_tensor = torch.tensor(np.ma.filled(self.data, np.nan),device=device)
        
        # # Compute mean and std over the plevel, lat, lon dimensions (ignoring time)
        # self.mean = torch.nanmean(data_tensor, dim = (0,1), keepdim=True)
        # self.std = torch.std(data_tensor, dim = (0,1), keepdim=True)
        
        # # Normalize the data
        # self.data = ((data_tensor - self.mean) / self.std).to(device)


    def __len__(self):
        return self.data.shape[0] - self.time_step

    def __getitem__(self, idx):
        # Return input data and the corresponding target (next time step)
        x = self.data[idx, ...]  # current time step
        x = torch.tensor(x, dtype=torch.float32,device=device)
        # x_mean = torch.nanmean(x, dim = 0, keepdim = True)
        # x_std = nanstd(x, dim = 0, keepdim = True)
        # x = (x - x_mean)/(x_std + self.eps)

        y = self.data[idx + self.time_step, ...]  # next time step (target)
        y = torch.tensor( y, dtype=torch.float32,device=device)
        # y_mean = torch.nanmean(y, dim= 0, keepdim = True)
        # y_std = nanstd(y, dim = 0, keepdim = True)
        # y = (y - y_mean)/(y_std + self.eps)        

        return x, y

# creating dataset
temp_dataset = TemperatureDataset(temperature_data)
data_tensor = torch.tensor(temperature_data,device=device)

# For IMF dataset
# imfs_data = imf_gen(temperature_data[:500, ...], device)
# print("imfs_data shape:", imfs_data.shape)
# temp_dataset = TemperatureDataset(imfs_data[ 0, ...])
# data_tensor = torch.tensor(imfs_data[ 0, ...],device=device)

# Splitting the dataset
train_size = int(0.85 * len(temp_dataset))
test_size = int(len(temp_dataset) - train_size)

# Normalized data
train_mean = torch.nanmean(data_tensor[:train_size, ...], dim = 1, keepdim=True)
train_std = torch.std(data_tensor[:train_size, ...], dim = 1, keepdim=True)
train_norm = (data_tensor[:train_size] - train_mean)/train_std
train_dataset = TemperatureDataset(train_norm)

# non normalized train data
# train_dataset = TemperatureDataset(data_tensor[:train_size], eps = 1e-3)

test_mean = torch.nanmean(data_tensor[train_size: , ...], dim = 1, keepdim=True)
test_std = torch.std(data_tensor[train_size:, ...], dim = 1, keepdim=True)
test_norm = (data_tensor[train_size:] - test_mean)/test_std
test_dataset = TemperatureDataset(test_norm) 

# mean and std for the predicted data :-
predicted_mean = torch.nanmean(data_tensor[train_size + 1: , ...], dim = 1, keepdim = True)
predicted_std = torch.std(data_tensor[train_size + 1: , ...], dim = 1, keepdim = True)

# non normalized test data
# test_dataset = TemperatureDataset(data_tensor[train_size:])

# Initialize dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle = True)

################################################################
# training and evaluation
################################################################
modes = 2
width = 20

learn_rate = 0.01
epochs = 40

"""
# Model without bypass layer 
model = SpectralConv3d_fast(in_channels=16, out_channels=16, modes1=4, modes2=4).to(device)
"""

model = SimpleBlock3d(modes1= modes, modes2 = modes, in_ch= 16, out_ch=16 ,width = width)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = lr_scheduler.LinearLR(optimizer, start_factor = 0.8, end_factor=learn_rate, total_iters=30)

#myloss = LpLoss(size_average=False)

def plot_loss(train_losses):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted(actual, predicted):
    plt.figure(figsize=(8,6))

    print("Mutual Info:",compute_mutual_information(actual[10, ...], predicted[10, ...]))
    print("ACC:",anomaly_correlation_coefficient(predicted, actual))
    # stds = predicted_std.to("cpu")

    # means = predicted_mean.to("cpu")

    # actual = torch.from_numpy(actual).to("cpu")
    # actual = (actual*stds) + means

    # predicted = torch.from_numpy(predicted).to("cpu")
    # predicted = (predicted*stds) + means

    # p_levels = np.arange(10, 2, -1)
    # levels = len(p_levels)
    # p_levels = torch.from_numpy(p_levels)

    for i in range(10):
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        X, Y = np.meshgrid(latitude[75:85], longitude[50:60])
        ax.plot_surface(X, Y, actual[10,i, ...], cmap='viridis')
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(X, Y, predicted[10, i, ...], cmap='viridis')
        plt.suptitle(f"Actual vs Predicted Output for P{10-i} level")
        
        # plt.subplot(1, 2, 1)
        # plt.plot(actual[i, :-2, 0, 0], p_levels)
        # plt.xlabel("Temperature")
        # plt.ylabel("Pressure Levels")
        # plt.title(f"Actual variation at {i}th timestep")

        # plt.subplot(1, 2, 2)
        # plt.plot(predicted[i, :-2, 0, 0], p_levels, color = "orange")
        # plt.xlabel("Temperature")
        # plt.ylabel("Pressure Levels")
        # plt.title(f"Predicted variation at {i}th timestep")

        # err_predict = torch.norm(actual[i, :-2, 0, 0] - predicted[i, :-2, 0 , 0])/levels
        # plt.suptitle(f"Error between actual and predicted : {err_predict}")
        plt.show()


def train_model(model, dataloader, epochs, optimizer, loss_fn, lambda_reg):
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            # Forward pass
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            # batch_y_mean = torch.nanmean(batch_y, dim = 1, keepdim=True)
            # batch_y_std = torch.std(batch_y, dim = 1, keepdim=True)
            # batch_y_norm = (batch_y - batch_y_mean)/batch_y_std
            loss = loss_fn(outputs, batch_y)
            # l2_norm = sum(p.abs().sum() for p in model.parameters())
            # loss += lambda_reg * l2_norm
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        torch.cuda.empty_cache()
    # Plot training loss after completion
    plot_loss(train_losses)

#Training
lambda_reg = 0.01
train_model(model, train_dataloader, epochs, optimizer, criterion, lambda_reg)

#Save the model
filepath = "FNO_3D.pth"
torch.save(model, filepath)

def evaluate_model(model, dataloader):
    model.eval()
    actuals = []
    predictions = []
    
    with torch.no_grad():
        avg_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            # batch_y_mean = torch.nanmean(batch_y, dim = 1, keepdim=True)
            # batch_y_std = torch.std(batch_y, dim = 1, keepdim=True)
            # batch_y_norm = (batch_y - batch_y_mean)/batch_y_std
            loss = criterion(outputs, batch_y)
            avg_loss += loss.item()
            # Store predictions and actuals for plotting
            actuals.append(batch_y.tolist())
            predictions.append(outputs.tolist())
        avg_loss = avg_loss/len(test_dataloader)
    # Convert lists to numpy arrays
    actuals = np.array(sum(actuals, []))
    predictions = np.array(sum(predictions, []))

    # Plot actual vs predicted values
    plot_actual_vs_predicted(actuals, predictions)
    print("Avg test loss", avg_loss)

def anomaly_correlation_coefficient(predicted, actual):
    # Calculate anomalies by subtracting the mean
    predicted_anomaly = predicted - np.mean(predicted)
    actual_anomaly = actual - np.mean(actual)
    
    # Calculate the numerator as the dot product of anomalies
    numerator = np.sum(predicted_anomaly * actual_anomaly)
    
    # Calculate the denominator as the product of the norms of the anomaly vectors
    denominator = np.sqrt(np.sum(predicted_anomaly ** 2) * np.sum(actual_anomaly ** 2))
    
    # Calculate ACC
    acc = numerator / denominator
    return acc

initial_data = data_tensor[train_size + 1, ...]
input = initial_data.reshape(1, 16, 10, 10)
acc = np.empty(100)

for i in range(100):
    output = model(input)
    input = output
    temp = np.array((output.cpu()).detach().numpy())
    acc[i] = anomaly_correlation_coefficient(temp, temperature_data[train_size + (i+1)])

iterations = np.arange(0, 100, 1)
plt.plot(iterations, acc)
plt.title("Autoregressive testing")
plt.xlabel("iterations")
plt.ylabel("ACC")
plt.show()
# Testing
#evaluate_model(model, test_dataloader)

# MI(2modes): 1.9402356669001055
# ACC(2 modes): 0.9813193344439611


