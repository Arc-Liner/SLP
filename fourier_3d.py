import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import itertools

import netCDF4 as nc
import matplotlib.pyplot as plt
from utilities3 import *

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

        # Convert the complex tensor to real-valued representation for compl_mul2d
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
        self.fc0 = nn.Linear(16, self.width).to(device)  #acts only on last dimension

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2).to(device)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1).to(device)  #in_channels, out_channels, kernel
        self.w1 = nn.Conv2d(self.width, self.width, 1).to(device)
        self.w2 = nn.Conv2d(self.width, self.width, 1).to(device)
        self.w3 = nn.Conv2d(self.width, self.width, 1).to(device)
        
        self.bn0 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn1 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn2 = torch.nn.BatchNorm2d(self.width).to(device)
        self.bn3 = torch.nn.BatchNorm2d(self.width).to(device)


        self.fc1 = nn.Linear(self.width, 128).to(device)
        self.fc2 = nn.Linear(128, out_ch).to(device)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.fc2(x)
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

# Extract temperature data, the shape will be [time, plevel, latitude, longitude]
# For near ocean 
temperature_data = (dataset.variables['TMP_prl'][:])[... , :10, :10]

# Preparing dataset as pairs of consecutive time steps (t -> t+1)
class TemperatureDataset(Dataset):
    def __init__(self, data, time_step=1):
        self.data = data  # Data is a 4D tensor [time, plevel, lat, lon]
        self.time_step = time_step
        data_tensor = torch.tensor(np.ma.filled(self.data, np.nan),device=device)
        
        # Compute mean and std over the plevel, lat, lon dimensions (ignoring time)
        self.mean = torch.nanmean(data_tensor, dim = 1, keepdim=True)
        self.std = torch.std(data_tensor, dim = 1, keepdim=True)
        
        # Normalize the data
        self.data = ((data_tensor - self.mean) / self.std).to(device)


    def __len__(self):
        return self.data.shape[0] - self.time_step

    def __getitem__(self, idx):
        # Return input data and the corresponding target (next time step)
        x = self.data[idx]  # current time step
        y = self.data[idx + self.time_step, :10, ...]  # next time step (target)
        return torch.tensor(x, dtype=torch.float32,device=device), torch.tensor(y, dtype=torch.float32,device=device)

# creating dataset
temp_dataset = TemperatureDataset(temperature_data)

#Splitting the dataset
train_size = int(0.85 * len(temp_dataset))
test_size = int(len(temp_dataset) - train_size)
train_dataset = TemperatureDataset(temperature_data[:train_size])
test_dataset = TemperatureDataset(temperature_data[train_size:]) 

# Initialize dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle = True)

################################################################
# training and evaluation
################################################################
modes = 3
width = 20

learn_rate = 0.001
epochs = 40

"""
# Model without bypass layer 
model = SpectralConv3d_fast(in_channels=16, out_channels=16, modes1=4, modes2=4).to(device)
"""

model = SimpleBlock3d(modes1= modes, modes2 = modes, in_ch= 16, out_ch=10 ,width = width)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=learn_rate, total_iters=30)

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
    for i in range(5):
        plt.plot(actual.flatten()[100*i:100*(i+1)], label='Actual')
        plt.plot(predicted.flatten()[100*i:100*(i+1)], label='Predicted', linestyle='--')
        plt.title(f"Actual vs Predicted Output for {i}th P level")
        plt.xlabel('Samples')
        plt.ylabel('Temperature')
        plt.legend()
        plt.grid(True)
        plt.show()

def train_model(model, dataloader, epochs, optimizer, loss_fn):
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            # Forward pass
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
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
train_model(model, train_dataloader, epochs, optimizer, criterion)

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
    predicted_anomaly = predicted - torch.mean(predicted)
    actual_anomaly = actual - torch.mean(actual)
    
    # Calculate the numerator as the dot product of anomalies
    numerator = torch.sum(predicted_anomaly * actual_anomaly)
    
    # Calculate the denominator as the product of the norms of the anomaly vectors
    denominator = torch.sqrt(torch.sum(predicted_anomaly ** 2) * torch.sum(actual_anomaly ** 2))
    
    # Calculate ACC
    acc = numerator / denominator
    return acc

# After training the model, call evaluate_model to see actual vs predicted values.
evaluate_model(model, test_dataloader)




