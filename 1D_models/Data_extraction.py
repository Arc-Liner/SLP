import numpy as np
import pandas as pd
import netCDF4 as NC
import os
import glob
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
#from numpy.fft import fft, ifft
from scipy.signal import welch
from scipy.fftpack import fft, ifft
from scipy.fftpack import fftfreq

path = os.getcwd()
path=os.path.join(path,"data")
temporary = os.listdir(path)
data_files = glob.glob(path+os.sep+'hyd_temp_*.nc')
print(data_files)

A=[]
U=[]

time = np.empty([0])
Temperature_values_loc = np.empty([0])
a_smallest = 321768
u_smallest = 321768

for i in range(len(data_files)):
    Data_filename = NC.Dataset(data_files[i],'r')
    time = np.append(time, Data_filename['time'])
    # print(Data_filename['plevel'][0])
    if i%2==0:
      #Temperature_values_loc_a = np.append(Temperature_values_loc_a, Data_filename['TMP_prl'][:,0,1,1])
      #time_a = np.append(time_a, Data_filename['time'])
      A.append(Data_filename['TMP_prl'][:,0,1,1])

      a_smallest = min(a_smallest,len(A[-1]))

    else:
      # Temperature_values_loc_u = np.append(Temperature_values_loc_u, Data_filename['TMP_prl'][:,0,1,1])
      # time_u = np.append(time_u, Data_filename['time'])
      U.append(Data_filename['TMP_prl'][:,0,1,1])

      u_smallest = min(u_smallest, len(U[-1])) 

#temp_data = list(zip(time,Temperature_values_loc))
#temp_df=pd.DataFrame(temp_data ,columns=["time","Temperature"])

for i in range(8):
      A[i]=A[i][:a_smallest]
      U[i]=U[i][:u_smallest]

# for month:-
a_month=[]
u_month=[]

A=np.array(A)
U=np.array(U)

temp1=A[:][:2880]
temp2=U[:][:2880]

a_num=0
u_num=0

for i in range(16):
  if i%2==0:
    for j in range(12):
      if j%2==0:
        a_month.append(temp1[a_num][240*j:240*(j+1)])
            
      else:
        u_month.append(temp1[a_num][240*j:240*(j+1)])
      
    a_num=a_num + 1

  else:
    for j in range(12):
      if j%2==0:
        a_month.append(temp2[u_num][240*j:240*(j+1)])
    
      else:
        u_month.append(temp2[u_num][240*j:240*(j+1)])

    u_num=u_num+1

a_month=np.array(a_month)
u_month=np.array(u_month)

# for day:
a_day=[]
u_day=[]

a_num=0
u_num=0

# We will loop through each year and through each month, assuming each month has 30 days
for i in range(192): # Since we are looping through the yearly data having 192 samples 
  if i%2==0:
    for j in range(30):
      if j%2==0:
        a_day.append(a_month[a_num][8*j:8*(j+1)])
            
      else:
        u_day.append(u_month[a_num][8*j:8*(j+1)])
      
    a_num=a_num + 1

  else:
    for j in range(30):
      if j%2==0:
        a_day.append(a_month[u_num][8*j:8*(j+1)])
    
      else:
        u_day.append(u_month[u_num][8*j:8*(j+1)])

    u_num=u_num+1

a_day=np.array(a_day)
u_day=np.array(u_day)

print(time.shape)