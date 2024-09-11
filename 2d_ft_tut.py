import numpy as np
import matplotlib.pyplot as plt

# Create a 2D signal (image with sinusoidal pattern)
x = np.linspace(0, 4 * np.pi, 100)
y = np.linspace(0, 4 * np.pi, 100)
X, Y = np.meshgrid(x, y)
signal_2d = np.sin(X) + np.sin(Y)

# Perform 2D Fourier Transform
signal_2d_ft = np.fft.fftshift(np.fft.fft2(signal_2d))

# Plot the original signal and its Fourier Transform
plt.figure(figsize=(12, 5))

# Original 2D Signal
plt.subplot(1, 2, 1)
plt.imshow(signal_2d, cmap='gray')
plt.title("Original 2D Signal")

# Fourier Modes (Magnitude)
plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(signal_2d_ft)), cmap='gray')  # Log scale for better visibility
plt.title("Fourier Modes (Magnitude)")
plt.show()
