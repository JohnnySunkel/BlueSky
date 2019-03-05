# Visualize an audio signal

# import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# read the audio file
sampling_freq, signal = wavfile.read('random_sound.wav')

# print the shape of the signal, datatype, and duration
# of the audio signal
print('\nSignal shape:', signal.shape)
print('Datatype:', signal.dtype)
print('Signal duration:', 
      round(signal.shape[0] / float(sampling_freq), 2), 'seconds')

# normalize the signal
signal = signal / np.power(2, 15)

# extract the first 50 values for plotting
signal = signal[:50]

# construct the time axis in milliseconds
time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)

# plot the audio signal
plt.plot(time_axis, signal, color = 'black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()
