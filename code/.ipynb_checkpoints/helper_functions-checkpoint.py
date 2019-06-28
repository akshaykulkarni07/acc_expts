import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack
import scipy.stats
import scipy.signal

#-------------------------------------------------------------
# Data Loading Helper Functions
#-------------------------------------------------------------

def return_segments_3axis(path_to_df, one_hot_encoded = False):
    ''' Returns 2 arrays : 
        x of shape (num_examples = len(df), num_features = 450)
        if one_hot_encoded = False : 
            y of shape (num_examples = len(df), num_classes = 5)
        else :
            y of shape (num_examples = len(df), class_label = 1)
        Extracts this from Pandas DataFrame whose path is given
        This function returns all axes IMU data 
        (for single axis IMU data, there is another function `return_segments_1axis`)
    '''
    # Read csv file from given path
    df = pd.read_csv(path_to_df, header = None)
    
    # initialize NumPy arrays to be used for the output
    x = np.zeros((len(df) // 150, 450))
    y = np.zeros((len(df) // 150, 5))
    
    for i in range(1, len(df) // 150) : 
        # taking 150 values of 3 channels at a time, flattening and 
        temp = df.iloc[(i - 1) * 150 : i * 150,  : 3].values
        x[i - 1] = np.transpose(temp).reshape(450)
        # finding single one_hot encoded label (which occurs maximum times in 150 values)
        label_array = df.iloc[((i - 1) * 150) : i * 150, 3 : ].values
        ind = np.argmax(np.sum(label_array, axis = 0))
        label = np.zeros_like(df.iloc[0, 3 : ].values)
        label = label.astype('float')
        label[ind] = 1
        y[i - 1] = label
        
    num = len(df) // 150
    # last example isn't considered in the loop
    temp = df.iloc[(num - 1) * 150 : , : 3].values
    x[num - 1] = np.transpose(temp).reshape(450)
    # just taking the central value for the last label
    y[num - 1] = df.iloc[((num - 1) * 150) + 75, 3 : ].values
    
    # convert labels to one hot encoded form if needed
    if one_hot_encoded : 
        y_ = y
    else : 
        y_ = np.argmax(y, axis = 1)
    
    return x, y_

def return_segments_1axis(path_to_df, axis, one_hot_encoded = False):
    ''' Returns 2 arrays : 
        x of shape (num_examples, num_features = 150)
        if one_hot_encoded = False :
            y of shape (num_examples, num_classes = 5)
        else : 
            y of shape (num_examples, class_label = 1)
        Extracts only one axis data and labels from Pandas DataFrame whose path is given
        (There is another function `return_segments_3axis` for getting all axes data)
    '''
    # Read csv file from given path
    df = pd.read_csv(path_to_df, header = None)
    
    # initialize NumPy arrays to be used for the output
    x = np.zeros((len(df) // 150, 150))
    y = np.zeros((len(df) // 150, 5))
    
    # in case axis is not correctly given (may cause error later while accessing in dataframe)
    if axis < 0 or axis > 2 : 
        print('Please give axis parameter between 0 and 2 (0 for x, 1 for y, 2 for z). Exiting with zero output')
        return x, y
    
    for i in range(1, len(df) // 150) : 
        # taking 150 values of 3 channels at a time, flattening and 
        x[i - 1] = df.iloc[(i - 1) * 150 : i * 150, axis].values
        # finding single one_hot encoded label (which occurs maximum times in 150 values)
        label_array = df.iloc[((i - 1) * 150) : i * 150, 3 : ].values
        ind = np.argmax(np.sum(label_array, axis = 0))
        label = np.zeros_like(df.iloc[0, 3 : ].values)
        label = label.astype('float')
        label[ind] = 1
        y[i - 1] = label
        
    num = len(df) // 150
    # last example isn't considered in the loop
    x[num - 1] = df.iloc[(num - 1) * 150 : , 0].values
    # just taking the central value for the last label
    y[num - 1] = df.iloc[((num - 1) * 150) + 75, 3 : ].values
    
    if one_hot_encoded : 
        y_ = y
    else : 
        y_ = np.argmax(y, axis = 1)

    return x, y_

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------

#-------------------------------------------------------------
# Preprocessing Functions
#-------------------------------------------------------------

# Source : https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
# Below 2 functions are needed for low pass filtering single row (or column) data
def butter_lowpass(cutoff, nyq_freq, order = 4) :
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype = 'lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order = 4) :
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order = order)
    y = signal.filtfilt(b, a, data)
    return y

def low_pass_filter(x, axes, cutoff_freq = 5.0, nyq_freq = 50 // 2, order = 4) : 
    ''' Returns 4th order Butterworth low pass filtered output of each example
    Assumes x is the first output of return_segments_1axis or return_segments_3axis function
    Use parameter axes = 3 for 3 axes data and axes = 1 for 1 axis data
    '''
    # initializing output NumPy array
    y = np.zeros_like(x)
    # Checking if axes is 1 and whether that is reflected in the data
    if axes == 1 and x.shape[1] == 150 :
        i = 0
        for x_acc in x : 
            y[i] = butter_lowpass_filter(x_acc, cutoff_freq = cutoff_freq, nyq_freq = nyq_freq, order = order)
            i = i + 1
        return y
    # Checking if axes is 3 and whether that is reflected in the data
    elif axes == 3 and x.shape[1] == 450 :
        i = 0
        for x_acc in x : 
            y[i, : 150] = butter_lowpass_filter(x_acc[ : 150], cutoff_freq = cutoff_freq, nyq_freq = nyq_freq, order = order)
            y[i, 150 : 300] = butter_lowpass_filter(x_acc[150 : 300], cutoff_freq = cutoff_freq, nyq_freq = nyq_freq, order = order)
            y[i, 300 : ] = butter_lowpass_filter(x_acc[300 : ], cutoff_freq = cutoff_freq, nyq_freq = nyq_freq, order = order)
            i = i + 1
        return y
    # otherwise, there is some issue, so exit the function
    else : 
        print('Use axes = 1 or 3 only and correctly') 
        return y
    
    return

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
    
#-------------------------------------------------------------
# Feature Extraction Functions
#-------------------------------------------------------------
def mean(x, axes) : 
    ''' Returns the mean of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the ouput NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = np.mean(x_)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the ouput NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = np.mean(x_[ : 150])
            y[i, 1] = np.mean(x_[150 : 300])
            y[i, 2] = np.mean(x_[300 : ])
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def std_dev(x, axes) :
    ''' Returns the standard deviation of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = np.std(x_)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = np.std(x_[ : 150])
            y[i, 1] = np.std(x_[150 : 300])
            y[i, 2] = np.std(x_[300 : ])
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def variance(x, axes) :
    ''' Returns the variance of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = np.var(x_)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = np.var(x_[ : 150])
            y[i, 1] = np.var(x_[150 : 300])
            y[i, 2] = np.var(x_[300 : ])
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def energy(x, axes) :
    ''' Returns the absolute energy of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = np.sum(x_ ** 2)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = np.sum(x_[ : 150] ** 2)
            y[i, 1] = np.sum(x_[150 : 300] ** 2)
            y[i, 2] = np.sum(x_[300 : ] ** 2)
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def rms(x, axes) : 
    ''' Returns the root mean square value of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = np.sqrt(np.mean(x_ ** 2))
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = np.sqrt(np.mean(x_[ : 150] ** 2))
            y[i, 1] = np.sqrt(np.mean(x_[150 : 300] ** 2))
            y[i, 2] = np.sqrt(np.mean(x_[300 : ] ** 2))
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def skewness(x, axes) : 
    ''' Returns the skewness of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = scipy.stats.skew(x_)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = scipy.stats.skew(x_[ : 150])
            y[i, 1] = scipy.stats.skew(x_[150 : 300])
            y[i, 2] = scipy.stats.skew(x_[300 : ])
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def kurtosis(x, axes) : 
    ''' Returns the kurtosis of each row in x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            y[i] = scipy.stats.kurtosis(x_)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            y[i, 0] = scipy.stats.kurtosis(x_[ : 150])
            y[i, 1] = scipy.stats.kurtosis(x_[150 : 300])
            y[i, 2] = scipy.stats.kurtosis(x_[300 : ])
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def num_peaks(x, axes) : 
    ''' Returns the number of peaks in each row in low pass filtered output of x
    Give axes = 1 if single axis data is input
    Give axes = 3 if 3 axis data is input
    Otherwise output will be None type
    '''
    # checking if single axis data is provided as input
    if axes == 1 and x.shape[1] == 150 : 
        # low pass filtering the input
        x = low_pass_filter(x, axes = axes)
        # initializing the output NumPy array
        y = np.zeros(x.shape[0])
        i = 0
        for x_ in x : 
            peaks, _ = scipy.signal.find_peaks(x_)
            y[i] = len(peaks)
            i = i + 1
        return y
    # checking if 3 axis data is provided as input
    elif axes == 3 and x.shape[1] == 450 : 
        # low pass filtering the input
        x = low_pass_filter(x, axes = axes)
        # initializing the output NumPy array
        y = np.zeros((x.shape[0], axes))
        i = 0
        for x_ in x : 
            peaks, _ = scipy.signal.find_peaks(x_[ : 150])
            y[i, 0] = len(peaks)
            peaks, _ = scipy.signal.find_peaks(x_[150 : 300])
            y[i, 1] = len(peaks)
            peaks, _ = scipy.signal.find_peaks(x_[300 : ])
            y[i, 2] = len(peaks)
            i = i + 1
        return y
    # in any other case, there is an issue with the data
    else : 
        print('Check data and try again')
        return None
    
def return_features(x, axes) : 
    ''' Returns all features (whose functions are defined here)
    axes = 1 or 3 (depending on which data is being passed)
    '''
    x_mean = mean(x, axes)
    x_std = std_dev(x, axes)
    x_var = variance(x, axes)
    x_energy = energy(x, axes)
    x_rms = rms(x, axes)
    x_skew = skewness(x, axes)
    x_kurtosis = kurtosis(x, axes)
    x_num_peaks = num_peaks(x, axes)
    # return concatenated feature array
    return np.concatenate((x_mean, x_std, x_var, x_energy, x_rms, x_skew, x_kurtosis, x_num_peaks), axis = 1)

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
    
# Testing code
# x_train, y_train = return_segments_1axis('../data/test.csv', axis = 2)
# y = low_pass_filter(x_train, axes = 1)

# fig, ax = plt.subplots()
# ax.plot(x_train[90], 'r')
# ax.plot(y[90], 'b', linewidth = 3)
# plt.show()