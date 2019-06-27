import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        y_ = np.argmax(y, axis = 1)
    else : 
        y_ = y
    
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
        y_ = np.argmax(y, axis = 1)
    else : 
        y_ = y

    return x, y_

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------