# File contains some preprocessing functions that are useful.

import numpy as np

# Function to add transition delays to the locomotion mode labels.
def add_transition_delays(labels, delay_table):
    processed_labels = np.zeros(labels.shape)
    TD = delay_table
    for i in range(labels.shape[0]):
        processed_labels[i] = labels[i]
    for i in range(labels.shape[0]-1):
        if labels[i] != labels[i+1]:
            # Check if there needs to be a delay for this one
            y_c = "{}".format(int(labels[i]))
            y_n = "{}".format(int(labels[i+1]))
            if y_n in TD[y_c].keys():
                delay = TD[y_c][y_n]
                if delay < 0:
                    processed_labels[i+delay:i+1] = labels[i+1]
                elif delay > 0:
                    processed_labels[i:i+delay+1] = labels[i]
    return processed_labels

# Function to preprocess the data into sequences for the RNN
def get_sub_sequences(data_array, y_array, window_size=120, step_size=90, dims=None, seq_out=False, causal=True):
    rows = data_array.shape[0]
    cols = data_array.shape[1]

    if dims == None:
        outdims = [i for i in range(cols)]
    else:
        outdims = dims
    
    idxs = range(window_size, rows, step_size)
    sequences = len(idxs)
    out_x = np.zeros((sequences, window_size, len(outdims)))
    if seq_out:
        out_y = np.zeros((sequences, window_size, y_array.shape[1]))
    else:
        out_y = np.zeros((sequences, y_array.shape[1]))
    
    for i, j in enumerate(idxs):
        out_x[i, :, :] = data_array[j-window_size:j, outdims]
        if seq_out:
            out_y[i, :, :] = y_array[j-window_size:j, :]
        else:
            out_y[i, :] = y_array[j, :]
    
    return out_x, out_y


def data_summary(name, data):
    shape = data.shape
    return '{}: {}'.format(name, shape)

#def plot_data_with_labels(data, labels, n_labels, label_string_dict=None):
#    # If the labels are numeric
#    if labels.shape == 1:
#        if label_string_dict is not None:
#            
#    for i in range()