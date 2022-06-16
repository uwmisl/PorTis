import cv2
import math as m
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from scipy.signal import savgol_filter
import seaborn as sns
from tqdm import tqdm

"""
SignalProcessor is used to process capture events. Different methods of pre-processing can be applied,
and parameters for each of these can be specified. The `transform` function applies the pre-processing
steps specified, based on the parameters of the SignalProcessor.

Example Usage
-------------
signal_processor = SignalProcessor(downsample=True, reshape=True, rescale_dim=64)
data, labels, samples, indices = signal_processor.transform(data, labels, samples, indices)
"""
class SignalProcessor:
    
    def __init__(
        self, extract_features=False,
        delete_peak=2000, norm_method=None, smooth_method=None, smooth_window=301, smooth_polord=2,
        downsample=False, downsample_points=None, downsample_rate=None,
        pad=False, pad_value=2, pad_len=100000, preserve_longer_than_pad_length=False,
        add_channels_dim=False,
        reshape=False, rescale_dim=60, stack_depth=1, shuffle=False,
        duplicate_num=-1
    ): 
        self.extract_features = extract_features    # extract 6 features for each (min, max, mean, mode, median, and standard deviation)

        self.delete_peak = delete_peak              # delete first 2000 values to get rid of the initial peak
        self.norm_method = norm_method              # normalization method is to center to 0 and normalize between -1 and 1, others: scaled, centered

        self.smooth_method = smooth_method          # shape smoothing method default is None, others: savgol
        self.smooth_window = smooth_window          # smoothing window if smoothing is done
        self.smooth_polord = smooth_polord          # polynomial order to smooth if smoothing is done

        self.downsample = downsample                # indicates whether to downsample; will either downsample to a specific length, if downsample_points is specified or downsample based on a rate if downsample_rate is specified
        self.downsample_points = downsample_points  # reduce the signal to downsample_points, the rate is len(signal) // downsample_points
        self.downsample_rate = downsample_rate      # reduce the signal by rate 1 / downsample_rate

        self.pad = pad                              # indicates whether to pad the signal
        self.pad_value = pad_value                  # pad the signal with an integer outside of the -1 and 1 integer, in this case default is 2
        self.pad_len = pad_len                      # set this length as the total padded length of the signal that would have before downsampling
        self.preserve_longer_than_pad_length = preserve_longer_than_pad_length  # set this to true if you don't want to truncate events longer than pad_length
        
        self.add_channels_dim = add_channels_dim    # add channels dimension, useful for data fed to CNNs (in Pytorch, dim of channels is 1)

        self.reshape = reshape                      # indicates whether to reshape to square
        self.rescale_dim = rescale_dim              # size to rescale to after reshaping into square; only use if reshape method is not None
        self.stack_depth = stack_depth              # number of signals to stack into one data point
        self.shuffle = shuffle                      # shuffle data in the stacks; only used if stack_depth > 1
        self.duplicate_num = duplicate_num        # each stack contains duplicate_data number of the same event
        
    # DEFINE functions for feature extraction
    
    def extract_stats_features(self, sig):
        return [
            np.min(sig),
            np.max(sig),
            np.mean(sig),
            stats.mode(sig, axis=None).mode[0],
            np.median(sig),
            np.std(sig),
            len(sig)
        ]
 
    # DEFINE functions for normalization and smoothing steps (no need for capping, downsampling or padding steps)
    
    # normalization functions
    def norm_cent(self, sig):
        # scale to -1 and 1, center mean at 0
        signal_minusmean = sig - np.mean(sig)
        return signal_minusmean/np.max(np.abs(signal_minusmean))
    
    def norm_scal(self, sig):
        # scale to 0 and 1
        sig_max = np.max(sig)
        sig_min = np.min(sig)
        return (sig - sig_min) / (sig_max - sig_min)
    
    # smoothing function (only 1 for the moment)
    def smooth_savgol(self, sig):
        return savgol_filter(sig, self.smooth_window, self.smooth_polord)
    
    # downsampling functions
    def downsample_to_length(self, sig):
        downsample_rate = len(sig) // self.downsample_points
        if downsample_rate > 0:
            return sig[::downsample_rate][:self.downsample_points]
        else:
            # If the length of the signal is less than self.downsample_points, then
            # we should just leave the signal as is. It is up to the user to add padding.
            # Padding gets added after downsampling.
            return sig
    
    def downsample_by_rate(self, sig):
        return sig[::self.downsample_rate]
    
    # padding function
    def pad_length(self, sig):
        # calculate length of downsampled signals from pad_len
#         sign_length = int(self.pad_len / self.downsample_points) # ex. 100000/1000 = 100
        sign_length = self.pad_len
        if len(sig) < sign_length:
            return np.pad(sig, (0, sign_length-len(sig)), mode='constant', constant_values=self.pad_value) #pad with pad_value
        elif not self.preserve_longer_than_pad_length:
            return sig[:sign_length]
        else:
            return sig
    
    # DEFINE functions for CNNs
    
    # truncate signal to the nearest square length and reshape into a 2D square 
    def reshape_square(self, sig):
        square_size = int(m.floor(m.sqrt(len(sig))))
        square_len = int(square_size * square_size)
        sig = sig[:square_len]
        sig = np.reshape(sig, (square_size, square_size))
        return cv2.resize(sig, dsize=(self.rescale_dim, self.rescale_dim), interpolation=cv2.INTER_CUBIC)

    # stack multiple events into a single data point
    def stack_data(self, data, labels, samples, indices=None):
        final_data = []
        final_labels = []
        final_samples = []
        final_indices = []

        num_classes = len(np.unique(labels))
        for c_id in range(num_classes):
            idxs = np.where(labels == c_id)[0]
            class_data, class_labels, class_samples = \
                data[idxs], labels[idxs], samples[idxs]
            if indices is not None:
                class_indices = indices[idxs]
            for s_id in np.unique(class_samples):
                # Subset data for a single class and a single sample, since it doesn't make sense
                # to stack signals from different samples.
                idxs = np.where(class_samples == s_id)[0]
                sample_data, sample_labels, sample_samples = \
                    class_data[idxs], class_labels[idxs], class_samples[idxs]
                if indices is not None:
                    sample_indices = class_indices[idxs]

                # Remove some signals, so that we can evenly stack
                num_signals = int(self.stack_depth * m.floor(len(idxs) / self.stack_depth))
                num_stacks = int(num_signals / self.stack_depth)
                sample_data = sample_data[:num_signals]
                sample_labels = sample_labels[:num_signals]
                sample_samples = sample_samples[:num_signals]
                if indices is not None:
                    sample_indices = sample_indices[:num_signals]

                # Shuffle data before stacking, if specified
                if self.shuffle:
                    randomize = np.arange(len(sample_data))
                    np.random.shuffle(randomize)
                    sample_data = np.take(sample_data, randomize, axis=0)
                    if indices is not None:
                        sample_indices = np.take(sample_indices, randomize)

                # Stack signals into specified examples per stack
                sample_data = sample_data.reshape(num_stacks, self.stack_depth, *(sample_data.shape[1:]))
                # Note that we can do the following, since the label and sample ids are the same
                sample_labels = sample_labels[:num_stacks].reshape(num_stacks, -1)
                sample_samples = sample_samples[:num_stacks].reshape(num_stacks, -1)
                if indices is not None:
                    sample_indices = sample_indices.reshape(num_stacks, self.stack_depth, -1)

                final_data.append(sample_data)
                final_labels.append(sample_labels)
                final_samples.append(sample_samples)
                if indices is not None:
                    final_indices.append(sample_indices)

        final_data = np.vstack(final_data)
        final_labels = np.vstack(final_labels).reshape((-1,))
        final_samples = np.vstack(final_samples)
        final_indices = np.vstack(final_indices) if indices is not None else np.array([])
        
        print(final_data.shape, final_labels.shape, final_samples.shape)

        return final_data, final_labels, final_samples, final_indices
    
    # duplicate each event in a single data point
    def duplicate_data(self, data, labels, samples, indices):
        data = np.repeat(data[..., np.newaxis], self.duplicate_num, -1)
        data = np.moveaxis(data, -1, 1)
        return data, labels, samples, indices
        
    # Applies various transformations and pre-processing steps to the given data
    def transform(self, data, labels, samples, indices,  *_):
        
        # empty object to store processed signals        
        proc_data = []
        
        # select function for normalization
        if self.norm_method == 'centered':
            norm_func = self.norm_cent
        elif self.norm_method == 'scaled':
            norm_func = self.norm_scal
        else:
            norm_func = lambda sig: sig # return signal itself
        
        # select function for smoothing
        if self.smooth_method == 'savgol':
            smooth_func = self.smooth_savgol
        else:
            smooth_func = lambda sig: sig # return signal itself
            
        # select function for downsampling
        if self.downsample:
            if self.downsample_points is not None:
                downsample_func = self.downsample_to_length
            elif self.downsample_rate is not None:
                downsample_func = self.downsample_by_rate
        else:
            downsample_func = lambda sig: sig # return signal itself

        # select function for padding
        if self.pad:
            pad_func = self.pad_length
        else:
            pad_func = lambda sig: sig # return signal itself
        
        # select function for reshaping and rescaling
        if self.reshape:
            reshape_func = self.reshape_square
        else:
            reshape_func = lambda sig: sig # return signal itself
            
        # select function for extracting features
        if self.extract_features:
            extract_features_func = self.extract_stats_features
        else:
            extract_features_func = lambda sig: sig # return signal itself
        
        
        # actually process signals
        for signal in tqdm(data):
            # CAP SIGNAL
            processed_signal = signal
            if len(signal) > self.delete_peak:
                processed_signal = signal[self.delete_peak:]
            
            # NORMALIZE, if specified
            processed_signal = norm_func(processed_signal)
                
            # SMOOTH THE SIGNAL, if specified
            processed_signal = smooth_func(processed_signal)    

            # DOWNSCALE THE SIGNAL, if specified
            processed_signal = downsample_func(processed_signal)
            
            # PAD THE SIGNAL, if specified; ensure all have the same length cutting if necessary
            processed_signal = pad_func(processed_signal)
            
            # RESHAPE THE SIGNAL, if specified
            processed_signal = reshape_func(processed_signal)
            
            # EXTRACT FEATURES, if specified
            processed_signal = extract_features_func(processed_signal)

            proc_data.append(processed_signal)
            
        # result is a 2D array (number_signals, signal_length)
        data = np.array(proc_data, dtype='float64')
        
        # stack data, if specified
        if self.duplicate_num != -1:
            data, labels, samples, indices = self.duplicate_data(data, labels, samples, indices)
        elif self.stack_depth > 1:
            # TODO: Ensure that channels is the correct dim (1)
            data, labels, samples, indices = self.stack_data(data, labels, samples, indices)
        # or add channels dim if required
        elif self.add_channels_dim:
            data = np.expand_dims(data, axis=1)
            
        return data, labels, samples, indices