
import torch
import numpy as np
from gwpy.timeseries import TimeSeries


def init_weights(net, init_type='normal', scaling=0.02):
    """
    Initialize the weights of the neural network according to the specified
    initialization type.

    Parameters:
        - net (nn.Module): The neural network model.
        - init_type (str): Type of initialization. Options: 'normal', 'xavier'
            (default is 'normal').
        - scaling (float): Scaling factor for weight initialization
            (default is 0.02).
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        # BatchNorm Layer's weight is not a matrix; only normal
        # distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function


def calculate_iou_2d(generated, target, threshold):
    """
    Calculate Intersection over Union (IoU) in the 2D plane at the specified
    intensity threshold.

    Parameters:
    - generated: List of time series representing the generated spectrograms
    - target: List of time series representing the target spectrograms
    - threshold: Intensity threshold for determining the binary masks

    Returns:
    - IoU: Intersection over Union
    """
    # Extract spectrogram values from time series
    # print(generated[0][0])
    # print(generated[0][0].shape)
    # print(type(generated[0][0]))

    spectrograms_gen = [TimeSeries(
        t[0], dt=1/4096.0).q_transform(frange=(10, 1000)).value
        for t in generated]
    spectrograms_real = [TimeSeries(
        t[0], dt=1/4096.0).q_transform(frange=(10, 1000)).value
        for t in target]

    # Create binary masks based on the intensity threshold
    mask1 = [spectrogram >= threshold for spectrogram in spectrograms_gen]
    mask2 = [spectrogram >= threshold for spectrogram in spectrograms_real]

    # Calculate the intersection and union of the binary masks
    intersection = [np.logical_and(m1, m2) for m1, m2 in zip(mask1, mask2)]
    union = [np.logical_or(m1, m2) for m1, m2 in zip(mask1, mask2)]

    # Calculate Intersection over Union (IoU)
    iou_list = np.array([np.sum(inter) / np.sum(uni)
                        for inter, uni in zip(intersection, union)])

    iou = iou_list.mean()
    return iou
