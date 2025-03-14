from typing import Union

import torch
from torch import nn
import numpy as np
import time

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
    
def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs

    if isinstance(params["output_activation"], str):
        output_activation = _str_to_activation[params["output_activation"]]
    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    raise NotImplementedError

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def print_gpu_info():
    """
    Print information about GPU availability and memory usage
    """
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {device}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Print memory usage
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Print memory summary for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Memory Summary:")
            print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("CUDA is not available. Using CPU.")

def cuda_sync():
    """
    Synchronize CUDA for accurate timing measurements
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_function(func, *args, **kwargs):
    """
    Time a function execution with proper CUDA synchronization
    
    Args:
        func: Function to time
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        result: Result of the function
        elapsed_time: Time taken in seconds
    """
    # Synchronize before starting
    cuda_sync()
    start_time = time.time()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Synchronize after completion
    cuda_sync()
    end_time = time.time()
    
    return result, end_time - start_time
