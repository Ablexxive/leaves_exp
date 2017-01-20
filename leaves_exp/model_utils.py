import numpy as np

def std_He(fan_in):
    gain = np.sqrt(2.0) #gain = squrt(2) for relu
    sigma = gain * np.sqrt(1.0/fan_in)
    return sigma
