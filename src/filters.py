import numpy as np

def ensure64(im):
    return np.minimum(im.real.astype(np.float64), 1)

def hf_2d(rawIm, lowGain=0.6, highGain=2, cutoffMultiplier=1.2):
    im = np.maximum(rawIm, 1e-6)
    logIm = np.log(im)

    fft = np.fft.fftshift(np.fft.fft2(logIm))

    height, width = im.shape
    freqX, freqY = np.meshgrid(np.arange(width)- width // 2, np.arange(height) - height // 2)
    radius = np.sqrt(freqY**2 + freqX**2)

    cutoffFrequency = cutoffMultiplier * min(height, width)
    highpass = ((lowGain - highGain)*(1-np.exp(-0.5*(radius/cutoffFrequency)**2)) + lowGain)
    pfft = fft*highpass

    pifft = ensure64(np.exp(np.fft.ifft2(np.fft.ifftshift(pfft)).real))

    return pifft