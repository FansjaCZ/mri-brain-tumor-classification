import numpy as np
import SimpleITK as sitk

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

def n4(rawIm, corrector=sitk.N4BiasFieldCorrectionImageFilter):
    sitk_im = sitk.GetImageFromArray(rawIm.astype("float32", copy=False))
    sitk_mask = sitk.OtsuThreshold(sitk_im, 0, 1)

    sitk_result = corrector.Execute(sitk_im, sitk_mask)

    return sitk.GetArrayFromImage(sitk_result)
