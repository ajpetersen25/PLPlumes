from scipy.signal import detrend
from scipy.signal import convolve
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import cv2
from scipy import stats

def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def esd1d(signal, freq, windows, overlap=0.5):
    """ calculate the energy spectrum density of a signal--
    The ESD represents the energy per unit frequency of a signal.
    The signal is zero-padded to the next power or 2. The input series
    is assumed to be uniformly spaced.

    Inputs: signal      (N,) array:    1-d signal (evenly spaced)
            freq        float:          frequency of signal
            windows     int:            number of hanning windows to apply
            overlap     float           window overlap (default = 0.5)
    Outputs:
            ESD         2xN array:      first row is the frequency series,
                                        second row is the energy spectrum estimate
    """
    window_size = np.floor(
        len(signal) / (windows * (1 - overlap) + overlap)).astype('int')
    signal = detrend(signal)  # detrend the signal
    # create hanning window
    W = np.hanning(window_size)

    # overlapping points
    on = np.floor(overlap * window_size).astype('int')

    # FFT length, padded
    nfft = 2 * nextpow2(2 * window_size - 1)

    # indices or first window
    ii = np.arange(0, window_size)

    # Autocovariance transform output
    R_hat = np.zeros((nfft, windows))
    for i in range(0, windows):
        s = signal[ii]                # fluctuations in the current window
        s = s * W                         # window filter in the domain
        s_hat = np.fft.fft(s, nfft)    # zero-padded fft
        R_hat[:, i] = np.abs(s_hat)**2  # autocovariance transform
        ii = ii + on
    # average R_hat from each window
    R_hat = np.nanmean(R_hat, axis=1)
    # E = 2xR_hat by overlapping the symmetric halves of the result
    # n = 0 and n = N/2 are not scaled by 2
    E = 2 * R_hat[0:int(nfft / 2 + 1)]
    # create frequency series based on sampling frequency
    f = np.linspace(0, 1, int(nfft / 2 + 1))
    f = freq / 2 * f

    # scale to force the integral of E(f) to equal the variance
    E2 = E * np.var(signal) / np.trapz(E, f)

    # Identify and remove values below the minimum frequency
    fmin = freq / window_size
    E2 = E2[f >= fmin]
    f = f[f >= fmin]

    return np.concatenate((f.reshape(len(f), 1), E2.reshape(len(E2), 1)), axis=1)


def windowed_average(a, kernel_size, mode='same'):
    k = np.ones((kernel_size),dtype=int)
    window_sum = convolve(a,k,mode)
    window_count = convolve(np.ones(a.shape, dtype=bool),k,mode)
    return np.array(window_sum/window_count)

def gaussian(x, a, x0, sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+offset

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2

def concentration_cal_calc(I,C,kernel_size,img_length):
    alphas = []
    r_sqr = []
    y_int = []
    for i in range(0,img_length):
        p=np.polyfit(I[:,i],C,1)
        alphas.append(p[0])
        r_sqr.append(rsquared(C,p[0]*I[:,i]))
        y_int.append(p[1])
    alphas = np.asarray(alphas)
    r_sqr = np.asarray(r_sqr)
    y_int = np.asarray(y_int)
    return alphas,r_sqr,y_int

def convert_I_to_C(image,alphas,smoothing_kernel):
    return image*gaussian_filter(alphas,sigma=smoothing_kernel)


def plume_outline(frame,kernel,threshold):
    img_outline = frame>threshold
    image, contours, hierarchy =   cv2.findContours(img_outline.copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_size = ([c.size for c in contours])
    lc = np.where(c_size == np.max(c_size))[0][0]
    plume_contour =cv2.drawContours(image,contours[lc],-1, 255, 1)

    return plume_contour

def gaussian_plume_width():
    """ based on the near half of the plume image, fit a gaussian"""
