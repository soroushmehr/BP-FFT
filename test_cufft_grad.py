import ipdb
import numpy
np = numpy
import theano
import theano.tensor as T

from theano.sandbox.cuda.cuda_fft import cufft
np.random.seed(seed=123)


frame_size = 200

# calculate real fft 2-D matrix (batch, 1-D signals) from complex fft.
# frame_size should be even. Or use //2+1
def realfft(y):
    y_fft_in = T.zeros(shape=(y.shape[0], y.shape[1], 2), dtype=theano.config.floatX)
    y_fft_in = T.set_subtensor(y_fft_in[:, :, 0], y)
    return cufft(y_fft_in)[:, :frame_size/2+1, :]


# abs/magnitude of a fft response
# y: realfft() complex output
def magnitude(y):
    return T.sqrt(y[:, :, 0]**2 + y[:, :, 1]**2)


# following part needs this library:
# https://github.com/jameslyons/python_speech_features
# ignore it if not necessary.
from features.base import get_filterbanks
fb_coeff = get_filterbanks(nfilt=64, nfft=frame_size, samplerate=16000)
fb_coeff = fb_coeff.astype(theano.config.floatX)
fb_coeff = np.transpose(fb_coeff)
fb_coeff = theano.shared(fb_coeff)

def spectral_magnitude_log_distance_error(y_true, y_pred):
    Y_true = realfft(y_true)
    Y_pred = realfft(y_pred)
    mag_Y_true = magnitude(Y_true)
    mag_Y_pred = magnitude(Y_pred)
    mel_true = T.dot(mag_Y_true, fb_coeff)+1.0e-8
    mel_pred = T.dot(mag_Y_pred, fb_coeff)+1.0e-8
    return T.mean(T.sqr(T.log(mel_pred)-T.log(mel_true)).sum(axis=1))

