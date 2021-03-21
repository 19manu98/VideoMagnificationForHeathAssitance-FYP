from scipy import signal
from scipy.signal import butter,filtfilt,lfilter
from pyramids import check_content
import numpy as np

def processing(buffer_green_mean,times, buffer_size,real_fps):
    if check_content(buffer_green_mean) < 0.5:
        bpm = 0
    else:
        # calculate real fps regarding processor
        if real_fps is None:
            real_fps = float(buffer_size) / (times[-1] - times[0])
        # signal detrending
        signal_detrend = signal.detrend(buffer_green_mean)

        # butterworth filter
        nyq = 0.5 * real_fps
        order = 3

        lowsignal = 0.6667 / nyq  # 0.6667 correspond to 40bpm
        highsignal = 3 / nyq  # 3 correspond to 180bpm

        b, a = butter(order, [lowsignal, highsignal], btype='band', analog=True)
        # signal_detrend = filtfilt(b,a,signal_detrend)
        signal_detrend = lfilter(b, a, signal_detrend)

        # signal interpolation
        even_times = np.linspace(times[0], times[-1], buffer_size)
        interp = np.interp(even_times, times, signal_detrend)

        signal_interpolated = np.hamming(buffer_size) * interp
        signal_interpolated = signal_interpolated - np.mean(signal_interpolated)
        # normalize the signal
        # signal_normalization = signal_interpolated/np.linalg.norm(signal_interpolated)
        signal_normalization = signal_interpolated / np.std(signal_interpolated)

        # fast fourier transform
        raw_signal = np.fft.fft(signal_normalization)
        fft = np.abs(raw_signal)
        # fft = signal.detrend(fft)

        # freqs = float(real_fps)/current_size*np.arange(current_size/2+1)
        freqs = np.fft.rfftfreq(buffer_size, 1. / real_fps)

        freqs = 60. * freqs

        idx = np.where((freqs >= 40) & (freqs < 130))

        pruned = fft[idx]
        pfreq = freqs[idx]
        freqs = pfreq
        idx2 = np.argmax(pruned)

        bpm = freqs[idx2]
    return bpm