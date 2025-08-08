import os
import pandas as pd
import numpy as np
import scipy.signal
import scipy.io.wavfile
from pynwb import NWBHDF5IO
import pywt
import MelFilterBank as mel
import pyworld
import scipy

# Helper function for Hilbert transform
hilbert_fast = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]

def extract_high_gamma(data, sr, window_length=0.05, frame_shift=0.01):
    """
    Extract high-gamma band features from iEEG signals.
    """
    data = scipy.signal.detrend(data, axis=0)
    num_windows = int(np.floor((data.shape[0] - window_length * sr) / (frame_shift * sr)))
    
    # Bandpass filter for high-gamma band
    sos = scipy.signal.iirfilter(4, [70 / (sr / 2), 170 / (sr / 2)], btype="bandpass", output="sos")
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    
    # Remove line noise harmonics
    for freq in [98, 102, 148, 152]:
        sos = scipy.signal.iirfilter(4, [(freq - 2) / (sr / 2), (freq + 2) / (sr / 2)], btype="bandstop", output="sos")
        data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    
    # Extract envelope using Hilbert transform
    data = np.abs(hilbert_fast(data))
    features = np.zeros((num_windows, data.shape[1]))
    
    for win in range(num_windows):
        start = int(np.floor((win * frame_shift) * sr))
        stop = int(np.floor(start + window_length * sr))
        features[win, :] = np.mean(data[start:stop, :], axis=0)
    
    return features

def extract_wavelet_coefficients(data, sr, window_length=0.05, frame_shift=0.01, wavelet="db4", level=5):
    """
    Extract wavelet coefficients from iEEG signals.
    """
    data = scipy.signal.detrend(data, axis=0)
    num_windows = int(np.floor((data.shape[0] - window_length * sr) / (frame_shift * sr)))
    features = []

    for chan in range(data.shape[1]):
        chan_features = []
        for win in range(num_windows):
            start = int(np.floor((win * frame_shift) * sr))
            stop = int(np.floor(start + window_length * sr))
            segment = data[start:stop, chan]
            coeffs = pywt.wavedec(segment, wavelet=wavelet, level=level)
            detail_coeffs = coeffs[1:]  # Exclude approximation coefficients
            energies = [np.sum(np.abs(d) ** 2) for d in detail_coeffs]
            chan_features.append(energies)
        features.append(np.array(chan_features))
    
    return np.array(features).transpose(1, 0, 2)

def extract_mel_spectrogram(audio, sr, window_length=0.05, frame_shift=0.01, n_mels=23):
    """
    Extract Mel spectrogram from audio signals.
    """
    num_windows = int(np.floor((audio.shape[0] - window_length * sr) / (frame_shift * sr)))
    win = scipy.signal.windows.hann(int(np.floor(window_length * sr + 1)))[:-1]
    spectrogram = np.zeros((num_windows, int(np.floor(window_length * sr / 2 + 1))), dtype="complex")
    
    for w in range(num_windows):
        start = int(np.floor((w * frame_shift) * sr))
        stop = int(np.floor(start + window_length * sr))
        segment = audio[start:stop]
        spec = np.fft.rfft(win * segment)
        spectrogram[w, :] = spec
    
    mfb = mel.MelFilterBank(spectrogram.shape[1], n_mels, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype("float")
    
    return spectrogram

def extract_neural_coupling(data, sr, low_freq_band=(4, 8), high_freq_band=(70, 170)):
    """
    Extract Cross-Frequency Coupling (CFC) features.
    """
    # Bandpass filters for low and high frequencies
    sos_low = scipy.signal.iirfilter(4, [low_freq_band[0] / (sr / 2), low_freq_band[1] / (sr / 2)], btype='bandpass', output='sos')
    sos_high = scipy.signal.iirfilter(4, [high_freq_band[0] / (sr / 2), high_freq_band[1] / (sr / 2)], btype='bandpass', output='sos')
    
    low_freq_data = scipy.signal.sosfiltfilt(sos_low, data, axis=0)
    high_freq_data = scipy.signal.sosfiltfilt(sos_high, data, axis=0)
    
    # Extract phase and amplitude
    low_phase = np.angle(scipy.signal.hilbert(low_freq_data))
    high_amplitude = np.abs(scipy.signal.hilbert(high_freq_data))
    
    # Compute Phase-Amplitude Coupling (PAC)
    pac = np.mean(high_amplitude * np.exp(1j * low_phase), axis=0).real
    return pac

def extract_speech_prosody(audio, sr, window_length=0.05, frame_shift=0.01):
    """
    Extract prosody features (pitch, energy, shimmer, duration, and CFC).
    """
    num_windows = int(np.floor((len(audio) - window_length * sr) / (frame_shift * sr)))
    prosody_features = []

    for w in range(num_windows):
        start = int(np.floor((w * frame_shift) * sr))
        stop = int(np.floor(start + window_length * sr))
        segment = audio[start:stop]

        # Default values
        pitch, energy, shimmer, duration = 0.0, 0.0, 0.0, 0.0
        cfc_features = [0.0]

        if len(segment) >= sr * window_length and np.abs(segment).max() >= 1e-6:
            try:
                # Pitch estimation using DIO
                f0, t = pyworld.dio(segment.astype(np.float64), sr)
                f0 = pyworld.stonemask(segment.astype(np.float64), f0, t, sr)
                valid_f0 = f0[f0 > 0]
                pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 0.0
            except Exception as e:
                print(f"Error in pitch estimation: {e}")

            # Energy and shimmer
            energy = np.sqrt(np.mean(segment ** 2))
            amplitudes = np.abs(segment)
            shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes) if len(amplitudes) > 1 else 0.0

            # Duration
            duration = len(segment) / sr

            # Phase variability
            try:
                analytic_signal = scipy.signal.hilbert(segment)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                phase_variability = np.std(np.diff(instantaneous_phase))
            except Exception as e:
                print(f"Error in phase variability computation: {e}")
                phase_variability = 0.0

            # Cross-Frequency Coupling (CFC)
            try:
                cfc_features = extract_neural_coupling(segment, sr, low_freq_band=(4, 8), high_freq_band=(70, 170))
                if np.isscalar(cfc_features):
                    cfc_features = [cfc_features]
            except Exception as e:
                print(f"Error in CFC extraction: {e}")

        prosody_features.append([pitch, energy, shimmer, duration, phase_variability] + list(cfc_features))
    
    return np.array(prosody_features)

def downsample_labels(labels, sr, window_length=0.05, frame_shift=0.01):
    """
    Downsample labels using mode.
    """
    num_windows = int(np.floor((labels.shape[0] - window_length * sr) / (frame_shift * sr)))
    new_labels = np.empty(num_windows, dtype=object)
    
    for w in range(num_windows):
        start = int(np.floor(w * frame_shift * sr))
        stop = int(np.floor(start + window_length * sr))
        segment = labels[start:stop]
        unique, counts = np.unique(segment, return_counts=True)
        new_labels[w] = unique[np.argmax(counts)]
    
    return new_labels

def generate_feature_names(elecs, model_order=4):
    """
    Generate feature names with temporal context.
    """
    names = np.repeat(elecs.astype(np.dtype(("U", 10))), 2 * model_order + 1).reshape(2 * model_order + 1, -1).T
    for i, off in enumerate(range(-model_order, model_order + 1)):
        names[i, :] = [e[0] + "T" + str(off) for e in elecs]
    return names.flatten()

if __name__ == "__main__":
    # Parameters
    win_length = 0.05
    frame_shift = 0.01
    model_order = 4
    step_size = 5
    path_bids = r"/path/to/dataset"
    path_output = r"./features"
    participants = pd.read_csv(os.path.join(path_bids, "participants.tsv"), delimiter="\t")

    for p_id, participant in enumerate(participants['participant_id']):
        try:
            print(f"Processing participant: {participant}")
            nwb_path = os.path.join(path_bids, participant, 'ieeg', f'{participant}_task-wordProduction_ieeg.nwb')
            channels_path = os.path.join(path_bids, participant, 'ieeg', f'{participant}_task-wordProduction_channels.tsv')

            if not os.path.exists(nwb_path) or not os.path.exists(channels_path):
                print(f"NWB or channels file not found for {participant}. Skipping.")
                continue

            io = NWBHDF5IO(nwb_path, 'r')
            nwbfile = io.read()

            # Load data
            audio = nwbfile.acquisition['Audio'].data[:]
            audio_sr = 48000
            eeg = nwbfile.acquisition['iEEG'].data[:]
            eeg_sr = 1024
            words = nwbfile.acquisition['Stimulus'].data[:]
            words = np.array(words, dtype=str)
            io.close()

            # Process audio
            target_sr = 16000
            audio = scipy.signal.resample(audio, int(len(audio) * target_sr / audio_sr))
            audio_sr = target_sr
            scaled = audio / np.max(np.abs(audio))
            scaled_int16 = np.int16(scaled * 32767)

            os.makedirs(path_output, exist_ok=True)
            scipy.io.wavfile.write(os.path.join(path_output, f'{participant}_orig_audio.wav'), audio_sr, scaled_int16)

            # Extract features
            wavelet_features = extract_wavelet_coefficients(eeg, eeg_sr, window_length=win_length, frame_shift=frame_shift)
            prosody_features = extract_speech_prosody(scaled, audio_sr, window_length=win_length, frame_shift=frame_shift)
            mel_spectrogram = extract_mel_spectrogram(scaled_int16, audio_sr, window_length=win_length, frame_shift=frame_shift)

            # Combine features
            combined_features = np.concatenate((wavelet_features.reshape(wavelet_features.shape[0], -1), prosody_features), axis=1)

            # Save results
            np.save(os.path.join(path_output, f'{participant}_feat.npy'), combined_features)
            np.save(os.path.join(path_output, f'{participant}_spec.npy'), mel_spectrogram)
            np.save(os.path.join(path_output, f'{participant}_prosody.npy'), prosody_features)

        except Exception as e:
            print(f"Error processing participant {participant}: {e}")
            continue

    print("Finished processing all participants.")