import numpy as np
import scipy
import scipy.io.wavfile as wavfile
import reconstructWave as rW
from scipy.signal import stft, istft, butter, lfilter, windows
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model
from torch.cuda.amp import GradScaler, autocast
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav_write
import MelFilterBank as mel
from scipy.ndimage import gaussian_filter

# ========================================================
# ------------------- Audio Processing -------------------
# ========================================================

def neural_stft(audio_signal, frame_size=1024, overlap_factor=4):
    """Custom STFT function for neural vocoding"""
    hop_length = int(frame_size / overlap_factor)
    hann_window = windows.hann(frame_size + 1)[:-1]
    return np.array([np.fft.rfft(hann_window * audio_signal[i:i + frame_size]) 
                     for i in range(0, len(audio_signal) - frame_size, hop_length)])


def neural_istft(spectrogram, overlap_factor=4):
    """Custom ISTFT function for neural vocoding"""
    frame_size = (spectrogram.shape[1] - 1) * 2
    hop_length = int(frame_size / overlap_factor)
    hann_window = windows.hann(frame_size + 1)[:-1]
    
    reconstructed_signal = np.zeros(spectrogram.shape[0] * hop_length)
    window_sum = np.zeros(spectrogram.shape[0] * hop_length)

    for idx, start_idx in enumerate(range(0, len(reconstructed_signal) - frame_size, hop_length)):
        reconstructed_signal[start_idx:start_idx + frame_size] += np.real(np.fft.irfft(spectrogram[idx])) * hann_window
        window_sum[start_idx:start_idx + frame_size] += hann_window ** 2

    return reconstructed_signal


def spectral_normalization(spectrogram):
    """Normalize spectrogram for stable neural phase reconstruction"""
    spectrogram = spectrogram - np.min(spectrogram)
    spectrogram = spectrogram / (np.max(spectrogram) + 1e-8)
    return spectrogram


def spectral_smoothing(spectrogram, sigma=1.2):
    """Smooth spectrogram using Gaussian filtering to reduce artifacts"""
    return gaussian_filter(spectrogram, sigma=sigma)


def phase_initialization(spectrogram, seed=99):
    """Initialize phase with reduced randomness for improved convergence"""
    np.random.seed(seed)
    phase_matrix = np.random.randn(*spectrogram.shape) * 0.08
    return phase_matrix


def low_pass_filter(audio_signal, cutoff_freq=4000, sr=16000, order=4):
    """Apply low-pass filter to reduce high-frequency noise"""
    nyquist_rate = 0.5 * sr
    normalized_cutoff = cutoff_freq / nyquist_rate
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio_signal)


# ========================================================
# ------------ Iterative Neural Phase Reconstruction ------
# ========================================================

def neural_waveform_reconstruction(spectrogram, audio_length, frame_size=1024, overlap_factor=4, iterations=10):
    """Neural waveform reconstruction using refined Griffin-Lim algorithm"""
    print(f"Spectrogram shape: {spectrogram.shape}")

    # Step 1: Normalize and smooth spectrogram
    spectrogram = spectral_normalization(spectrogram)
    spectrogram = spectral_smoothing(spectrogram, sigma=1.2)

    # Step 2: Phase Initialization
    phase_matrix = phase_initialization(spectrogram)
    reconstructed_waveform = np.random.rand(audio_length)

    # Step 3: Iterative reconstruction loop
    phase_spectrum = None

    for _ in range(iterations):
        # Forward STFT
        stft_output = neural_stft(reconstructed_waveform, frame_size=frame_size, overlap_factor=overlap_factor)

        if stft_output.size == 0:
            print("[Warning] STFT returned empty output. Skipping iteration.")
            continue

        # Shape adjustment
        if stft_output.shape[0] != spectrogram.shape[0]:
            stft_output = stft_output[:spectrogram.shape[0], :]
            phase_matrix = phase_matrix[:spectrogram.shape[0], :]

        # Phase refinement
        phase_spectrum = spectrogram * np.exp(1j * (np.angle(stft_output) + phase_matrix))
        reconstructed_waveform = neural_istft(phase_spectrum, overlap_factor=overlap_factor)

    # Post-process waveform with low-pass filtering
    if phase_spectrum is not None:
        reconstructed_waveform = low_pass_filter(reconstructed_waveform, cutoff_freq=4000, sr=16000)
    else:
        print("[Warning] Reconstruction incomplete. Using initial random waveform.")

    return reconstructed_waveform[:audio_length]


# ========================================================
# ------------------- Audio Generation -------------------
# ========================================================

def generate_audio_from_spectrogram(spectrogram, sample_rate=16000, window_length=0.05, frame_shift=0.01):
    """Generate reconstructed audio from Mel spectrogram using neural vocoding"""
    if spectrogram.size == 0:
        print("[Warning] Input spectrogram is empty.")
        return np.array([])

    mel_filter_bank = mel.MelFilterBank(int((sample_rate * window_length) / 2 + 1), spectrogram.shape[1], sample_rate)
    num_segments = 10
    segment_length = int(spectrogram.shape[0] / num_segments)
    reconstructed_audio = np.array([])

    # Convert to linear spectrogram
    linear_spectrogram = mel_filter_bank.fromLogMels(spectrogram)

    for segment_start in range(0, spectrogram.shape[0], segment_length):
        segment = linear_spectrogram[segment_start:min(segment_start + segment_length, linear_spectrogram.shape[0]), :]
        if segment.size == 0:
            print(f"[Warning] Segment at index {segment_start} is empty. Skipping.")
            continue

        reconstructed_segment = neural_waveform_reconstruction(segment, segment.shape[0] * segment.shape[1],
                                                               frame_size=int(sample_rate * window_length),
                                                               overlap_factor=int(window_length / frame_shift))
        reconstructed_audio = np.append(reconstructed_audio, reconstructed_segment)

    if reconstructed_audio.size == 0:
        print("[Warning] No audio reconstructed.")
        return np.array([])

    # Final normalization
    reconstructed_audio = np.int16(reconstructed_audio / np.max(np.abs(reconstructed_audio)) * 32767)
    return reconstructed_audio


# ========================================================
# -------------------- Main Pipeline ---------------------
# ========================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File Paths
    data_path = "/project/features/"
    output_path = "/project/results/"

    participants = ['sub-%02d' % i for i in range(1, 11)]
    window_length = 0.05
    frame_shift = 0.01
    sample_rate = 16000

    # Initialize results matrix
    all_results = np.zeros((len(participants), 10))
    explained_variance = np.zeros((len(participants), 10))

    # Process each participant
    for idx, participant in enumerate(participants):
        print(f"\n[INFO] Processing participant: {participant}")

        # Load input spectrograms
        original_spectrogram = np.load(os.path.join(data_path, f'{participant}_spec.npy'))
        predicted_spectrogram = np.load(os.path.join(output_path, f'{participant}_predicted_spec.npy'))

        # Generate audio from ground-truth spectrogram
        original_audio = generate_audio_from_spectrogram(original_spectrogram, sample_rate=sample_rate,
                                                         window_length=window_length, frame_shift=frame_shift)
        wavfile.write(os.path.join(output_path, f'{participant}_orig_synthesized.wav'), int(sample_rate), original_audio)

        # Generate audio from predicted spectrogram
        reconstructed_audio = generate_audio_from_spectrogram(predicted_spectrogram, sample_rate=sample_rate,
                                                              window_length=window_length, frame_shift=frame_shift)
        wavfile.write(os.path.join(output_path, f'{participant}_predicted.wav'), int(sample_rate), reconstructed_audio)

    print("\n[INFO] Neural Speech Reconstruction Completed ✅")
