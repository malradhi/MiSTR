import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import reconstructWave as rW
import MelFilterBank as mel

# Define Neural Compression Network (Autoencoder)
class NeuralCompressor(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(NeuralCompressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_neural_compressor(train_data, input_dim, latent_dim, device, num_epochs=50, batch_size=32):
    """
    Train the Neural Compressor for dimensionality reduction.
    """
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    model = NeuralCompressor(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            _, decoded = model(batch_data)
            loss = loss_fn(decoded, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    model.eval()
    return model

# Define Temporal Attention Network (Transformer)
class TemporalAttentionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, num_layers=2, dim_feedforward=None, dropout=0.1):
        super(TemporalAttentionNetwork, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = input_dim  # Default to input_dim if not provided
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, dim_feedforward))  # Max sequence length
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.output_proj = nn.Linear(dim_feedforward, output_dim)

    def forward(self, x):
        x = self.input_proj(x)  # Shape: (batch_size, seq_length, dim_feedforward)
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_length, dim_feedforward)
        x = self.output_proj(x)  # Shape: (batch_size, seq_length, output_dim)
        return x

def train_temporal_attention(train_data, train_labels, test_data, input_dim, output_dim, device, num_epochs=50, batch_size=32):
    """
    Train the Temporal Attention Network for spectrogram prediction.
    """
    train_data = train_data.reshape(train_data.shape[0], 1, input_dim)  # Add sequence dimension
    test_data = test_data.reshape(test_data.shape[0], 1, input_dim)

    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

    model = TemporalAttentionNetwork(input_dim=input_dim, output_dim=output_dim, dim_feedforward=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            predictions = model(batch_data)  # Shape: (batch_size, seq_length=1, output_dim)
            predictions = predictions.mean(dim=1)  # Collapse sequence dimension
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Testing
    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor)  # Shape: (batch_size, seq_length=1, output_dim)
        predictions = predictions.mean(dim=1).cpu().numpy()  # Collapse sequence dimension

    return predictions

def synthesize_audio(spectrogram, audio_sr=16000, win_length=0.05, frame_shift=0.01):
    """
    Synthesize audio waveform from a spectrogram.
    """
    mfb = mel.MelFilterBank(int((audio_sr * win_length) / 2 + 1), spectrogram.shape[1], audio_sr)
    nfolds = 10
    hop = int(spectrogram.shape[0] / nfolds)
    reconstructed_audio = np.array([])
    for_reconstruction = mfb.fromLogMels(spectrogram)

    for w in range(0, spectrogram.shape[0], hop):
        spec = for_reconstruction[w:min(w + hop, for_reconstruction.shape[0]), :]
        rec = rW.reconstructWavFromSpectrogram(spec, spec.shape[0] * spec.shape[1], fftsize=int(audio_sr * win_length),
                                               overlap=int(win_length / frame_shift))
        reconstructed_audio = np.append(reconstructed_audio, rec)

    scaled = np.int16(reconstructed_audio / np.max(np.abs(reconstructed_audio)) * 32767)
    return scaled

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_path = r'./features'
    result_path = r'./results'
    participants = [f'sub-{i:02d}' for i in range(1, 11)]

    win_length = 0.05
    frame_shift = 0.01
    audio_sr = 16000
    nfolds = 10
    latent_dim = 32  # Latent space dimension

    # Load example spectrogram to determine the number of Mel bands
    spectrogram_example = np.load(os.path.join(feat_path, f'{participants[0]}_spec.npy'))
    num_mel_bands = spectrogram_example.shape[1]

    all_results = np.zeros((len(participants), nfolds, num_mel_bands))

    # Main Loop
    for p_id, participant in enumerate(participants):
        try:
            print(f"Processing participant: {participant}")
            spectrogram = np.load(os.path.join(feat_path, f'{participant}_spec.npy'))
            data = np.load(os.path.join(feat_path, f'{participant}_feat.npy'))

            # Normalize data
            mu = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1e-8
            normalized_data = (data - mu) / std

            # Train Neural Compressor
            neural_compressor = train_neural_compressor(normalized_data, input_dim=data.shape[1], latent_dim=latent_dim, device=device)

            # Encode data
            encoded_data = neural_compressor.encoder(torch.tensor(normalized_data, dtype=torch.float32).to(device)).detach().cpu().numpy()

            # K-Fold Cross-Validation
            rec_spec = np.zeros(spectrogram.shape)
            kf = KFold(n_splits=nfolds, shuffle=False)

            for k, (train_idx, test_idx) in enumerate(kf.split(encoded_data)):
                train_data, test_data = encoded_data[train_idx], encoded_data[test_idx]
                train_labels, test_labels = spectrogram[train_idx], spectrogram[test_idx]

                # Train Temporal Attention Network
                predictions = train_temporal_attention(
                    train_data=train_data,
                    train_labels=train_labels,
                    test_data=test_data,
                    input_dim=latent_dim,
                    output_dim=num_mel_bands,
                    device=device
                )

                rec_spec[test_idx] = predictions

                # Calculate correlations
                for spec_bin in range(num_mel_bands):
                    if not (np.all(spectrogram[test_idx, spec_bin] == spectrogram[test_idx, spec_bin][0]) or
                            np.all(predictions[:, spec_bin] == predictions[0, spec_bin])):
                        r, _ = pearsonr(spectrogram[test_idx, spec_bin], predictions[:, spec_bin])
                        all_results[p_id, k, spec_bin] = r if not np.isnan(r) else 0.0

            print(f'{participant} mean correlation: {np.nanmean(all_results[p_id, :, :]):.4f}')

            # Save results and audio
            os.makedirs(result_path, exist_ok=True)
            np.save(os.path.join(result_path, f'{participant}_predicted_spec.npy'), rec_spec)
            reconstructed_wav = synthesize_audio(rec_spec, audio_sr=audio_sr, win_length=win_length, frame_shift=frame_shift)
            wavfile.write(os.path.join(result_path, f'{participant}_predicted.wav'), int(audio_sr), reconstructed_wav)
            orig_wav = synthesize_audio(spectrogram, audio_sr=audio_sr, win_length=win_length, frame_shift=frame_shift)
            wavfile.write(os.path.join(result_path, f'{participant}_orig_synthesized.wav'), int(audio_sr), orig_wav)

        except Exception as e:
            print(f"Error processing participant {participant}: {e}")
            continue

    np.save(os.path.join(result_path, 'temporal_attention_results.npy'), all_results)
    print("Processing completed!")