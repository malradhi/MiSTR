import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

if __name__ == "__main__":
    # ============================
    #     File Paths & Loading
    # ============================
    result_dir = '/home/results/'
    features_dir = './features'
    participant = 'sub-08'

    # Load correlation results
    correlation_results = np.load(os.path.join(result_dir, 'transformer_autoencoder_results.npy'))
    print("Shape of correlation_results:", correlation_results.shape)

    # ============================
    #     Correlation Plotting
    # ============================
    # Calculate mean and std deviation
    if len(correlation_results.shape) == 2:
        mean_corr = np.mean(correlation_results, axis=1)
        std_corr = np.std(correlation_results, axis=1)
        spec_mean = correlation_results
        spec_std = np.zeros_like(correlation_results)
        spec_bins = np.arange(correlation_results.shape[1])

    elif len(correlation_results.shape) == 3:
        mean_corr = np.mean(correlation_results, axis=(1, 2))
        std_corr = np.std(correlation_results, axis=(1, 2))
        spec_mean = np.mean(correlation_results, axis=1)
        spec_std = np.std(correlation_results, axis=1)
        spec_bins = np.arange(correlation_results.shape[2])

    else:
        raise ValueError("Unexpected shape of correlation_results. Check your input file.")

    # Visualization settings
    colors = ['C' + str(i) for i in range(len(mean_corr))]
    x_range = range(len(mean_corr))
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # ---------------- Barplot of Average Results ----------------
    ax[0].bar(x_range, mean_corr, yerr=std_corr, alpha=0.5, color=colors)
    for idx in range(correlation_results.shape[0]):
        if len(correlation_results.shape) == 2:
            ax[0].scatter(np.full(correlation_results.shape[1], idx), correlation_results[idx, :], color=colors[idx])
        elif len(correlation_results.shape) == 3:
            ax[0].scatter(np.zeros(correlation_results[idx, :, :].shape[0]) + idx, 
                          np.mean(correlation_results[idx, :, :], axis=1), color=colors[idx])

    ax[0].set_xticks(x_range)
    ax[0].set_xticklabels([f'sub-{i+1:02d}' for i in x_range], rotation=45, ha='right', fontsize=20)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Correlation', fontsize=20)
    ax[0].set_title('a', fontsize=20, fontweight="bold")
    plt.setp(ax[0].spines.values(), linewidth=2)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # ---------------- Mean Across Folds or Spectral Bins ----------------
    for idx in range(correlation_results.shape[0]):
        ax[1].plot(spec_bins, spec_mean[idx, :], color=colors[idx])
        if len(correlation_results.shape) == 3:
            error = spec_std[idx, :] / np.sqrt(correlation_results.shape[1])
            ax[1].fill_between(spec_bins, spec_mean[idx, :] - error, spec_mean[idx, :] + error, 
                               alpha=0.5, color=colors[idx])

    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0, len(spec_bins))
    ax[1].set_xlabel('Spectral Bin' if len(correlation_results.shape) == 3 else 'Fold', fontsize=20)
    ax[1].set_ylabel('Correlation', fontsize=20)
    ax[1].set_title('b', fontsize=20, fontweight="bold")
    plt.setp(ax[1].spines.values(), linewidth=2)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'results.png'), dpi=600)
    plt.show()

    print("Visualization completed!")

    # ============================
    #     Spectrogram Plotting
    # ============================
    start_s, stop_s = 5.5, 19.5
    frameshift = 0.01
    eeg_sr = 1024

    # Load spectrograms
    rec_spec = np.load(os.path.join(result_dir, f'{participant}_predicted_spec.npy'))
    spectrogram = np.load(os.path.join(features_dir, f'{participant}_spec.npy'))

    # Load words
    words = np.load(os.path.join(features_dir, f'{participant}_procWords.npy'), allow_pickle=True)[
        int(start_s * eeg_sr):int(stop_s * eeg_sr)]
    words = [words[i] for i in np.arange(1, len(words)) if words[i] != words[i - 1] and words[i] != '']

    # Plot spectrograms
    cm = 'viridis'
    fig, ax = plt.subplots(2, sharex=True)
    start_frame = int(start_s * (1 / frameshift))
    stop_frame = int(stop_s * (1 / frameshift))
    ax[0].imshow(np.flipud(spectrogram[start_frame:stop_frame, :].T), cmap=cm, aspect='auto')
    ax[0].set_ylabel('Log Mel-Spec Bin')

    ax[1].imshow(np.flipud(rec_spec[start_frame:stop_frame, :].T), cmap=cm, aspect='auto')
    num_ticks = len(words)
    xticks = np.linspace(int(1 / frameshift), spectrogram[start_frame:stop_frame, :].shape[0], 
                         num=num_ticks, endpoint=False, dtype=int)
    plt.setp(ax[1], xticks=xticks, xticklabels=words)
    ax[1].set_ylabel('Log Mel-Spec Bin')

    plt.savefig(os.path.join(result_dir, 'spec_example.png'), dpi=600)
    plt.savefig(os.path.join(result_dir, 'spec_example.pdf'), transparent=True)
    plt.show()

    # ============================
    #     Waveform Plotting
    # ============================
    rate, audio = wavfile.read(os.path.join(result_dir, f'{participant}_orig_synthesized.wav'))
    rate, rec_audio = wavfile.read(os.path.join(result_dir, f'{participant}_predicted.wav'))

    orig = audio[int(start_s * rate):int(stop_s * rate)]
    rec = rec_audio[int(start_s * rate):int(stop_s * rate)]

    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(orig)
    axarr[1].plot(rec)

    # Adjust xticks to match words
    xts = np.linspace(0, orig.shape[0], num=num_ticks, endpoint=False, dtype=int)
    axarr[1].set_xticks(xts)
    axarr[1].set_xticklabels(words)

    axarr[0].set_xlim([0, orig.shape[0]])
    axarr[0].set_ylim([-np.max(np.abs(orig)), np.max(np.abs(orig))])
    axarr[1].set_ylim([-np.max(np.abs(rec)), np.max(np.abs(rec))])

    # Annotation for 3 seconds
    axarr[1].annotate("", xy=(xts[0], 27000), xycoords='data', xytext=(xts[1], 27000), textcoords='data',
                      arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    axarr[1].annotate("3 seconds", xy=((xts[0] + xts[1]) / 2, 22000), horizontalalignment='center')

    # Labels and formatting
    axarr[0].set_ylabel('Amplitude')
    axarr[0].set_yticks([])
    axarr[1].set_yticks([])
    axarr[1].set_ylabel('Amplitude')
    axarr[1].text(orig.shape[0], 0, 'Reconstruction', horizontalalignment='left', verticalalignment='center',
                  rotation='vertical')
    axarr[0].text(orig.shape[0], 0, 'Original', horizontalalignment='left', verticalalignment='center', 
                  rotation='vertical')

    for axes in axarr:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)

    plt.savefig(os.path.join(result_dir, 'wav_example.png'), dpi=600)
    plt.savefig(os.path.join(result_dir, 'wav_example.pdf'), transparent=True)
    plt.show()

    # ============================
    #     Prosody Feature Plotting
    # ============================
    prosody_path = os.path.join(features_dir, f"{participant}_prosody.npy")
    if os.path.exists(prosody_path):
        prosody_features = np.load(prosody_path)
        prosody_columns = ["Pitch", "Energy", "Shimmer", "Duration", "Phase Variability"] + \
                          [f"CFC_{i}" for i in range(prosody_features.shape[1] - 5)]

        fig, ax = plt.subplots(3, 2, figsize=(12, 15))
        ax = ax.flatten()

        for i, feature_name in enumerate(prosody_columns):
            if i >= len(ax):
                break
            ax[i].plot(prosody_features[:, i], label=feature_name)
            ax[i].set_title(feature_name)
            ax[i].set_xlabel("Time Frame")
            ax[i].set_ylabel("Value")
            ax[i].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{participant}_prosody_visualization.png"), dpi=300)
        plt.show()

    else:
        print(f"[Warning] Prosody features not found for {participant}.")
