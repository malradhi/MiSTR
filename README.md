<!-- # MiSTR: Multi-Modal iEEG-to-Speech Synthesis with Transformer-Based Prosody Prediction and Neural Phase Reconstruction -->


<h1 align="center"><strong>MiSTR: Multi-Modal iEEG-to-Speech Synthesis with Transformer-Based Prosody Prediction and Neural Phase Reconstruction</strong></h1>

<p align="center" style="font-size: 1 em; margin-top: 1em">
<a href="https://malradhi.github.io/">Mohammed Salah Al-Radhi<sup>1</sup></a>,  
<a href="https://scholar.google.ro/citations?user=Qf5PHwoAAAAJ&hl=en/">Géza Németh<sup>1</sup></a>,  
<a href="https://gerazov.github.io/">Branislav Gerazov<sup>2</sup></a>,
</p>

<p align="center">
  <sup>1</sup>Department of Telecommunications and Artificial Intelligence, Budapest University of Technology and Economics, Budapest, Hungary<br>
  <sup>2</sup>Faculty of Electrical Engineering and Information Technologies, University of Ss. Cyril and Methodius, Skopje, Macedonia<br>
</p>

<!-- <div align="center">
  <a href="https://github.com/ZhikangNiu/A-DMA">
    <img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="Python">
  </a>
  <a href="https://arxiv.org/abs/2505.19595v1">
    <img src="https://img.shields.io/badge/arXiv-2505.19595-b31b1b.svg?logo=arXiv" alt="arXiv">
  </a>
  <a href="https://mm.kaist.ac.kr/projects/A-DMA">
    <img src="https://img.shields.io/badge/GitHub-Demo%20page-orange.svg" alt="Demo">
  </a>
</div
 -->
 
 
<br>
<br> 


## 📜 News
🧠 [2025.5.29] We are releasing all the code to support research on accelerating MiSTR multi-modal models.

🥳 [2025.05.19, 11:54 AM CET] Our paper has been accepted to [Interspeech 2025](https://www.interspeech2025.org/home). Looking forward to seeing you in Rotterdam, The Netherlands at the conference!

<br>
<br> 

## 💡 Highlights
1. **Multi-Modal iEEG Feature Encoding**: MiSTR introduces a wavelet-based encoder combined with prosody-aware features (pitch, energy, shimmer, duration, phase variability) to model the neural dynamics of speech production.
2. **Transformer-Based Prosody Decoder**: A novel Transformer architecture captures long-range dependencies in brain activity to predict expressive and fluent Mel spectrograms aligned with speech prosody.
3. **Neural Phase Vocoder (IHPR)**: MiSTR proposes Iterative Harmonic Phase Reconstruction (IHPR), ensuring phase continuity and harmonic consistency for high-fidelity audio synthesis without vocoder artifacts.
4. **State-of-the-Art Performance**: Achieves a Pearson correlation of 0.91, STOI of 0.73, and MOSA score of 3.38, outperforming all existing baselines in iEEG-to-speech synthesis.
5. **Clinically Inspired Design**: Designed with speech neuroprosthetics in mind, MiSTR offers a scalable, robust pipeline for restoring natural speech in individuals with severe communication impairments.
6. **Code and Samples Available**: Full implementation, pretrained models, and inference samples are provided in this GitHub repository to support reproducibility and further research.



