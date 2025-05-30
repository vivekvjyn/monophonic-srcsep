import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class HarmonicSynthesizer(nn.Module):
    def __init__(self, n_harmonics=64, sample_rate=16000):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate

    def forward(self, f0, harmonic_amps):
        """
        f0: [batch, time]
        harmonic_amps: [batch, n_harmonics, time]
        """
        batch_size, n_harmonics, time = harmonic_amps.shape

        # Ensure amplitudes are positive and normalized
        harmonic_amps = F.relu(harmonic_amps)
        harmonic_amps /= (harmonic_amps.sum(dim=1, keepdim=True) + 1e-6)

        # Time vector
        t = torch.linspace(0, time / self.sample_rate, steps=time, device=f0.device)  # [time]

        audio = torch.zeros((batch_size, time), device=f0.device)

        for i in range(self.n_harmonics):
            harmonic_freq = (i + 1) * f0  # [batch, time]
            phase = 2 * torch.pi * harmonic_freq * t.unsqueeze(0)  # [batch, time]
            sine_wave = torch.sin(phase)
            audio += harmonic_amps[:, i, :] * sine_wave

        return audio

# === Noise Synthesizer ===

class NoiseSynthesizer(nn.Module):
    def __init__(self, n_bins=64, sample_rate=16000):
        super().__init__()
        self.n_bins = n_bins
        self.sample_rate = sample_rate

    def forward(self, noise_mag):
        """
        noise_mag: [batch, n_bins, time]
        """
        batch_size, n_bins, time = noise_mag.shape
        device = noise_mag.device

        # Generate white noise
        white_noise = torch.randn(batch_size, time, device=device)

        # Apply a basic noise shaping filter
        shaped_noise = torch.zeros_like(white_noise)

        bin_size = time // n_bins

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else time
            shaped_noise[:, start:end] = white_noise[:, start:end] * noise_mag[:, i, :].mean(dim=1, keepdim=True)

        return shaped_noise

class Separator(nn.Module):
    def __init__(self, n_harmonics=64, n_noise_bins=64, sample_rate=16000):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bins = n_noise_bins
        self.sample_rate = sample_rate

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # <-- output 1 frame
            nn.Flatten(start_dim=1),
            nn.Linear(128, self.n_harmonics + self.n_noise_bins)  # <-- predict one vector
        )

        self.harmonic_synth = HarmonicSynthesizer(n_harmonics=n_harmonics, sample_rate=sample_rate)
        self.noise_synth = NoiseSynthesizer(n_bins=n_noise_bins, sample_rate=sample_rate)

    def forward(self, audio, f0):
        """
        audio: [batch, time]
        f0: [batch, time]
        """
        batch_size, time = audio.shape

        x = audio.unsqueeze(1)  # [batch, 1, time]
        features = self.encoder(x)  # [batch, n_harmonics + n_noise_bins]

        harmonic_amps = features[:, :self.n_harmonics]  # [batch, n_harmonics]
        noise_mag = features[:, self.n_harmonics:]      # [batch, n_noise_bins]

        # Expand dims to match [batch, n_harmonics, time]
        harmonic_amps = harmonic_amps.unsqueeze(-1).expand(-1, -1, time)
        noise_mag = noise_mag.unsqueeze(-1).expand(-1, -1, time)

        # Synthesize
        harmonic_audio = self.harmonic_synth(f0, harmonic_amps)
        noise_audio = self.noise_synth(noise_mag)

        output_audio = harmonic_audio + noise_audio

        return output_audio
