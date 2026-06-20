import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

def load_audio(filepath, sr=11025):
    """
    Load an audio file and resample it to the target sampling rate in mono.
    """
    try:
        # sr=None preserves native sampling rate; we force resampling to sr
        y, sample_rate = librosa.load(filepath, sr=sr, mono=True)
        return y, sample_rate
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def compute_dft(y):
    """
    Compute the DFT magnitude and corresponding frequency bins of the entire signal.
    """
    N = len(y)
    # We only take the positive half of the spectrum for real signals
    dft_val = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=1.0) # Normalised frequency or bin numbers; we can scale it to Hz later
    magnitude = np.abs(dft_val)
    return freqs, magnitude

def compute_spectrogram(y, window_len=1024, overlap=512, sr=11025):
    """
    Manually computes the Short-Time Fourier Transform (STFT) spectrogram of a signal.
    Uses a Hanning window.
    
    Returns:
    - S: 2D numpy array of shape (freq_bins, time_frames) representing magnitude
    - times: 1D array of time at each frame center (seconds)
    - freqs: 1D array of frequency at each bin (Hz)
    """
    hop_len = window_len - overlap
    num_samples = len(y)
    
    # Generate Hanning window
    win = np.hanning(window_len)
    
    # Determine the number of frames
    num_frames = int(np.floor((num_samples - window_len) / hop_len)) + 1
    if num_frames <= 0:
        # If signal is shorter than window, pad it
        y = np.pad(y, (0, window_len - num_samples), mode='constant')
        num_frames = 1
        
    num_freq_bins = window_len // 2 + 1
    S = np.zeros((num_freq_bins, num_frames))
    
    for i in range(num_frames):
        start_idx = i * hop_len
        end_idx = start_idx + window_len
        frame = y[start_idx:end_idx] * win
        # Compute real FFT
        rfft_vals = np.fft.rfft(frame)
        S[:, i] = np.abs(rfft_vals)
        
    # Calculate time and frequency axes
    freqs = np.fft.rfftfreq(window_len, d=1.0 / sr)
    times = np.arange(num_frames) * hop_len / sr + (window_len / (2 * sr))
    
    return S, times, freqs

def plot_dft_and_spectrograms(song_path, output_dir="report/images"):
    """
    Generates and saves:
    1. Full DFT magnitude plot
    2. Spectrogram with short window (256 samples)
    3. Spectrogram with long window (4096 samples)
    4. Spectrogram with default window (1024 samples)
    """
    os.makedirs(output_dir, exist_ok=True)
    y, sr = load_audio(song_path, sr=11025)
    if y is None:
        print("Failed to load audio for plotting.")
        return
        
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    duration = len(y) / sr
    
    # 1. Plot DFT of Entire Song
    plt.figure(figsize=(10, 4))
    freqs, magnitude = compute_dft(y)
    # Convert freqs to Hz
    freqs_hz = freqs * sr
    plt.plot(freqs_hz, magnitude, color='#1f77b4', alpha=0.8)
    plt.title(f"DFT Magnitude of Entire Song: {song_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    dft_path = os.path.join(output_dir, "dft_entire_song.png")
    plt.savefig(dft_path, dpi=200)
    plt.close()
    
    # Define window sizes for comparison
    windows = {
        "short": (256, 128, "Short Window (N=256, ~23ms)"),
        "default": (1024, 512, "Default Window (N=1024, ~93ms)"),
        "long": (4096, 2048, "Long Window (N=4096, ~372ms)")
    }
    
    paths = {"dft": dft_path}
    for name, (w_len, overlap, title) in windows.items():
        S, times, freqs_bins = compute_spectrogram(y, window_len=w_len, overlap=overlap, sr=sr)
        
        plt.figure(figsize=(10, 4))
        # Use log magnitude spectrogram for visualization
        S_db = 20 * np.log10(S + 1e-6)
        
        # Plot spectrogram
        plt.pcolormesh(times, freqs_bins, S_db, shading='gouraud', cmap='viridis')
        plt.title(f"Spectrogram with {title} - {song_name}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude (dB)")
        plt.ylim(0, 5000) # Show up to 5 kHz
        plt.tight_layout()
        
        img_path = os.path.join(output_dir, f"spectrogram_{name}.png")
        plt.savefig(img_path, dpi=200)
        plt.close()
        paths[name] = img_path
        
    print(f"Generated spectrogram and DFT plots for {song_name}.")
    return paths

if __name__ == "__main__":
    # Test execution
    song_dir = "data/songs"
    if os.path.exists(song_dir):
        songs = [f for f in os.listdir(song_dir) if f.endswith(".mp3")]
        if songs:
            test_song = os.path.join(song_dir, songs[0])
            plot_dft_and_spectrograms(test_song)
