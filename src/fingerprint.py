import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from src.spectrogram import load_audio, compute_spectrogram

def get_peaks(S, threshold_db=-20, neighborhood_size=(15, 15), max_peaks_per_sec=15, sr=11025, window_len=1024, overlap=512):
    """
    Find local maxima in the spectrogram.
    Returns a list of tuples: (time_frame_idx, freq_bin_idx) sorted chronologically.
    """
    # Convert spectrogram to dB scale
    S_db = 20 * np.log10(S + 1e-6)
    
    # 2D local maximum filter
    local_max = (S_db == maximum_filter(S_db, size=neighborhood_size))
    
    # Relative thresholding: keep peaks that are above relative threshold
    max_val = np.max(S_db)
    threshold = max(max_val + threshold_db, -60) # Ensure we don't pick up silent noise
    
    peaks_mask = local_max & (S_db > threshold)
    
    # Extract coordinates
    freq_indices, time_indices = np.where(peaks_mask)
    peaks = list(zip(time_indices, freq_indices))
    
    # Sort peaks by amplitude to cap density
    peak_amplitudes = [S_db[f, t] for t, f in peaks]
    sorted_peaks = [p for _, p in sorted(zip(peak_amplitudes, peaks), reverse=True)]
    
    # Calculate audio duration
    hop_len = window_len - overlap
    num_frames = S.shape[1]
    duration = num_frames * hop_len / sr
    
    # Cap peaks density
    max_peaks = int(max_peaks_per_sec * duration)
    if len(sorted_peaks) > max_peaks:
        sorted_peaks = sorted_peaks[:max_peaks]
        
    # Re-sort chronologically
    sorted_peaks.sort(key=lambda x: x[0])
    return sorted_peaks

def generate_hashes(peaks, min_dt=1, max_dt=30, fan_value=3):
    """
    Pairs nearby peaks in time to create hashes.
    Each hash is: (f1, f2, dt) -> value: t1
    Returns a list of tuples: ((f1, f2, dt), t1)
    """
    hashes = []
    num_peaks = len(peaks)
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        # Search target zone: peaks occurring slightly after t1
        count = 0
        for j in range(i + 1, num_peaks):
            t2, f2 = peaks[j]
            dt = t2 - t1
            
            if min_dt <= dt <= max_dt:
                hashes.append(((f1, f2, dt), t1))
                count += 1
                if count >= fan_value:
                    break
            elif dt > max_dt:
                # Chronologically sorted, so no more peaks will fit
                break
    return hashes

def generate_single_hashes(peaks):
    """
    Generates single-peak hashes for comparison.
    Each hash is: f1 -> value: t1
    Returns a list of tuples: (f1, t1)
    """
    return [(f, t) for t, f in peaks]

def plot_constellation_map(song_path, output_path="report/images/constellation.png", duration_limit=15):
    """
    Computes spectrogram and plots constellation map (peaks overlaid on spectrogram).
    Saves image to output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y, sr = load_audio(song_path, sr=11025)
    if y is None:
        return
        
    # Standard STFT params
    w_len, overlap = 1024, 512
    S, times, freqs = compute_spectrogram(y, window_len=w_len, overlap=overlap, sr=sr)
    peaks = get_peaks(S, sr=sr, window_len=w_len, overlap=overlap)
    
    # Convert peak frame/bin indices to time/frequency values
    hop_len = w_len - overlap
    peak_times = [t * hop_len / sr + (w_len / (2 * sr)) for t, f in peaks]
    peak_freqs = [f * sr / w_len for t, f in peaks]
    
    plt.figure(figsize=(10, 5))
    S_db = 20 * np.log10(S + 1e-6)
    plt.pcolormesh(times, freqs, S_db, shading='gouraud', cmap='magma', alpha=0.8)
    plt.scatter(peak_times, peak_freqs, color='cyan', marker='x', s=40, label='Peaks')
    
    plt.title(f"Constellation Map (Peaks) - {os.path.basename(song_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.ylim(0, 5000)
    if duration_limit:
        plt.xlim(0, duration_limit) # Zoom in for clarity
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved constellation map to {output_path}.")

def build_database(songs_dir, db_output_path="database.pkl", sr=11025):
    """
    Index all song files in songs_dir.
    Saves a serialized database dictionary of hashes to db_output_path.
    """
    paired_db = {}
    single_db = {}
    song_names = []
    
    if not os.path.exists(songs_dir):
        print(f"Songs directory {songs_dir} does not exist.")
        return False
        
    files = [f for f in os.listdir(songs_dir) if f.endswith(".mp3") or f.endswith(".wav")]
    print(f"Indexing {len(files)} songs from {songs_dir}...")
    
    for filename in files:
        song_name = os.path.splitext(filename)[0]
        song_names.append(song_name)
        song_path = os.path.join(songs_dir, filename)
        
        y, sample_rate = load_audio(song_path, sr=sr)
        if y is None:
            continue
            
        S, times, freqs = compute_spectrogram(y, window_len=1024, overlap=512, sr=sr)
        peaks = get_peaks(S, sr=sr)
        
        # 1. Paired peak hashes
        paired_hashes = generate_hashes(peaks)
        for hash_val, t1 in paired_hashes:
            if hash_val not in paired_db:
                paired_db[hash_val] = []
            paired_db[hash_val].append((song_name, t1))
            
        # 2. Single peak hashes
        single_hashes = generate_single_hashes(peaks)
        for f1, t1 in single_hashes:
            if f1 not in single_db:
                single_db[f1] = []
            single_db[f1].append((song_name, t1))
            
    database = {
        "paired": paired_db,
        "single": single_db,
        "songs": song_names
    }
    
    with open(db_output_path, "wb") as f:
        pickle.dump(database, f)
        
    print(f"Database built and saved to {db_output_path} containing {len(song_names)} songs.")
    print(f"Total paired hashes: {len(paired_db)}, total single hashes: {len(single_db)}")
    return True

if __name__ == "__main__":
    # Test execution
    build_database("data/songs", "database.pkl")
    song_dir = "data/songs"
    if os.path.exists(song_dir):
        songs = [f for f in os.listdir(song_dir) if f.endswith(".mp3")]
        if songs:
            plot_constellation_map(os.path.join(song_dir, songs[0]))
