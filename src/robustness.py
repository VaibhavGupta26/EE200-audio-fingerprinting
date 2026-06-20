import os
import random
import pickle
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import soundfile as sf
from src.spectrogram import load_audio, compute_spectrogram
from src.fingerprint import get_peaks, generate_hashes
from src.matcher import match_query

def add_noise(y, snr_db):
    """
    Add additive white Gaussian noise (AWGN) to a signal for a given SNR in dB.
    """
    sig_power = np.mean(y ** 2)
    if sig_power == 0:
        return y
    # SNR = P_signal / P_noise => P_noise = P_signal / 10^(SNR_dB/10)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def speed_shift(y, shift_percent):
    """
    Shift the pitch and time of a signal by a given percentage using resampling.
    A positive percent increases pitch and speeds up time (time contraction).
    e.g. shift_percent = 2 means 2% pitch increase (speed factor 1.02).
    """
    if shift_percent == 0:
        return y
    rate = 1.0 + (shift_percent / 100.0)
    N = len(y)
    x = np.arange(N)
    # Resample the signal using linear interpolation
    new_x = np.linspace(0, N - 1, int(N / rate))
    f_interp = scipy.interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f_interp(new_x)

def time_stretch(y, stretch_percent):
    """
    Stretches time without changing pitch, using a simple phase vocoder.
    We can also do simple resampling for simplicity, but if the spec separates them,
    speed_shift changes pitch+time.
    If we want pure time-stretch or pure pitch-shift, we can use librosa.effects.time_stretch.
    For this project, resampling (which changes both pitch and speed) is the most standard
    and robust mathematical approach to test. Let's provide resampling as default speed shift.
    """
    rate = 1.0 + (stretch_percent / 100.0)
    # Simple linear interpolation (which shifts pitch too, but works as a test)
    N = len(y)
    x = np.arange(N)
    new_x = np.linspace(0, N-1, int(N * rate))
    f_interp = scipy.interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f_interp(new_x)

def generate_test_query(song_path, duration=5.0, sr=11025, noise_db=None, pitch_percent=0.0):
    """
    Generates a query clip from a song: extracts a 5-second slice from the middle,
    adds noise (optional), and shifts pitch/speed (optional).
    """
    y, sample_rate = load_audio(song_path, sr=sr)
    if y is None or len(y) < duration * sr:
        return None
        
    # Take a 5-second slice from the middle of the song to avoid silence at start/end
    total_duration = len(y) / sr
    start_time = max(0.0, (total_duration / 2) - (duration / 2))
    start_idx = int(start_time * sr)
    end_idx = start_idx + int(duration * sr)
    
    query = y[start_idx:end_idx].copy()
    
    if pitch_percent != 0.0:
        query = speed_shift(query, pitch_percent)
        
    if noise_db is not None:
        query = add_noise(query, noise_db)
        
    return query

def run_experiments(songs_dir, db_path="database.pkl", output_dir="report/images", sr=11025):
    """
    Runs the experiments required for Q3A:
    1. Single peak vs paired hash matching comparison on clean queries
    2. Robustness sweep over SNR (30 dB down to -10 dB)
    3. Robustness sweep over Pitch/Speed Shift (up to ±5%)
    Saves plots to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify database exists
    if not os.path.exists(db_path):
        from src.fingerprint import build_database
        build_database(songs_dir, db_path, sr)
        
    files = [f for f in os.listdir(songs_dir) if f.endswith(".mp3")]
    if not files:
        print("No songs found to run experiments.")
        return
        
    # We will run tests on a subset of 5 random songs for efficiency
    random.seed(42)
    test_files = random.sample(files, min(5, len(files)))
    
    # Temporary test query path
    temp_query_path = "temp_query.wav"
    
    print("\n--- EXPERIMENT 1: Paired Hashes vs Single Peaks (Clean Queries) ---")
    paired_success = 0
    single_success = 0
    total_tests = 0
    
    for filename in test_files:
        song_name = os.path.splitext(filename)[0]
        song_path = os.path.join(songs_dir, filename)
        
        # Generate a clean 5s query
        query_audio = generate_test_query(song_path, duration=5.0, sr=sr)
        if query_audio is None:
            continue
            
        sf.write(temp_query_path, query_audio, sr)
        total_tests += 1
        
        # Match using Paired Hashes
        pred_p, _, _, _ = match_query(temp_query_path, db_path, mode="paired", sr=sr)
        if pred_p == song_name:
            paired_success += 1
            
        # Match using Single Peaks
        pred_s, _, _, _ = match_query(temp_query_path, db_path, mode="single", sr=sr)
        if pred_s == song_name:
            single_success += 1
            
    print(f"Paired Hashes Success: {paired_success}/{total_tests} ({paired_success/total_tests*100:.1f}%)")
    print(f"Single Peaks Success: {single_success}/{total_tests} ({single_success/total_tests*100:.1f}%)")
    
    # 2. SNR Noise Sweep Experiment
    print("\n--- EXPERIMENT 2: Noise Robustness Sweep ---")
    snr_levels = [30, 20, 15, 10, 5, 0, -5, -10]
    paired_snr_acc = []
    single_snr_acc = []
    
    for snr in snr_levels:
        p_ok = 0
        s_ok = 0
        valid_tests = 0
        
        for filename in test_files:
            song_name = os.path.splitext(filename)[0]
            song_path = os.path.join(songs_dir, filename)
            
            query_audio = generate_test_query(song_path, duration=5.0, sr=sr, noise_db=snr)
            if query_audio is None:
                continue
                
            sf.write(temp_query_path, query_audio, sr)
            valid_tests += 1
            
            # Paired
            pred_p, _, _, _ = match_query(temp_query_path, db_path, mode="paired", sr=sr)
            if pred_p == song_name:
                p_ok += 1
                
            # Single
            pred_s, _, _, _ = match_query(temp_query_path, db_path, mode="single", sr=sr)
            if pred_s == song_name:
                s_ok += 1
                
        paired_acc = (p_ok / valid_tests) * 100
        single_acc = (s_ok / valid_tests) * 100
        paired_snr_acc.append(paired_acc)
        single_snr_acc.append(single_acc)
        print(f"SNR: {snr:3d} dB | Paired Accuracy: {paired_acc:5.1f}% | Single Accuracy: {single_acc:5.1f}%")
        
    # Plot Noise Sweep
    plt.figure(figsize=(8, 5))
    plt.plot(snr_levels, paired_snr_acc, 'o-', linewidth=2, color='#1f77b4', label='Paired Hashes')
    plt.plot(snr_levels, single_snr_acc, 's--', linewidth=1.5, color='#d62728', label='Single Peaks')
    plt.title("Noise Robustness Sweep: Song Identification Accuracy vs. SNR")
    plt.xlabel("SNR of Query (dB)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().invert_xaxis() # Show higher noise to the right
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_noise_sweep.png"), dpi=200)
    plt.close()
    
    # 3. Pitch / Speed Shift Sweep Experiment
    print("\n--- EXPERIMENT 3: Pitch/Speed Shift Sweep ---")
    # Shift in percent (e.g. -4% to +4%)
    shifts = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    paired_shift_acc = []
    single_shift_acc = []
    
    for shift in shifts:
        p_ok = 0
        s_ok = 0
        valid_tests = 0
        
        for filename in test_files:
            song_name = os.path.splitext(filename)[0]
            song_path = os.path.join(songs_dir, filename)
            
            query_audio = generate_test_query(song_path, duration=5.0, sr=sr, pitch_percent=shift)
            if query_audio is None:
                continue
                
            sf.write(temp_query_path, query_audio, sr)
            valid_tests += 1
            
            # Paired
            pred_p, _, _, _ = match_query(temp_query_path, db_path, mode="paired", sr=sr)
            if pred_p == song_name:
                p_ok += 1
                
            # Single
            pred_s, _, _, _ = match_query(temp_query_path, db_path, mode="single", sr=sr)
            if pred_s == song_name:
                s_ok += 1
                
        paired_acc = (p_ok / valid_tests) * 100
        single_acc = (s_ok / valid_tests) * 100
        paired_shift_acc.append(paired_acc)
        single_shift_acc.append(single_acc)
        print(f"Shift: {shift:+5.1f}% | Paired Accuracy: {paired_acc:5.1f}% | Single Accuracy: {single_acc:5.1f}%")
        
    # Plot Pitch Sweep
    plt.figure(figsize=(8, 5))
    plt.plot(shifts, paired_shift_acc, 'o-', linewidth=2, color='#2ca02c', label='Paired Hashes')
    plt.plot(shifts, single_shift_acc, 's--', linewidth=1.5, color='#ff7f0e', label='Single Peaks')
    plt.title("Pitch/Speed Shift Sweep: Song Identification Accuracy vs. Shift")
    plt.xlabel("Pitch/Speed Shift (%)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_pitch_sweep.png"), dpi=200)
    plt.close()
    
    # Cleanup temp file
    if os.path.exists(temp_query_path):
        os.remove(temp_query_path)
        
    print("Experiments complete. Plots saved to output folder.")
    return {
        "snr_levels": snr_levels,
        "paired_snr_acc": paired_snr_acc,
        "single_snr_acc": single_snr_acc,
        "shifts": shifts,
        "paired_shift_acc": paired_shift_acc,
        "single_shift_acc": single_shift_acc
    }

if __name__ == "__main__":
    run_experiments("data/songs", "database.pkl")
