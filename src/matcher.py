import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src.spectrogram import load_audio, compute_spectrogram
from src.fingerprint import get_peaks, generate_hashes, generate_single_hashes

def match_query(query_path, db_path="database.pkl", mode="paired", sr=11025):
    """
    Matches an audio query clip against the database.
    
    Returns:
    - predicted_song: Name of the song with the highest offset alignment count
    - scores: Dict mapping song names to their peak alignment scores
    - best_offsets: List of offsets for the predicted song (useful for histogram plotting)
    - match_details: Dict of additional info
    """
    # Load database
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file {db_path} not found. Please build it first.")
        
    with open(db_path, "rb") as f:
        db = pickle.load(f)
        
    paired_db = db["paired"]
    single_db = db["single"]
    all_songs = db["songs"]
    
    # Load query and compute spectrogram
    y, sample_rate = load_audio(query_path, sr=sr)
    if y is None:
        return None, {}, [], {}
        
    w_len, overlap = 1024, 512
    S, times, freqs = compute_spectrogram(y, window_len=w_len, overlap=overlap, sr=sr)
    peaks = get_peaks(S, sr=sr)
    
    offsets_by_song = {song: [] for song in all_songs}
    
    if mode == "paired":
        query_hashes = generate_hashes(peaks)
        for hash_val, t1_q in query_hashes:
            if hash_val in paired_db:
                for song_name, t1_s in paired_db[hash_val]:
                    offsets_by_song[song_name].append(t1_s - t1_q)
    elif mode == "single":
        query_hashes = generate_single_hashes(peaks)
        for f1, t1_q in query_hashes:
            if f1 in single_db:
                for song_name, t1_s in single_db[f1]:
                    offsets_by_song[song_name].append(t1_s - t1_q)
                    
    # Find peak score for each song
    scores = {}
    best_offset_vals = {}
    for song_name, offsets in offsets_by_song.items():
        if len(offsets) > 0:
            counts = Counter(offsets)
            best_offset, peak_count = counts.most_common(1)[0]
            scores[song_name] = peak_count
            best_offset_vals[song_name] = best_offset
        else:
            scores[song_name] = 0
            best_offset_vals[song_name] = 0
            
    # Find predicted song
    if len(scores) > 0:
        predicted_song = max(scores, key=scores.get)
        # Check if the highest score is greater than zero
        if scores[predicted_song] == 0:
            predicted_song = "Unknown / No Match"
    else:
        predicted_song = "Unknown / No Match"
        
    best_offsets = offsets_by_song.get(predicted_song, [])
    
    match_details = {
        "query_peaks_count": len(peaks),
        "total_matches_found": sum(len(offsets_by_song[s]) for s in all_songs),
        "best_offset": best_offset_vals.get(predicted_song, 0),
        "offsets_by_song": offsets_by_song
    }
    
    return predicted_song, scores, best_offsets, match_details

def plot_offset_histogram(offsets, song_name, output_path="report/images/offset_histogram.png"):
    """
    Plots the time-offset histogram for a matched song.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(offsets) == 0:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No offset alignments found", ha='center', va='center')
        plt.title(f"Offset Histogram: {song_name} (No Match)")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return
        
    plt.figure(figsize=(10, 4))
    
    # We use integer bins corresponding to frame indices
    min_offset = min(offsets) - 10
    max_offset = max(offsets) + 10
    bins = np.arange(min_offset, max_offset + 1)
    
    plt.hist(offsets, bins=bins, color='#e377c2', edgecolor='#7f7f7f', alpha=0.85)
    plt.title(f"Offset Histogram for Matched Song: {song_name}")
    plt.xlabel("Time Offset (Spectrogram Frames)")
    plt.ylabel("Number of Matching Hashes")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight the peak
    counts = Counter(offsets)
    best_offset, peak_count = counts.most_common(1)[0]
    plt.annotate(f"Peak: {peak_count} matches at offset {best_offset}",
                 xy=(best_offset, peak_count),
                 xytext=(best_offset + (max_offset-min_offset)*0.1, peak_count * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
                 
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved offset histogram to {output_path}.")
