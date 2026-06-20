import os
import sys
import tempfile

# Add project root directory to path to allow resolution of 'src'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from src.spectrogram import load_audio, compute_spectrogram
from src.fingerprint import get_peaks, build_database
from src.matcher import match_query

# Configure Streamlit page
st.set_page_config(
    page_title="Zapptain America - Sonic Signatures",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Sleek Dark theme overrides)
st.markdown("""
<style>
    /* Premium background styling */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
        color: #f0f2f6;
    }
    
    /* Sleek card elements */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        margin-bottom: 20px;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Success text */
    .success-text {
        font-weight: bold;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Application constants
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "database.pkl")
SONGS_DIR = os.path.join(PROJECT_ROOT, "data", "songs")

# Helper function to check/build database
@st.cache_resource
def initialize_database():
    if not os.path.exists(DB_PATH):
        st.warning("Database file not found. Indexing database songs (this happens once at startup)...")
        success = build_database(SONGS_DIR, DB_PATH)
        if success:
            st.success("Database indexed and ready!")
        else:
            st.error("Failed to build song database. Please check data/songs folder.")

# Run initialization
initialize_database()

# Load DB metadata to list indexed songs in sidebar
if os.path.exists(DB_PATH):
    import pickle
    with open(DB_PATH, "rb") as f:
        db_meta = pickle.load(f)
    indexed_songs = sorted(db_meta["songs"])
else:
    indexed_songs = []

# Sidebar navigation and info
st.sidebar.title("🎵 Sonic Signatures")
st.sidebar.markdown("An EE200 Signals, Systems & Networks project for audio fingerprinting and Shazam-style search.")

mode = st.sidebar.radio("Navigation", ["Single-Clip Mode", "Batch Mode", "Database Info"])

st.sidebar.markdown("---")
st.sidebar.subheader("Database Status")
if indexed_songs:
    st.sidebar.success(f"Indexed {len(indexed_songs)} songs")
    with st.sidebar.expander("View Indexed Songs"):
        for s in indexed_songs:
            st.sidebar.markdown(f"- {s}")
else:
    st.sidebar.error("Database unindexed")

# --- DATABASE INFO SECTION ---
if mode == "Database Info":
    st.title("🎵 Indexed Song Library")
    st.markdown("This database is built by extracting spectral peak fingerprints from the unmodified audio files in `data/songs/`.")
    
    if indexed_songs:
        col1, col2 = st.columns(2)
        half = len(indexed_songs) // 2 + 1
        with col1:
            for s in indexed_songs[:half]:
                st.markdown(f"🔹 **{s}**")
        with col2:
            for s in indexed_songs[half:]:
                st.markdown(f"🔹 **{s}**")
    else:
        st.info("No songs indexed yet. Put files in data/songs/ and refresh.")

# --- SINGLE-CLIP IDENTIFICATION MODE ---
elif mode == "Single-Clip Mode":
    st.title("⚡ Single-Clip Identification")
    st.markdown("Upload a noisy or clean audio snippet. The system will extract its spectral peaks, build paired hashes, and align them against the indexed database.")
    
    uploaded_file = st.file_uploader("Upload audio clip (.mp3, .wav)", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        # Save file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            
        st.audio(uploaded_file)
        
        with st.spinner("Processing audio and querying database..."):
            try:
                # Run the matcher
                predicted_song, scores, best_offsets, match_details = match_query(
                    tmp_path, db_path=DB_PATH, mode="paired", sr=11025
                )
                
                # Render identification card
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                if predicted_song == "Unknown / No Match" or scores.get(predicted_song, 0) < 5:
                    st.error("❌ Identification Failed: Audio could not be matched with high confidence.")
                else:
                    st.markdown(f"### 🎉 Match Identified: <span class='success-text'>{predicted_song}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score (offset peak height):** {scores.get(predicted_song, 0)} matching hashes")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # --- VISUALIZATIONS SECTION ---
                st.subheader("📊 Intermediate Processing Steps")
                
                # Load audio data for plots
                y, sr = load_audio(tmp_path, sr=11025)
                w_len, overlap = 1024, 512
                hop_len = w_len - overlap
                
                if y is not None:
                    # 1. Compute Spectrogram & Peaks
                    S, times, freqs = compute_spectrogram(y, window_len=w_len, overlap=overlap, sr=sr)
                    peaks = get_peaks(S, sr=sr, window_len=w_len, overlap=overlap)
                    
                    # Convert peak indices to physical units
                    peak_times = [t * hop_len / sr + (w_len / (2 * sr)) for t, f in peaks]
                    peak_freqs = [f * sr / w_len for t, f in peaks]
                    
                    col1, col2 = st.columns(2)
                    
                    # Col 1: Spectrogram & Peaks Constellation Map
                    with col1:
                        st.write("##### 1. Spectrogram & Constellation of Peaks")
                        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                        S_db = 20 * np.log10(S + 1e-6)
                        im = ax1.pcolormesh(times, freqs, S_db, shading='gouraud', cmap='viridis', alpha=0.85)
                        ax1.scatter(peak_times, peak_freqs, color='cyan', marker='x', s=45, label='Detected Peaks')
                        ax1.set_xlabel("Time (seconds)")
                        ax1.set_ylabel("Frequency (Hz)")
                        ax1.set_ylim(0, 5000)
                        ax1.legend()
                        fig1.colorbar(im, ax=ax1, label="dB Magnitude")
                        fig1.patch.set_facecolor('#0e1117')
                        ax1.set_facecolor('#0e1117')
                        ax1.spines['bottom'].set_color('#ffffff')
                        ax1.spines['left'].set_color('#ffffff')
                        ax1.xaxis.label.set_color('#ffffff')
                        ax1.yaxis.label.set_color('#ffffff')
                        ax1.title.set_color('#ffffff')
                        ax1.tick_params(colors='#ffffff')
                        plt.tight_layout()
                        st.pyplot(fig1)
                        
                    # Col 2: Match Alignment Offset Histogram
                    with col2:
                        st.write("##### 2. Match Alignment Offset Histogram")
                        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                        
                        if len(best_offsets) > 0:
                            min_offset = min(best_offsets) - 10
                            max_offset = max(best_offsets) + 10
                            bins = np.arange(min_offset, max_offset + 1)
                            ax2.hist(best_offsets, bins=bins, color='#e377c2', edgecolor='#7f7f7f', alpha=0.9)
                            
                            # Annotate peak
                            counts = Counter(best_offsets)
                            best_offset_val, peak_count = counts.most_common(1)[0]
                            ax2.axvline(best_offset_val, color='cyan', linestyle='--', alpha=0.8)
                            ax2.text(best_offset_val + (max_offset-min_offset)*0.02, peak_count * 0.9, 
                                     f"Peak: {peak_count}\nat offset {best_offset_val}", color='cyan', fontweight='bold')
                        else:
                            ax2.text(0.5, 0.5, "No offset alignments found", ha='center', va='center', color='white')
                            
                        ax2.set_xlabel("Relative Time Offset (frames)")
                        ax2.set_ylabel("Match Frequency Count")
                        fig2.patch.set_facecolor('#0e1117')
                        ax2.set_facecolor('#0e1117')
                        ax2.spines['bottom'].set_color('#ffffff')
                        ax2.spines['left'].set_color('#ffffff')
                        ax2.xaxis.label.set_color('#ffffff')
                        ax2.yaxis.label.set_color('#ffffff')
                        ax2.tick_params(colors='#ffffff')
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
            except Exception as e:
                st.error(f"Error executing match: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- BATCH RUNNING MODE ---
elif mode == "Batch Mode":
    st.title("📂 Batch Song Identification")
    st.markdown("Upload multiple query clips at once. The app will process each clip and export a standardized `results.csv` containing the predictions.")
    
    uploaded_files = st.file_uploader(
        "Upload set of query clips (.mp3, .wav)", 
        type=["mp3", "wav"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Loaded {len(uploaded_files)} files. Press run to identify them.")
        
        if st.button("🚀 Run Batch Classification"):
            results = []
            progress_bar = st.progress(0)
            
            for i, up_file in enumerate(uploaded_files):
                # Write to temp file
                ext = os.path.splitext(up_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(up_file.read())
                    tmp_path = tmp_file.name
                    
                try:
                    # Run matcher
                    pred, scores, _, _ = match_query(tmp_path, db_path=DB_PATH, mode="paired", sr=11025)
                    
                    # Ensure prediction represents ONLY the filename without extension
                    if pred == "Unknown / No Match":
                        pred_name = "unknown"
                    else:
                        pred_name = pred # Already holds the base song name from the DB
                        
                    results.append({
                        "filename": up_file.name,
                        "prediction": pred_name
                    })
                except Exception as e:
                    results.append({
                        "filename": up_file.name,
                        "prediction": f"error: {str(e)}"
                    })
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Display results preview
            st.success("Batch run complete!")
            st.dataframe(df)
            
            # Export to EXACT format (results.csv: filename,prediction)
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )
            st.info("The downloaded CSV follows the exact header formatting required by the grader.")
