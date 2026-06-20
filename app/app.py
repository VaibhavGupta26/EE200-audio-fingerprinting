import os
import sys
import tempfile
import soundfile as sf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add project root directory to path to allow resolution of 'src'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spectrogram import load_audio, compute_spectrogram
from src.fingerprint import get_peaks, build_database
from src.matcher import match_query
from src.robustness import generate_test_query

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
    st.markdown("Identify songs from short clips. You can upload an audio file or select any song from the database and customize noise injection/pitch shifts to test the system.")
    
    input_type = st.radio("Choose Input Method:", ["Generate query from Database Song", "Upload custom audio file"])
    
    query_filepath = None
    uploaded_file = None
    is_temp_file = False
    
    if input_type == "Generate query from Database Song":
        if not indexed_songs:
            st.error("No songs indexed in the database. Please check your data/songs directory.")
        else:
            selected_song = st.selectbox("Select Database Song:", indexed_songs)
            
            # Param sliders
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                duration = st.slider("Query Duration (seconds):", 3.0, 15.0, 5.0, 0.5)
            with col_b:
                add_noise_flag = st.checkbox("Inject Noise (AWGN)")
                snr = st.slider("SNR Level (dB):", -10, 30, 10, disabled=not add_noise_flag)
            with col_c:
                add_pitch_flag = st.checkbox("Shift Pitch/Speed (Resampling)")
                shift = st.slider("Pitch/Speed Shift (%):", -5.0, 5.0, 0.0, 0.5, disabled=not add_pitch_flag)
                
            if st.button("⚡ Generate & Identify"):
                # Find corresponding original file (mp3 or wav)
                song_filename = None
                for ext in [".mp3", ".wav"]:
                    test_name = f"{selected_song}{ext}"
                    if os.path.exists(os.path.join(SONGS_DIR, test_name)):
                        song_filename = test_name
                        break
                        
                if song_filename is None:
                    st.error(f"Original audio file for '{selected_song}' not found in data/songs/. Ensure database files are pushed to GitHub.")
                else:
                    song_path = os.path.join(SONGS_DIR, song_filename)
                    with st.spinner("Extracting slice and injecting distortions..."):
                        query_audio = generate_test_query(
                            song_path, 
                            duration=duration, 
                            sr=11025, 
                            noise_db=snr if add_noise_flag else None, 
                            pitch_percent=shift if add_pitch_flag else 0.0
                        )
                        if query_audio is not None:
                            # Save query to a temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                sf.write(tmp_file.name, query_audio, 11025)
                                query_filepath = tmp_file.name
                                is_temp_file = True
                            st.write(f"Generated query from **{selected_song}**:")
                            st.audio(query_filepath)
                        else:
                            st.error("Failed to load original song or slice. Verify audio decoding libraries.")
                            
    else:
        uploaded_file = st.file_uploader("Upload audio clip (.mp3, .wav)", type=["mp3", "wav"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                query_filepath = tmp_file.name
                is_temp_file = True
            st.audio(uploaded_file)
            
    if query_filepath is not None:
        with st.spinner("Processing audio and querying database..."):
            try:
                # Run the matcher
                predicted_song, scores, best_offsets, match_details = match_query(
                    query_filepath, db_path=DB_PATH, mode="paired", sr=11025
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
                y, sr = load_audio(query_filepath, sr=11025)
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
                if is_temp_file and query_filepath is not None and os.path.exists(query_filepath):
                    os.remove(query_filepath)

# --- BATCH RUNNING MODE ---
elif mode == "Batch Mode":
    st.title("📂 Batch Song Identification")
    st.markdown("Run batch classification on multiple query clips. You can upload multiple audio files, or select a list of database songs and auto-generate test queries for them.")
    
    batch_input_type = st.radio("Choose Batch Method:", ["Test using Database Songs (Auto-generate queries)", "Upload multiple query files"])
    
    results = []
    
    if batch_input_type == "Test using Database Songs (Auto-generate queries)":
        if not indexed_songs:
            st.error("No songs indexed in the database. Please check your data/songs directory.")
        else:
            selected_batch_songs = st.multiselect("Select Songs to Test:", indexed_songs, default=indexed_songs[:5])
            
            # Param sliders
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                batch_dur = st.slider("Query Duration (seconds):", 3.0, 15.0, 5.0, 0.5, key="batch_dur")
            with col_y:
                batch_noise_flag = st.checkbox("Inject Noise (AWGN)", key="batch_noise")
                batch_snr = st.slider("SNR Level (dB):", -10, 30, 10, disabled=not batch_noise_flag, key="batch_snr")
            with col_z:
                batch_pitch_flag = st.checkbox("Shift Pitch/Speed (Resampling)", key="batch_pitch")
                batch_shift = st.slider("Pitch/Speed Shift (%):", -5.0, 5.0, 0.0, 0.5, disabled=not batch_pitch_flag, key="batch_shift")
                
            if st.button("🚀 Run Batch Test"):
                progress_bar = st.progress(0)
                
                for i, song_name in enumerate(selected_batch_songs):
                    song_filename = None
                    for ext in [".mp3", ".wav"]:
                        test_name = f"{song_name}{ext}"
                        if os.path.exists(os.path.join(SONGS_DIR, test_name)):
                            song_filename = test_name
                            break
                            
                    if song_filename is None:
                        results.append({
                            "filename": f"{song_name}_query.wav",
                            "prediction": "error: song file not found"
                        })
                        continue
                        
                    song_path = os.path.join(SONGS_DIR, song_filename)
                    query_audio = generate_test_query(
                        song_path, 
                        duration=batch_dur, 
                        sr=11025, 
                        noise_db=batch_snr if batch_noise_flag else None, 
                        pitch_percent=batch_shift if batch_pitch_flag else 0.0
                    )
                    
                    if query_audio is None:
                        results.append({
                            "filename": f"{song_name}_query.wav",
                            "prediction": "error: loading failed"
                        })
                        continue
                        
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        sf.write(tmp_file.name, query_audio, 11025)
                        tmp_path = tmp_file.name
                        
                    try:
                        pred, _, _, _ = match_query(tmp_path, db_path=DB_PATH, mode="paired", sr=11025)
                        
                        # Format prediction (matched song name without extension)
                        if pred == "Unknown / No Match":
                            pred_name = "unknown"
                        else:
                            pred_name = pred
                            
                        results.append({
                            "filename": f"{song_name}_query.wav",
                            "prediction": pred_name
                        })
                    except Exception as e:
                        results.append({
                            "filename": f"{song_name}_query.wav",
                            "prediction": f"error: {str(e)}"
                        })
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            
                    progress_bar.progress((i + 1) / len(selected_batch_songs))
                    
                df = pd.DataFrame(results)
                st.success("Batch run complete!")
                st.dataframe(df)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download results.csv",
                    data=csv_data,
                    file_name="results.csv",
                    mime="text/csv"
                )
                
    else:
        uploaded_files = st.file_uploader(
            "Upload set of query clips (.mp3, .wav)", 
            type=["mp3", "wav"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Loaded {len(uploaded_files)} files. Press run to identify them.")
            
            if st.button("🚀 Run Batch Classification"):
                progress_bar = st.progress(0)
                
                for i, up_file in enumerate(uploaded_files):
                    ext = os.path.splitext(up_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                        tmp_file.write(up_file.read())
                        tmp_path = tmp_file.name
                        
                    try:
                        pred, _, _, _ = match_query(tmp_path, db_path=DB_PATH, mode="paired", sr=11025)
                        
                        if pred == "Unknown / No Match":
                            pred_name = "unknown"
                        else:
                            pred_name = pred
                            
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
                            
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                df = pd.DataFrame(results)
                st.success("Batch run complete!")
                st.dataframe(df)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download results.csv",
                    data=csv_data,
                    file_name="results.csv",
                    mime="text/csv"
                )
                st.info("The downloaded CSV follows the exact header formatting required by the grader.")
