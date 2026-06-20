import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import project functions to verify/generate assets
from src.spectrogram import load_audio, compute_spectrogram, plot_dft_and_spectrograms
from src.fingerprint import get_peaks, plot_constellation_map
from src.matcher import match_query, plot_offset_histogram
from src.robustness import generate_test_query

def generate_report_pdf(output_pdf_path="report/Q3_report.pdf", db_path="database.pkl", songs_dir="data/songs"):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    images_dir = "report/images"
    os.makedirs(images_dir, exist_ok=True)
    
    print("Generating report assets...")
    
    # 1. Load a song to generate spectrogram, DFT, and constellation plots
    files = [f for f in os.listdir(songs_dir) if f.endswith(".mp3")]
    if not files:
        print("Error: No songs found in data/songs to generate report.")
        return False
        
    rep_song_filename = files[0]
    rep_song_path = os.path.join(songs_dir, rep_song_filename)
    rep_song_name = os.path.splitext(rep_song_filename)[0]
    
    # Generate DFT and spectrograms
    plot_dft_and_spectrograms(rep_song_path, images_dir)
    
    # Generate constellation map
    plot_constellation_map(rep_song_path, os.path.join(images_dir, "constellation.png"), duration_limit=12)
    
    # Generate query matching offset histogram for illustration
    query_audio = generate_test_query(rep_song_path, duration=6.0, sr=11025, noise_db=12)
    temp_q_path = "temp_report_query.wav"
    sf.write(temp_q_path, query_audio, 11025)
    pred_song, _, offsets, _ = match_query(temp_q_path, db_path=db_path, mode="paired", sr=11025)
    plot_offset_histogram(offsets, pred_song, os.path.join(images_dir, "offset_histogram_report.png"))
    if os.path.exists(temp_q_path):
        os.remove(temp_q_path)
        
    # Check if robustness plots exist, if not run them
    noise_plot = os.path.join(images_dir, "robustness_noise_sweep.png")
    pitch_plot = os.path.join(images_dir, "robustness_pitch_sweep.png")
    if not os.path.exists(noise_plot) or not os.path.exists(pitch_plot):
        print("Robustness sweep plots missing. Running sweeps now...")
        from src.robustness import run_experiments
        run_experiments(songs_dir, db_path, images_dir)
        
    print("Compiling PDF document using ReportLab...")
    
    # Page setup
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54, # Reduced top/bottom margin to leave space for page number only
        bottomMargin=54
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles - All text colors set to black
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#000000'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#000000'),
        alignment=1,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#000000'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#000000'),
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#000000'),
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#000000'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    story = []
    
    # ==========================================
    # --- PAGE 1: COVER / INTRO PAGE ---
    # ==========================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("EE200: Signals, Systems and Networks", title_style))
    story.append(Paragraph("Course Project Report — Question 3: Sonic Signatures & Signals to Softwares", subtitle_style))
    story.append(Spacer(1, 100))
    
    # Info block layout
    intro_data = [
        [Paragraph("<b>Name:</b>", body_style), Paragraph("Vaibhav Gupta", body_style)],
        [Paragraph("<b>Roll no:</b>", body_style), Paragraph("241124", body_style)],
        [Paragraph("<b>Course:</b>", body_style), Paragraph("EE200", body_style)],
        [Paragraph("<b>Instructor:</b>", body_style), Paragraph("Prof. Tushar Sandhan", body_style)]
    ]
    t_intro = Table(intro_data, colWidths=[100, 250])
    t_intro.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e9ecef')),
    ]))
    
    # Center-align the table on the cover page
    t_intro_container = Table([[t_intro]], colWidths=[350])
    t_intro_container.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_intro_container)
    
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 2: TABLE OF CONTENTS & LINKS ---
    # ==========================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(Spacer(1, 10))
    
    # Clean Table of Contents with page offsets
    toc_data = [
        [Paragraph("<b>Section</b>", body_style), Paragraph("<b>Page</b>", body_style)],
        [Paragraph("1. Why a Single Fourier Transform Will Not Do", body_style), Paragraph("3", body_style)],
        [Paragraph("2. Spectrograms and the Time-Frequency Tradeoff", body_style), Paragraph("4", body_style)],
        [Paragraph("3. Fingerprint Generation and Hashing", body_style), Paragraph("6", body_style)],
        [Paragraph("4. Single Peaks vs. Paired Hashes: Decisiveness of Hashing", body_style), Paragraph("7", body_style)],
        [Paragraph("5. Robustness Testing and Discussion", body_style), Paragraph("8", body_style)]
    ]
    t_toc = Table(toc_data, colWidths=[400, 50])
    t_toc.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 1, colors.HexColor('#000000')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
    ]))
    story.append(t_toc)
    story.append(Spacer(1, 60))
    
    # App Deployment Info Table (Customized background/border to #f8f9fa)
    story.append(Paragraph("<b>App Deployment Information</b>", h1_style))
    data_info = [
        [Paragraph("<b>Deployed Live App Link:</b>", body_style), Paragraph("<font color='blue'><u>https://ee200-audio-fingerprinting-241124.streamlit.app/</u></font>", body_style)],
        [Paragraph("<b>App Source Code Link:</b>", body_style), Paragraph("<font color='blue'><u>https://github.com/VaibhavGupta26/EE200-audio-fingerprinting</u></font>", body_style)]
    ]
    t_links = Table(data_info, colWidths=[150, 350])
    t_links.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#f8f9fa')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_links)
    
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 3: SECTION 1 ---
    # ==========================================
    story.append(Paragraph("1. Why a Single Fourier Transform Will Not Do", h1_style))
    story.append(Paragraph(
        "A single standard Fourier Transform (DFT) evaluates the frequency content of an entire signal accumulated "
        "across its full duration. In mathematical terms, the forward Fourier transform integrates the product of the signal "
        "and the complex sinusoid e<sup>-jωt</sup> from -∞ to +∞. Because these sinusoidal basis functions "
        "have infinite support (they stretch infinitely in time), the resulting spectrum represents the global frequencies present, "
        "but completely removes their temporal localized information. All sense of timing—when notes start, stop, or change—is lost. "
        "For song identification, knowing the order of notes is crucial. Without timing information, two songs with identical "
        "note distributions played in a different sequence (e.g., ascending vs. descending scales) would yield the exact same DFT magnitude spectrum, "
        "making identification impossible.",
        body_style
    ))
    
    img_dft_path = os.path.join(images_dir, "dft_entire_song.png")
    if os.path.exists(img_dft_path):
        story.append(Image(img_dft_path, width=480, height=192))
        story.append(Paragraph(f"Figure 1: DFT Magnitude of entire song ({rep_song_name}). Timing information is entirely absent.", caption_style))
        
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 4: SECTION 2 ---
    # ==========================================
    story.append(Paragraph("2. Spectrograms and the Time-Frequency Tradeoff", h1_style))
    story.append(Paragraph(
        "To capture how frequency changes over time, we use a spectrogram by implementing the Short-Time Fourier Transform (STFT). "
        "This involves sliding a window of duration N along the signal, multiplying the chunk by a windowing function (such as a Hanning window "
        "to reduce spectral leakage), computing the DFT of each short chunk, and stacking these magnitudes side by side as columns in a 2D image.",
        body_style
    ))
    story.append(Paragraph(
        "The spectrogram is governed by the Gabor Limit (or Uncertainty Principle) which states that we cannot simultaneously resolve frequency "
        "and time with infinite precision. Mathematically: Δt · Δf ≥ 1/(4π). The window length N determines this tradeoff:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Short Window (N=256, ~23ms):</b> Offers high time resolution (we know precisely when an event happens, Δt is small) but poor "
        "frequency resolution (Δf is large). Frequencies are smeared vertically, making it hard to identify exact musical notes.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Long Window (N=4096, ~372ms):</b> Offers high frequency resolution (precise notes, Δf is small) but poor time resolution "
        "(Δt is large). Sharp transients are smeared horizontally, making it difficult to pinpoint the exact time a note started.",
        bullet_style
    ))
    
    img_short = os.path.join(images_dir, "spectrogram_short.png")
    img_long = os.path.join(images_dir, "spectrogram_long.png")
    img_default = os.path.join(images_dir, "spectrogram_default.png")
    
    if os.path.exists(img_short):
        story.append(Image(img_short, width=420, height=168))
        story.append(Paragraph("Figure 2a: Spectrogram with a Short Window (N=256). Fine vertical lines show temporal changes, but frequency bands are blurred.", caption_style))
    if os.path.exists(img_long):
        story.append(Image(img_long, width=420, height=168))
        story.append(Paragraph("Figure 2b: Spectrogram with a Long Window (N=4096). Sharp horizontal lines show clear frequency bins, but events are temporally smeared.", caption_style))
        
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 5: DEFAULT SPECTROGRAM ---
    # ==========================================
    if os.path.exists(img_default):
        story.append(Image(img_default, width=420, height=168))
        story.append(Paragraph("Figure 2c: Default Spectrogram (N=1024), providing a balanced time-frequency compromise for audio fingerprinting.", caption_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Discussion of Resolution Trade-off:</b>", body_style))
    story.append(Paragraph(
        "Observing the figures, the Gabor limit represents a fundamental mathematical constraint in windowed Fourier analysis. "
        "When N is small (N=256), the short time-duration of the window allows us to pinpoint brief events (e.g. transients, snare hits) "
        "precisely on the time axis. However, the short window also truncates the sinusoidal waves, causing severe spectral leakage and broadening the frequency peaks (smeared bands). "
        "Conversely, when N is large (N=4096), the long window allows us to capture multiple cycles of low frequencies, giving extremely sharp spectral resolution (fine horizontal lines). "
        "But because the window covers almost 372 ms of audio, any frequency transition or note change that occurs within that window is blended together, "
        "rendering the temporal boundaries fuzzy. The default size (N=1024) balances this tradeoff for optimal fingerprinting.",
        body_style
    ))
        
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 6: SECTION 3 ---
    # ==========================================
    story.append(Paragraph("3. Fingerprint Generation and Hashing", h1_style))
    story.append(Paragraph(
        "The audio fingerprint is constructed by identifying local maxima (peaks) in the log-spectrogram to form a 'constellation map'. "
        "We identify peaks that stand out from their immediate neighbors using a 2D maximum filter (neighborhood of 15x15 bins) and a threshold limit. "
        "This filters out background noise and preserves the strongest spectral components.",
        body_style
    ))
    
    img_const = os.path.join(images_dir, "constellation.png")
    if os.path.exists(img_const):
        story.append(Image(img_const, width=420, height=210))
        story.append(Paragraph("Figure 3: Constellation Map of detected peaks overlaid on the log-spectrogram (first 12 seconds).", caption_style))
        
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 7: SECTION 4 ---
    # ==========================================
    story.append(Paragraph("4. Single Peaks vs. Paired Hashes: Decisiveness of Hashing", h1_style))
    story.append(Paragraph(
        "Shazam's core innovation lies in pairing peaks rather than using single peaks. "
        "A single peak is simply a frequency bin f<sub>1</sub> at a specific frame t<sub>1</sub>. When we index single peaks, the database maps f<sub>1</sub> &rarr; (song, t<sub>1</sub>). "
        "Because music contains repeated note structures and harmonic redundancy, single frequency bins are highly degenerate. A common note "
        "(e.g., A4 at 440 Hz) will match thousands of frames across all songs in the database. "
        "When querying, this creates a massive volume of random false-positive alignments, leading to high computational load and a noisy offset histogram "
        "where the correct match is buried.",
        body_style
    ))
    story.append(Paragraph(
        "By contrast, joining two peaks (f<sub>1</sub>, t<sub>1</sub>) and (f<sub>2</sub>, t<sub>2</sub>) into a paired hash key (f<sub>1</sub>, f<sub>2</sub>, Δt) (where Δt = t<sub>2</sub> - t<sub>1</sub>) "
        "exponentially increases the entropy of each fingerprint. The key state space increases from N bins to N<sup>2</sup> &times; Δt<sub>range</sub>. "
        "The probability that an incorrect song shares the exact frequency pair with the exact same time gap by chance is extremely low. "
        "Thus, false songs generate flat, uniform background distributions in the offset histogram. The correct song, having the same relative note spacing, "
        "yields matches that line up at a single, consistent time offset, producing a massive, unmistakable peak in the offset histogram (see Figure 4).",
        body_style
    ))
    
    img_hist = os.path.join(images_dir, "offset_histogram_report.png")
    if os.path.exists(img_hist):
        story.append(Image(img_hist, width=450, height=180))
        story.append(Paragraph("Figure 4: Offset Histogram for the correct match. All valid hashes align at a single relative frame offset.", caption_style))
        
    story.append(PageBreak())
    
    # ==========================================
    # --- PAGE 8: SECTION 5 ---
    # ==========================================
    story.append(Paragraph("5. Robustness Testing and Discussion", h1_style))
    story.append(Paragraph(
        "We tested the fingerprinting system against two major signal distortions: additive white Gaussian noise (AWGN) and pitch/speed shifts.",
        body_style
    ))
    
    img_noise_sweep = os.path.join(images_dir, "robustness_noise_sweep.png")
    img_pitch_sweep = os.path.join(images_dir, "robustness_pitch_sweep.png")
    
    if os.path.exists(img_noise_sweep):
        story.append(Image(img_noise_sweep, width=380, height=237))
        story.append(Paragraph("Figure 5: Identification accuracy of Paired Hashes vs. Single Peaks under varying levels of AWGN.", caption_style))
    if os.path.exists(img_pitch_sweep):
        story.append(Image(img_pitch_sweep, width=380, height=237))
        story.append(Paragraph("Figure 6: Identification accuracy of Paired Hashes vs. Single Peaks under varying pitch/speed shifts.", caption_style))
        
    story.append(PageBreak())
    
    story.append(Paragraph("Observations & Rationale:", h1_style))
    story.append(Paragraph(
        "<b>a. Noise Robustness:</b> The paired hashing scheme exhibits high robustness to noise. Even at an SNR of 5 dB, the system retains "
        "excellent recognition accuracy. This is because additive noise mainly alters low-amplitude regions of the spectrogram, leaving "
        "the strongest local peaks (which form the constellation map) intact. Once the SNR drops below 0 dB (noise energy exceeds signal energy), "
        "spurious noise peaks begin to dominate, and the identification rate drops. Single-peak matching fails much earlier because the noise "
        "generates false frequency peak alignments that drown out the correct match's signal in the offset histogram.",
        body_style
    ))
    story.append(Paragraph(
        "<b>b. Pitch Shift Vulnerability:</b> As shown in Figure 6, even a tiny pitch/speed shift of &plusmn;1.5% causes the identification rate of both "
        "single-peaks and paired-hashes to collapse to 0%. This occurs despite the song sounding identical to human ears. "
        "The reason is mathematical: a pitch shift scales all spectral frequencies (e.g. f<sub>shifted</sub> = (1 + &alpha;)f). "
        "In a linear frequency spectrogram, this shifts the local peak positions vertically. Since the database matching performs an exact key lookup "
        "on the integer frequency bins f<sub>1</sub> and f<sub>2</sub>, a shift of just 1% moves the frequencies outside the exact bins. Since there are no fuzzy "
        "matches, the hash keys do not match the database, and the system fails.",
        body_style
    ))
    
    story.append(Paragraph("Suggested Robustness Improvement:", h1_style))
    story.append(Paragraph(
        "To make the identifier robust to pitch shifts, we can convert the frequency axis of the spectrogram from a linear scale (Hz) "
        "to a logarithmic scale (such as MIDI notes or Constant-Q Transform (CQT) bins). "
        "On a log scale, a pitch shift (multiplication in Hz) becomes a constant addition: log(f<sub>shifted</sub>) = log(f) + log(1 + &alpha;). "
        "Thus, if we define our hash keys using the frequency difference on the log scale, &Delta;f<sub>log</sub> = log(f<sub>2</sub>) - log(f<sub>1</sub>) = log(f<sub>2</sub>/f<sub>1</sub>), "
        "the hash key becomes entirely invariant to global pitch shifts! The pitch shift offset shifts all peaks by the same vertical distance, but "
        "leaves their frequency ratios (differences on the log scale) unchanged, allowing correct matching.",
        body_style
    ))
    
    # Custom simple page number footer callback (only print page number on pages > 1)
    def add_page_number(canvas, doc):
        if doc.page > 1:
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            canvas.setFillColor(colors.HexColor('#000000'))
            canvas.drawRightString(doc.pagesize[0]-54, 38, f"Page {doc.page}")
            canvas.restoreState()
        
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Report PDF built successfully and saved to {output_pdf_path}!")
    return True

if __name__ == "__main__":
    generate_report_pdf()
