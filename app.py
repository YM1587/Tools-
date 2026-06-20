"""
app.py — Free Transcriber Web UI
Run:  python app.py  then open  http://localhost:7860
Then open:    http://localhost:7860
Anyone on your network can use it at http://YOUR_IP:7860
Speaker diarization uses librosa MFCCs + sklearn clustering.
Zero C++ compiler dependencies — works on Windows out of the box.
"""

import os
import time
import tempfile
import subprocess

import numpy as np
import gradio as gr
from faster_whisper import WhisperModel

SPEED_PRESETS = {
    "quality  (most accurate, slowest)": dict(beam_size=5, best_of=5),
    "fast     (recommended, ~2x quicker)": dict(beam_size=2, best_of=1),
    "turbo    (quickest, minor accuracy drop)": dict(beam_size=1, best_of=1),
}

_model_cache = {}

def get_model(size):
    if size not in _model_cache:
        _model_cache[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _model_cache[size]


def extract_wav(input_path, out_path):
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", out_path],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")


def format_timestamp(seconds):
    millis = int(round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def diarize_mfcc(wav_path, whisper_segments, num_speakers=None):
    """
    Speaker diarization using MFCC embeddings + agglomerative clustering.

    For each Whisper segment we extract MFCC mean+std as a speaker fingerprint,
    then cluster those fingerprints. No webrtcvad, no torchaudio, no C extensions.

    Returns list of (start_sec, end_sec, speaker_label) aligned to whisper segments.
    """
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    SR = 16000
    N_MFCC = 20
    MIN_DURATION = 0.3   # seconds — skip segments shorter than this

    y, _ = librosa.load(wav_path, sr=SR, mono=True)

    embeddings = []
    valid_segs  = []

    for seg in whisper_segments:
        duration = seg.end - seg.start
        if duration < MIN_DURATION:
            continue
        start_i = int(seg.start * SR)
        end_i   = int(seg.end   * SR)
        chunk   = y[start_i:end_i]
        if len(chunk) < int(MIN_DURATION * SR):
            continue
        mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=N_MFCC)
        emb  = np.concatenate([np.mean(mfcc, axis=1),
                                np.std(mfcc,  axis=1)])
        embeddings.append(emb)
        valid_segs.append(seg)

    if not valid_segs:
        return []

    if len(valid_segs) == 1:
        return [(valid_segs[0].start, valid_segs[0].end, "SPEAKER_00")]

    X = StandardScaler().fit_transform(embeddings)

    n = int(num_speakers) if num_speakers and int(num_speakers) > 1 else None
    clustering = AgglomerativeClustering(
        n_clusters=n,
        distance_threshold=None if n else 12.0,
        metric="euclidean",
        linkage="ward",
    )
    labels = clustering.fit_predict(X)

    return [
        (seg.start, seg.end, f"SPEAKER_{label:02d}")
        for seg, label in zip(valid_segs, labels)
    ]


def assign_speaker(seg_start, seg_end, diar_segs):
    """Find the speaker with most overlap in this time window."""
    overlap = {}
    for d_start, d_end, spk in diar_segs:
        o = min(seg_end, d_end) - max(seg_start, d_start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0) + o
    return max(overlap, key=overlap.get) if overlap else "SPEAKER_00"


def run_transcription(
    audio_file, model_size, speed_label, language,
    output_format, enable_diarization, num_speakers,
    progress=gr.Progress()
):
    if audio_file is None:
        return "Please upload an audio or video file.", None

    preset = SPEED_PRESETS[speed_label]
    lang   = language.strip() if language.strip() else None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "audio.wav")

            progress(0.05, desc="Loading Whisper model...")
            model = get_model(model_size)

            # Always extract WAV if diarization is on
            if enable_diarization:
                progress(0.10, desc="Extracting audio...")
                extract_wav(audio_file, wav_path)

            progress(0.20, desc="Transcribing speech...")
            segments, info = model.transcribe(
                audio_file,
                language=lang,
                beam_size=preset["beam_size"],
                best_of=preset["best_of"],
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                word_timestamps=False,
                condition_on_previous_text=False,
            )

            seg_list = list(segments)   # materialise the generator
            progress(0.65, desc="Transcription done, formatting...")

            diar_segs = []
            if enable_diarization and seg_list:
                progress(0.70, desc="Detecting speakers...")
                try:
                    n = int(num_speakers) if num_speakers and int(num_speakers) > 0 else None
                    diar_segs = diarize_mfcc(wav_path, seg_list, n)
                    n_spk = len(set(s[2] for s in diar_segs))
                    progress(0.85, desc=f"Found {n_spk} speaker(s), building output...")
                except Exception as e:
                    import traceback
                    return f"Speaker detection error: {e}\n\n{traceback.format_exc()}", None

            lines = []

            if enable_diarization and diar_segs:
                current_speaker = None
                buffer = []
                for seg in seg_list:
                    text = seg.text.strip()
                    if not text:
                        continue
                    speaker = assign_speaker(seg.start, seg.end, diar_segs)
                    if speaker != current_speaker:
                        if buffer:
                            lines.append(f"\n{current_speaker}:\n{' '.join(buffer)}")
                        current_speaker = speaker
                        buffer = [text]
                    else:
                        buffer.append(text)
                if buffer:
                    lines.append(f"\n{current_speaker}:\n{' '.join(buffer)}")
                transcript_text = "\n".join(lines)

            elif output_format == "SRT subtitles":
                for i, seg in enumerate(seg_list, 1):
                    t = seg.text.strip()
                    if t:
                        lines += [str(i),
                                  f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}",
                                  t, ""]
                transcript_text = "\n".join(lines)

            elif output_format == "Timestamped text":
                for seg in seg_list:
                    t = seg.text.strip()
                    if t:
                        lines.append(f"[{seg.start:.1f}s] {t}")
                transcript_text = "\n".join(lines)

            else:
                transcript_text = " ".join(
                    seg.text.strip() for seg in seg_list if seg.text.strip()
                )

            ext = "srt" if output_format == "SRT subtitles" else "txt"
            stable_path = os.path.join(
                tempfile.gettempdir(), f"transcript_{int(time.time())}.{ext}"
            )
            with open(stable_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            progress(1.0, desc="Done!")
            n_spk_note = f" | {len(set(s[2] for s in diar_segs))} speakers" if diar_segs else ""
            summary = (
                f"Language: {info.language.upper()} "
                f"({info.language_probability:.0%}) | "
                f"Audio: {info.duration/60:.1f} min{n_spk_note}"
            )
            return f"{summary}\n\n{transcript_text}", stable_path

    except Exception as e:
        import traceback
        return f"Error: {e}\n\n{traceback.format_exc()}", None


# ── UI ────────────────────────────────────────────────────────────────────────

css = """
.gradio-container { max-width: 860px !important; margin: 0 auto; }
#title  { text-align: center; padding: 24px 0 4px; }
#sub    { text-align: center; color: #64748b; margin-bottom: 20px; font-size:.95rem; }
#run-btn { background: #1e293b !important; border: none !important; }
#run-btn:hover { background: #334155 !important; }
"""

theme = gr.themes.Base(
    primary_hue="slate", neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks() as demo:
    gr.HTML('<h1 id="title">Free Transcriber</h1>')
    gr.HTML('<p id="sub">Powered by OpenAI Whisper · runs 100% on your machine · no subscriptions</p>')

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.File(
                label="Audio or Video File",
                file_types=[".mp3",".wav",".m4a",".flac",".ogg",
                            ".mp4",".mkv",".mov",".avi",".webm"],
            )
        with gr.Column(scale=2):
            model_size = gr.Dropdown(
                ["tiny","base","small","medium","large-v3"],
                value="small", label="Model size",
                info="larger = more accurate, slower"
            )
            speed_choice = gr.Dropdown(
                list(SPEED_PRESETS.keys()),
                value=list(SPEED_PRESETS.keys())[1],
                label="Speed mode",
            )
            language = gr.Textbox(
                label="Language (optional)",
                placeholder="e.g. en, sw, fr — blank = auto-detect",
                max_lines=1,
            )
            output_format = gr.Radio(
                ["Plain text","Timestamped text","SRT subtitles"],
                value="Plain text", label="Output format",
            )

    with gr.Accordion("Speaker identification — who said what", open=False):
        gr.Markdown(
            "Labels each speaker as SPEAKER_00, SPEAKER_01, etc.\n\n"
            "**No token or internet needed** — uses MFCC voice fingerprinting + "
            "clustering, runs fully offline."
        )
        enable_diarization = gr.Checkbox(
            label="Enable speaker identification", value=False
        )
        num_speakers = gr.Number(
            label="Number of speakers (0 = auto-detect)",
            value=0, minimum=0, maximum=20, precision=0
        )

    run_btn = gr.Button("Transcribe", variant="primary", elem_id="run-btn")
    output_text  = gr.Textbox(label="Transcript", lines=18, interactive=False)
    download_file = gr.File(label="Download transcript")

    run_btn.click(
        fn=run_transcription,
        inputs=[audio_input, model_size, speed_choice, language,
                output_format, enable_diarization, num_speakers],
        outputs=[output_text, download_file],
    )

    gr.Markdown("---\nFirst run downloads the Whisper model (~240 MB for `small`). Every run after is fully offline.")

if __name__ == "__main__":
    demo.launch(
        share=False, server_name="0.0.0.0",
        server_port=7860, theme=theme, css=css,
    )