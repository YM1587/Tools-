"""
app.py — Free Transcriber Web UI
Run locally:  python app.py
Then open:    http://localhost:7860
Anyone on your network can use it at http://YOUR_IP:7860
Deploy free to HuggingFace Spaces: just upload this file + requirements.txt
"""

import os
import time
import tempfile
import subprocess

import gradio as gr
from faster_whisper import WhisperModel

SPEED_PRESETS = {
    "quality  (most accurate, slowest)": dict(beam_size=5, best_of=5),
    "fast     (recommended, ~2× quicker)": dict(beam_size=2, best_of=1),
    "turbo    (quickest, minor accuracy drop)": dict(beam_size=1, best_of=1),
}

_model_cache = {}

def get_model(size: str) -> WhisperModel:
    if size not in _model_cache:
        _model_cache[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _model_cache[size]


def extract_wav(input_path: str, tmp_dir: str) -> str:
    out_path = os.path.join(tmp_dir, "audio.wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", out_path],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
    return out_path


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def assign_speaker(seg_start, seg_end, diar_segs):
    overlap = {}
    for d_start, d_end, spk in diar_segs:
        o = min(seg_end, d_end) - max(seg_start, d_start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0) + o
    return max(overlap, key=overlap.get) if overlap else "SPEAKER"


def run_transcription(
    audio_file,
    model_size,
    speed_label,
    language,
    output_format,
    enable_diarization,
    hf_token,
    num_speakers,
    progress=gr.Progress()
):
    if audio_file is None:
        return "Please upload an audio or video file.", None

    preset = SPEED_PRESETS[speed_label]
    lang = language.strip() if language.strip() else None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            progress(0.05, desc="Loading Whisper model...")
            model = get_model(model_size)

            diar_segs = []
            if enable_diarization:
                if not hf_token or not hf_token.strip().startswith("hf_"):
                    return (
                        "Speaker identification needs a HuggingFace token.\n\n"
                        "Steps:\n"
                        "1. Create a free account at https://huggingface.co\n"
                        "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                        "3. Get a token at https://huggingface.co/settings/tokens\n"
                        "4. Paste it in the HF Token box and try again.",
                        None
                    )
                progress(0.15, desc="Extracting audio for speaker detection...")
                wav_path = extract_wav(audio_file, tmp)

                progress(0.25, desc="Detecting speakers (downloads model once)...")
                try:
                    import torch
                    from pyannote.audio import Pipeline
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token.strip()
                    )
                    pipeline.to(torch.device("cpu"))
                    kwargs = {}
                    if num_speakers and int(num_speakers) > 0:
                        kwargs["num_speakers"] = int(num_speakers)
                    diarization = pipeline(wav_path, **kwargs)
                    diar_segs = [
                        (t.start, t.end, spk)
                        for t, _, spk in diarization.itertracks(yield_label=True)
                    ]
                except Exception as e:
                    return f"Speaker detection error: {e}\n\nMake sure you accepted the model terms on HuggingFace.", None

            progress(0.40, desc="Transcribing speech...")
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

            progress(0.75, desc="Formatting output...")
            seg_list = list(segments)
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
            summary = (
                f"✅  Language: {info.language.upper()} "
                f"({info.language_probability:.0%} confidence) | "
                f"Audio: {info.duration/60:.1f} min"
            )
            return f"{summary}\n\n{transcript_text}", stable_path

    except Exception as e:
        return f"Error: {e}", None


# ── UI ────────────────────────────────────────────────────────────────────────

css = """
.gradio-container { max-width: 860px !important; margin: 0 auto; }
#title { text-align: center; padding: 24px 0 4px; }
#subtitle { text-align: center; color: #64748b; margin-bottom: 20px; font-size: 0.95rem; }
#run-btn { background: #1e293b !important; border: none !important; }
#run-btn:hover { background: #334155 !important; }
"""

theme = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks() as demo:

    gr.HTML('<h1 id="title">🎙️ Free Transcriber</h1>')
    gr.HTML('<p id="subtitle">Powered by OpenAI Whisper · runs 100% on your machine · no subscriptions</p>')

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.File(
                label="Audio or Video File",
                file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg",
                            ".mp4", ".mkv", ".mov", ".avi", ".webm"],
            )
        with gr.Column(scale=2):
            model_size = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large-v3"],
                value="small",
                label="Model size",
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
                ["Plain text", "Timestamped text", "SRT subtitles"],
                value="Plain text",
                label="Output format",
            )

    with gr.Accordion("👥  Speaker identification (who said what)", open=False):
        gr.Markdown(
            "Identifies which speaker said each line. Requires a **free** HuggingFace token.\n\n"
            "**Steps:** [1] Sign up at huggingface.co  "
            "[2] Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)  "
            "[3] Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
        )
        enable_diarization = gr.Checkbox(label="Enable speaker identification", value=False)
        with gr.Row():
            hf_token = gr.Textbox(
                label="HuggingFace token",
                placeholder="hf_...",
                type="password",
                max_lines=1,
            )
            num_speakers = gr.Number(
                label="Number of speakers (0 = auto-detect)",
                value=0, minimum=0, maximum=20, precision=0
            )

    run_btn = gr.Button("▶  Transcribe", variant="primary", elem_id="run-btn")

    output_text = gr.Textbox(
        label="Transcript",
        lines=18,
        interactive=False,
    )
    download_file = gr.File(label="Download transcript")

    run_btn.click(
        fn=run_transcription,
        inputs=[
            audio_input, model_size, speed_choice,
            language, output_format,
            enable_diarization, hf_token, num_speakers,
        ],
        outputs=[output_text, download_file],
    )

    gr.Markdown(
        "---\n"
        "**Tip:** First run downloads the Whisper model (~240 MB for `small`). "
        "Every run after that is instant and fully offline."
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        css=css,
    )