"""

Transcribe audio/video WITH speaker labels (SPEAKER_00, SPEAKER_01, etc.)
by combining faster-whisper (speech-to-text) with pyannote.audio (who is speaking when).

One-time setup required:
  1. Create a free account at https://huggingface.co
  2. Accept the pyannote model terms at:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  3. Create a token at https://huggingface.co/settings/tokens (read access is enough)
  4. Pass it with --hf-token YOUR_TOKEN  (or set env var HF_TOKEN)

Examples:
    python transcribe_diarize.py interview.m4a --hf-token hf_xxxx
    python transcribe_diarize.py meeting.mp4 --hf-token hf_xxxx --mode fast
    python transcribe_diarize.py podcast.mp3 --hf-token hf_xxxx --speakers 2
"""

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np


def check_deps():
    missing = []
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    try:
        import pyannote.audio
    except ImportError:
        missing.append("pyannote.audio")
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}\nRun: pip install {' '.join(missing)}")


SPEED_PRESETS = {
    "quality": dict(beam_size=5, best_of=5,  temperature=0.0),
    "fast":    dict(beam_size=2, best_of=1,  temperature=0.0),
    "turbo":   dict(beam_size=1, best_of=1,  temperature=0.0),
}


def extract_audio_wav(input_path: Path, tmp_dir: str) -> Path:
    """Use ffmpeg to extract 16kHz mono WAV — what pyannote expects."""
    import subprocess
    out_path = Path(tmp_dir) / "audio_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16",
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr.decode()}")
    return out_path


def diarize(wav_path: Path, hf_token: str, num_speakers=None):
    """Run pyannote speaker diarization, return list of (start, end, speaker)."""
    from pyannote.audio import Pipeline
    import torch

    print("   Loading speaker diarization model (downloads once)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    pipeline.to(torch.device("cpu"))

    print("   Running diarization...")
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    return segments


def assign_speaker(seg_start, seg_end, diarization_segments):
    """Find the speaker with the most overlap in this time window."""
    overlap = {}
    for d_start, d_end, speaker in diarization_segments:
        o_start = max(seg_start, d_start)
        o_end   = min(seg_end,   d_end)
        if o_end > o_start:
            overlap[speaker] = overlap.get(speaker, 0) + (o_end - o_start)
    if not overlap:
        return "UNKNOWN"
    return max(overlap, key=overlap.get)


def transcribe_with_speakers(input_path: Path, args, hf_token: str) -> str:
    from faster_whisper import WhisperModel

    preset = SPEED_PRESETS[args.mode]
    print(f"\n-> Transcribing: {input_path.name}  [mode={args.mode}, beam={preset['beam_size']}]")
    start = time.time()

    # Step 1: Extract WAV for diarization
    with tempfile.TemporaryDirectory() as tmp:
        print("   Extracting audio for diarization...")
        wav_path = extract_audio_wav(input_path, tmp)

        # Step 2: Speaker diarization
        diar_segments = diarize(wav_path, hf_token, args.speakers)
        print(f"   Found {len(set(s[2] for s in diar_segments))} speakers")

        # Step 3: Transcription
        print("   Transcribing speech...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(wav_path),
            language=args.language,
            beam_size=preset["beam_size"],
            best_of=preset["best_of"],
            temperature=preset["temperature"],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        print(f"   Language: {info.language} ({info.language_probability:.0%})")

        # Step 4: Align speakers with transcript segments
        lines = []
        current_speaker = None
        current_lines = []

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            speaker = assign_speaker(seg.start, seg.end, diar_segments)

            if speaker != current_speaker:
                if current_lines:
                    lines.append(f"\n{current_speaker}:")
                    lines.append(" ".join(current_lines))
                current_speaker = speaker
                current_lines = [text]
            else:
                current_lines.append(text)

        if current_lines and current_speaker:
            lines.append(f"\n{current_speaker}:")
            lines.append(" ".join(current_lines))

        elapsed = time.time() - start
        print(f"   Done in {elapsed:.1f}s  (speed={info.duration/elapsed:.1f}x real-time)")
        return "\n".join(lines)


def main():
    check_deps()

    parser = argparse.ArgumentParser(description="Transcribe with speaker labels.")
    parser.add_argument("input", help="Audio or video file")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env variable)")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--mode", default="fast", choices=["quality", "fast", "turbo"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers if you know it (helps accuracy)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if not args.hf_token:
        sys.exit(
            "A HuggingFace token is required for speaker diarization.\n"
            "1. Create a free account at https://huggingface.co\n"
            "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Get a token at https://huggingface.co/settings/tokens\n"
            "4. Run: python transcribe_diarize.py audio.m4a --hf-token hf_YOUR_TOKEN"
        )

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    result = transcribe_with_speakers(input_path, args, args.hf_token)

    outdir = Path(args.outdir) if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"{input_path.stem}_speakers.txt"
    out_file.write_text(result, encoding="utf-8")
    print(f"   Saved -> {out_file}")
    print("\nAll done.")


if __name__ == "__main__":
    main()