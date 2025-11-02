import sys
import os
import json
import subprocess
import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def hhmmss(t: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    t = int(round(t))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02}:{m:02}:{s:02}"


def soft_trim(text: str, max_len: int = 120) -> str:
    """Trim text to max length, preferring to break at word boundaries."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rstrip()
    # avoid cutting in the middle of a word if possible
    last_space = cut.rfind(" ")
    if last_space > max_len * 0.6:
        cut = cut[:last_space]
    return cut + "…"


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def count_audio_streams(input_path: str) -> int:
    """Count number of audio streams in the input file using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-i",
            input_path,
            "-show_streams",
            "-select_streams",
            "a",
            "-loglevel",
            "error",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.count("[STREAM]")


def extract_mono_wav(input_path: str, out_dir: str) -> str:
    """Extract first audio stream as mono 16 kHz WAV (no probing)."""
    wav_path = os.path.join(out_dir, "audio.wav")
    print("🎙️  Extracting mono 16 kHz WAV...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", wav_path],
        check=True,
    )
    print(f"✅ Audio extracted: {wav_path}")
    return wav_path


def extract_track_wav(input_path: str, track_index: int, out_wav: str) -> None:
    """Extract specific audio track as mono 16 kHz WAV."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-map",
            f"0:a:{track_index}",
            "-ac",
            "1",
            "-ar",
            "16000",
            out_wav,
        ],
        check=True,
    )


def build_whisperx_cmd(
    wav_path: str,
    out_dir: str,
    model: str,
    language: str,
    batch_size: int = 16,
):
    """Build WhisperX command without diarization."""
    return [
        sys.executable,
        "-m",
        "whisperx",
        wav_path,
        "--language",
        language,
        "--model",
        model,
        "--device",
        "cpu",
        "--compute_type",
        "int8",
        "--output_format",
        "json",
        "--highlight_words",
        "True",
        "--output_dir",
        out_dir,
        "--batch_size",
        str(batch_size),
    ]


def transcribe_without_diarization(
    wav_path: str,
    out_dir: str,
    model: str,
    language: str,
    batch_size: int = 16,
):
    """Run WhisperX transcription without diarization."""
    cmd = build_whisperx_cmd(wav_path, out_dir, model, language, batch_size)
    subprocess.run(cmd, check=True)


def transcribe_with_diarization(
    wav_path: str,
    out_dir: str,
    model: str,
    language: str,
    min_speakers: int,
    max_speakers: int,
    hf_token: str,
    batch_size: int = 16,
):
    """Run WhisperX diarization with live logs."""
    print(f"\n📝 Transcribing + diarizing with WhisperX (model: {model})…\n")
    cmd = [
        sys.executable,
        "-m",
        "whisperx",
        wav_path,
        "--language",
        language,
        "--model",
        model,
        "--device",
        "cpu",
        "--compute_type",
        "int8",
        "--diarize",
        "--hf_token",
        hf_token,
        "--min_speakers",
        str(min_speakers),
        "--max_speakers",
        str(max_speakers),
        "--output_format",
        "json",
        "--highlight_words",
        "True",
        "--output_dir",
        out_dir,
        "--batch_size",
        str(batch_size),
    ]
    subprocess.run(cmd, check=True)
    print("\n✅ WhisperX finished.\n")


def load_segments(json_path: str):
    """Load and parse WhisperX JSON output."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", data)
    out = []
    for s in segs:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": text,
                "speaker": s.get("speaker"),  # SPEAKER_00 etc.
            }
        )
    return out


def summarize_speakers(segments):
    """Extract unique speakers and their total speaking durations."""
    speakers = set()
    dur = defaultdict(float)
    for s in segments:
        spk = s.get("speaker")
        if not spk:
            continue
        speakers.add(spk)
        dur[spk] += max(0.0, s["end"] - s["start"])
    return speakers, dur


def collect_samples_for_speaker(segments, speaker_id, n=5):
    """Return first2 + mid2 + last1 sample segments (chronological) for this speaker."""
    spk_segs = [s for s in segments if s.get("speaker") == speaker_id]
    spk_segs.sort(key=lambda x: x["start"])
    if not spk_segs:
        return []

    if len(spk_segs) <= n:
        return spk_segs

    # indices: first 2, middle 2, last 1
    first2 = [0, 1]
    last1 = [len(spk_segs) - 1]
    remain_needed = n - len(first2) - len(last1)  # typically 2
    # pick middles evenly
    mids = []
    if remain_needed > 0:
        # spread remaining indices across the middle range [2 .. len-2)
        start_mid = 2
        end_mid = max(2, len(spk_segs) - 2)
        if end_mid <= start_mid:
            # not enough middle room, just take subsequent ones
            candidates = list(
                range(start_mid, min(start_mid + remain_needed, len(spk_segs) - 1))
            )
        else:
            step = (end_mid - start_mid) / (remain_needed + 1)
            candidates = [int(round(start_mid + step * (i + 1))) for i in range(remain_needed)]
        mids = sorted(set(max(0, min(len(spk_segs) - 1, idx)) for idx in candidates))

    pick_ids = sorted(set(first2 + mids + last1))
    # ensure we have exactly n (dedupe could reduce)
    while len(pick_ids) < n:
        # add next available from start
        for j in range(len(spk_segs)):
            if j not in pick_ids:
                pick_ids.append(j)
                if len(pick_ids) == n:
                    break
    pick_ids = pick_ids[:n]
    return [spk_segs[i] for i in pick_ids]


def tty_input(prompt: str) -> str:
    """Robust interactive input via /dev/tty (macOS/Linux) with fallback."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        with open("/dev/tty", "r") as tty:
            line = tty.readline()
            if not line:  # EOF
                return ""
            return line.strip()
    except Exception:
        try:
            return input("").strip()
        except EOFError:
            return ""


def interactive_map_speakers(segments, speakers, durations, sample_lines: int = 5):
    """Interactive speaker identification with sample previews."""
    speakers = sorted(list(speakers))
    print("\n👥 Detected speakers:", ", ".join(speakers))
    for spk in speakers:
        d = durations.get(spk, 0.0)
        print(f"  - {spk}: {hhmmss(d)}")

    # Show sample lines for each speaker
    print("\n📑 Samples per speaker (first 2 + middle 2 + last 1):\n")
    for spk in speakers:
        samples = collect_samples_for_speaker(segments, spk, n=sample_lines)
        print(f"{spk} ({hhmmss(durations.get(spk, 0.0))} total)\nExamples:")
        if not samples:
            print("  (no text samples)")
        for s in samples:
            stamp = hhmmss(s["start"])
            line = soft_trim(s["text"], 120)
            print(f"  [{stamp}] {line}")
        print()

    # Pick which ID is You
    you_id = ""
    while True:
        you_id = tty_input(f"Which ID is YOU? Choose from {speakers} (blank to skip): ")
        if you_id == "" or you_id in speakers:
            break
        print("Invalid choice.")

    you_name = "You"
    if you_id:
        tmp = tty_input("Custom display name for YOU (Enter to keep 'You'): ")
        if tmp:
            you_name = tmp

    # Name the rest
    mapping = {}
    if you_id:
        mapping[you_id] = you_name

    guest_index = 1
    for spk in speakers:
        if spk == you_id:
            continue
        default_name = "Guest" if guest_index == 1 else f"Guest {guest_index}"
        custom = tty_input(f"Display name for {spk} (Enter to keep '{default_name}'): ")
        mapping[spk] = custom or default_name
        guest_index += 1

    print("\n✅ Mapping:")
    for spk, name in mapping.items():
        print(f"  {spk} → {name}")

    return mapping


def non_interactive_map_speakers(speakers, speaker_names: list):
    """
    Non-interactive speaker mapping using provided names.
    Maps speakers in order to provided names.
    """
    speakers = sorted(list(speakers))

    if len(speaker_names) != len(speakers):
        print(
            f"❌ Error: Provided {len(speaker_names)} speaker names "
            f"but detected {len(speakers)} speakers."
        )
        print(f"   Detected: {', '.join(speakers)}")
        print(f"   Provided: {', '.join(speaker_names)}")
        sys.exit(1)

    mapping = dict(zip(speakers, speaker_names))

    print("\n✅ Speaker mapping (non-interactive):")
    for spk, name in mapping.items():
        print(f"  {spk} → {name}")

    return mapping


def apply_mapping(segments, mapping):
    """Apply speaker name mapping to segments."""
    out = []
    for s in segments:
        spk = s.get("speaker") or "Unknown"
        out.append({**s, "speaker_name": mapping.get(spk, spk)})
    return out


def group_into_blocks(segments, block_seconds=30):
    """Group segments into time-based blocks."""
    if not segments:
        return [], {}
    end_time = max(s["end"] for s in segments)
    blocks = []
    t = 0.0
    while t < end_time + 1e-6:
        blocks.append((t, t + block_seconds))
        t += block_seconds
    buckets = defaultdict(list)
    for seg in segments:
        idx = int(seg["start"] // block_seconds)
        buckets[idx].append(seg)
    return blocks, buckets


def write_blocked_transcript(segments, out_dir: str, fname: str, block_seconds=30):
    """Write formatted transcript grouped into time blocks."""
    path = os.path.join(out_dir, fname)
    merged = sorted(segments, key=lambda x: x["start"])
    blocks, buckets = group_into_blocks(merged, block_seconds)

    with open(path, "w", encoding="utf-8") as f:
        for i, (b_start, b_end) in enumerate(blocks):
            idx = int(b_start // block_seconds)
            items = buckets.get(idx, [])
            if not items:
                continue
            f.write(f"{hhmmss(b_start)}–{hhmmss(b_end)}\n")
            for s in items:
                name = s.get("speaker_name") or s.get("speaker") or "Unknown"
                f.write(f"{name}: {s['text']}\n")
            f.write("\n")

    print(f"✅ Transcript saved to: {path}")


def find_first_json(directory: str) -> str:
    """Find first .json file in directory."""
    for f in os.listdir(directory):
        if f.lower().endswith(".json"):
            return os.path.join(directory, f)
    return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize interviews with interactive speaker identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (interactive mode)
  transcribe interview.mov

  # Specify model and speaker count
  transcribe podcast.mp4 --model large-v2 --max-speakers 4

  # Non-interactive mode with predefined speakers
  transcribe meeting.mp3 --speakers "Alice,Bob,Charlie"

  # Custom output directory
  transcribe talk.wav --output ./transcripts

  # Specify language (skip auto-detection)
  transcribe video.mov --language es

Environment:
  HF_TOKEN    HuggingFace token for pyannote.audio (required)
              Get yours at: https://huggingface.co/settings/tokens
              Grant access to: pyannote/speaker-diarization-3.1

Requirements:
  - ffmpeg must be installed (brew install ffmpeg)
  - HF_TOKEN environment variable must be set
        """,
    )

    parser.add_argument("input", help="Input audio or video file")

    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model size (default: large-v3-turbo)",
    )

    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en, use 'auto' for detection)",
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=2,
        help="Minimum number of speakers (default: 2)",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=2,
        help="Maximum number of speakers (default: 2)",
    )

    parser.add_argument(
        "--block-seconds",
        type=int,
        default=30,
        help="Transcript block duration in seconds (default: 30)",
    )

    parser.add_argument(
        "--sample-lines",
        type=int,
        default=5,
        help="Number of sample lines per speaker in preview (default: 5)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output directory (default: out_{filename}_diar)",
    )

    parser.add_argument(
        "--token",
        help="HuggingFace token (alternative to HF_TOKEN env var)",
    )

    parser.add_argument(
        "--speakers",
        help='Speaker names in order, comma-separated (e.g., "Alice,Bob"). '
        "Enables non-interactive mode.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="WhisperX batch size (default: 16)",
    )

    parser.add_argument(
        "--transcript-name",
        default="transcript.txt",
        help="Output transcript filename (default: transcript.txt)",
    )

    parser.add_argument(
        "--dual-track",
        action="store_true",
        help="Use dual-track mode for OBS-style multi-track recordings (no diarization needed)",
    )

    parser.add_argument(
        "--you-track",
        type=int,
        default=1,
        help="Audio track index for 'You' (default: 1, which is Track 2 in ffmpeg 0-indexing)",
    )

    parser.add_argument(
        "--guest-track",
        type=int,
        default=2,
        help="Audio track index for 'Guest' (default: 2, which is Track 3 in ffmpeg 0-indexing)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Check ffmpeg availability
    if not check_ffmpeg():
        print("❌ Error: ffmpeg is not installed or not in PATH")
        print("   Install with: brew install ffmpeg")
        sys.exit(1)

    # Validate input file
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    # Check HF token (only required for diarization mode)
    hf_token = None
    if not args.dual_track:
        hf_token = args.token or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("❌ Error: HuggingFace token not found")
            print("\nSet up your token:")
            print("  1. Get token at: https://huggingface.co/settings/tokens")
            print("  2. Grant access to: pyannote/speaker-diarization-3.1")
            print("  3. Run: export HF_TOKEN=hf_xxx...")
            print("     or use: transcribe --token hf_xxx... <file>")
            print("\nAlternatively, use --dual-track mode if you have OBS-style multi-track recordings")
            sys.exit(1)

    # Parse speaker names if provided
    speaker_names = None
    if args.speakers:
        speaker_names = [name.strip() for name in args.speakers.split(",")]
        if not speaker_names:
            print("❌ Error: --speakers cannot be empty")
            sys.exit(1)

    # Determine output directory
    if args.output:
        out_dir = args.output
    else:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        out_dir = f"out_{stem}_diar"

    os.makedirs(out_dir, exist_ok=True)

    if args.dual_track:
        # Dual-track mode: extract and transcribe separate tracks
        print("\n🎙️  Dual-track mode enabled")

        # Check audio stream count
        num_streams = count_audio_streams(input_path)
        print(f"   Found {num_streams} audio stream(s)")

        you_idx = args.you_track
        guest_idx = args.guest_track

        if num_streams <= you_idx:
            print(f"❌ Error: Track {you_idx+1} (You) not found. File has only {num_streams} audio stream(s).")
            sys.exit(1)
        if num_streams <= guest_idx:
            print(f"❌ Error: Track {guest_idx+1} (Guest) not found. File has only {num_streams} audio stream(s).")
            sys.exit(1)

        # Extract tracks
        print(f"\n📥 Extracting Track {you_idx+1} (You)...")
        you_wav = os.path.join(out_dir, "you.wav")
        extract_track_wav(input_path, you_idx, you_wav)

        print(f"📥 Extracting Track {guest_idx+1} (Guest)...")
        guest_wav = os.path.join(out_dir, "guest.wav")
        extract_track_wav(input_path, guest_idx, guest_wav)

        # Transcribe both tracks in parallel
        print("\n🎤 Transcribing both tracks in parallel...")
        you_dir = os.path.join(out_dir, "you_transcribe")
        os.makedirs(you_dir, exist_ok=True)

        guest_dir = os.path.join(out_dir, "guest_transcribe")
        os.makedirs(guest_dir, exist_ok=True)

        # Build whisperx commands
        you_cmd = build_whisperx_cmd(you_wav, you_dir, args.model, args.language, args.batch_size)
        guest_cmd = build_whisperx_cmd(
            guest_wav, guest_dir, args.model, args.language, args.batch_size
        )

        # Start both processes (output will be interleaved)
        print("   Starting You track transcription...")
        print("   Starting Guest track transcription...")
        print("   (Logs from both tracks will appear below)\n")

        you_process = subprocess.Popen(
            you_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        guest_process = subprocess.Popen(
            guest_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        # Stream output from both processes in real-time
        import select

        outputs = {you_process.stdout: "[You] ", guest_process.stdout: "[Guest] "}
        readable = list(outputs.keys())

        while readable:
            ready, _, _ = select.select(readable, [], [], 0.1)
            for stream in ready:
                line = stream.readline()
                if line:
                    print(outputs[stream] + line, end="", flush=True)
                else:
                    readable.remove(stream)

        # Wait for both to complete
        you_process.wait()
        guest_process.wait()

        if you_process.returncode != 0:
            print("\n❌ Error: You track transcription failed")
            sys.exit(1)
        if guest_process.returncode != 0:
            print("\n❌ Error: Guest track transcription failed")
            sys.exit(1)

        print("\n   ✅ Both transcriptions completed")

        # Load both transcripts
        you_json = find_first_json(you_dir)
        guest_json = find_first_json(guest_dir)

        if not you_json or not guest_json:
            print("❌ Error: Could not find transcription JSON outputs")
            sys.exit(1)

        you_segments = load_segments(you_json)
        guest_segments = load_segments(guest_json)

        # Tag segments with speaker labels
        for seg in you_segments:
            seg["speaker"] = "You"
        for seg in guest_segments:
            seg["speaker"] = "Guest"

        # Merge by timestamp
        all_segments = sorted(you_segments + guest_segments, key=lambda s: s["start"])

        # Write transcript
        write_blocked_transcript(all_segments, out_dir, args.transcript_name, args.block_seconds)

    else:
        # Standard diarization mode
        # Extract audio
        wav_path = extract_mono_wav(input_path, out_dir)

        # Transcribe with diarization
        transcribe_with_diarization(
            wav_path,
            out_dir,
            model=args.model,
            language=args.language,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            hf_token=hf_token,
            batch_size=args.batch_size,
        )

        # Find WhisperX JSON output
        json_path = find_first_json(out_dir)
        if not json_path:
            print("❌ Error: Could not find WhisperX JSON output")
            sys.exit(1)

        # Load and process segments
        segments = load_segments(json_path)
        speakers, durations = summarize_speakers(segments)

        if not speakers:
            print("❌ Error: No speakers detected")
            print("   Check diarization settings and HF token permissions")
            sys.exit(1)

        # Map speakers (interactive or non-interactive)
        if speaker_names:
            mapping = non_interactive_map_speakers(speakers, speaker_names)
        else:
            mapping = interactive_map_speakers(segments, speakers, durations, args.sample_lines)

        # Apply mapping and write transcript
        seg_mapped = apply_mapping(segments, mapping)
        write_blocked_transcript(seg_mapped, out_dir, args.transcript_name, args.block_seconds)

    print("\n🎉 All done!")
    print(f"📂 Output folder: {out_dir}")
    print(f"📄 Transcript: {os.path.join(out_dir, args.transcript_name)}")


if __name__ == "__main__":
    main()
