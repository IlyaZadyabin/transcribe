# interview-transcribe

Diarized interview transcription using WhisperX and pyannote with interactive speaker identification.

Transform audio/video interviews into readable, speaker-labeled transcripts with just one command.

## Features

- **Automatic transcription** using OpenAI's Whisper (via WhisperX)
- **Speaker diarization** using pyannote.audio 3.1
- **Interactive speaker identification** with sample previews
- **Non-interactive mode** for automation
- **Time-blocked formatting** for easy reading
- **Supports all audio/video formats** (via ffmpeg)

## Prerequisites

### 1. Python Version

This tool requires **Python 3.10, 3.11, 3.12, or 3.13**. Python 3.14+ is not yet supported by WhisperX dependencies.

Check your version:
```bash
python3 --version
```

If you have Python 3.14+, install Python 3.11:
```bash
brew install python@3.11
```

### 2. Install ffmpeg

```bash
brew install ffmpeg
```

### 3. Get a HuggingFace Token

1. Create an account at [HuggingFace](https://huggingface.co/)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)
4. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
5. Accept the terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### 4. Set your token

```bash
export HF_TOKEN=hf_your_token_here
```

Add this to your `~/.zshrc` or `~/.bashrc` to make it permanent:

```bash
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.zshrc
source ~/.zshrc
```

## Installation

### Option 1: Install with pipx (recommended)

```bash
# Install pipx if you don't have it
brew install pipx
pipx ensurepath

# Install interview-transcribe from GitHub (specify Python 3.11 if your default is 3.14+)
pipx install --python python3.11 git+https://github.com/IlyaZadyabin/transcribe.git
```

### Option 2: Install from local clone

```bash
# Clone the repository
git clone https://github.com/IlyaZadyabin/transcribe.git
cd transcribe

# Install with pipx (use Python 3.11 if your default is 3.14+)
pipx install --python python3.11 .
```

### Option 3: Development install

```bash
# Clone and install in editable mode
git clone https://github.com/IlyaZadyabin/transcribe.git
cd transcribe
pip install -e .
```

## Usage

### Interactive Mode (default)

The tool will show you sample quotes from each speaker and ask you to identify them:

```bash
transcribe interview.mov
```

**What happens:**
1. Extracts audio to mono 16kHz WAV
2. Runs WhisperX transcription with diarization
3. Shows you sample quotes from each detected speaker
4. Asks you to identify who's who
5. Generates a formatted transcript with speaker names

### Non-Interactive Mode

Pre-specify speaker names to skip the interactive prompts (useful for batch processing):

```bash
transcribe interview.mov --speakers "Alice,Bob"
```

Speaker names are assigned in order to detected speakers (sorted by ID).

### Advanced Options

```bash
# Different Whisper model (faster but less accurate)
transcribe podcast.mp4 --model medium

# More speakers
transcribe panel.mp3 --min-speakers 3 --max-speakers 5

# Different language
transcribe entrevista.mov --language es

# Custom output directory
transcribe talk.wav --output ./transcripts/talk1

# Longer time blocks in transcript
transcribe lecture.mp4 --block-seconds 60

# Pass token as argument instead of environment variable
transcribe video.mov --token hf_xxx...

# Full example
transcribe conference.mp4 \
  --model large-v2 \
  --max-speakers 4 \
  --block-seconds 45 \
  --output ./transcripts \
  --speakers "Moderator,Alice,Bob,Charlie"
```

### Get Help

```bash
transcribe --help
```

## Output Format

The tool creates a directory (e.g., `out_interview_diar/`) containing:

- `audio.wav` - Extracted mono audio
- `audio.json` - WhisperX output with word-level timestamps
- `transcript.txt` - Formatted transcript (default name)

### Example Transcript

```
00:00:00–00:00:30
Alice: Hi everyone, thanks for joining today's podcast.
Bob: Thanks for having me, Alice. Excited to be here.
Alice: So let's dive right in. What got you started in machine learning?

00:00:30–00:01:00
Bob: Well, it all started back in 2015 when I took Andrew Ng's course...
Alice: Oh yes, that's a classic. I took that too!
```

## Available Whisper Models

WhisperX uses OpenAI's Whisper models. For the full list of available models and their characteristics, see:
- [OpenAI Whisper Models](https://github.com/openai/whisper#available-models-and-languages)

**Recommended models:**
- `large-v3-turbo` (default) - Best quality, fastest large model
- `medium` - Balanced speed/quality for most use cases
- `small` - Fast processing, good for quick drafts

For performance tips and advanced usage, see [WhisperX Usage Guide](https://github.com/m-bain/whisperX?tab=readme-ov-file#english).

## Maintenance

```bash
# Update to latest version from GitHub
pipx reinstall interview-transcribe

# If you made local changes and want to test them
cd /path/to/transcribe
pipx reinstall --python python3.11 .

# Uninstall
pipx uninstall interview-transcribe
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) - Fast automatic speech recognition with word-level timestamps
- [OpenAI Whisper](https://github.com/openai/whisper) - Robust speech recognition
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [ffmpeg](https://ffmpeg.org/) - Multimedia processing