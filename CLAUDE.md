# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Setup (first time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python drum_agent.py
```

## Architecture

Single-file CLI drum machine (`drum_agent.py`) with a 16-step sequencer.

**DrumMachine class** handles:
- Sample loading from `samples/` directory (kick.wav, snare.wav, hihat.wav)
- Pattern storage: dict mapping instrument name → list of step numbers (1-16)
- Audio playback via `sounddevice.OutputStream` with callback-based streaming
- Loop buffer: pre-rendered audio array rebuilt on any pattern/volume/tempo change

**Audio pipeline:**
1. `_load_samples()` - loads WAV files, converts to mono, resamples to 44100Hz
2. `_rebuild_loop()` - renders full bar into `_loop_buffer` by placing samples at step positions
3. `_audio_callback()` - streams from buffer with seamless looping, protected by threading lock

**Command parsing:** `parse_command()` handles CLI input → method calls on DrumMachine instance.

## Key Constants

- `SAMPLE_RATE = 44100`
- `STEPS = 16` (16th notes per bar)
- Tempo range: 30-300 BPM
- Volume range: 0-100 per instrument
