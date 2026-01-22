# Jam Agent

A text-controlled drum machine. Play a 16-step drum loop and modify it in real-time with simple commands.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python drum_agent.py
```

Then type `play` to start the loop.

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `play` | Start playback | |
| `stop` | Stop playback | |
| `set <inst> <steps>` | Set pattern for instrument | `set kick 1 5 9 13` |
| `clear <inst\|all>` | Clear pattern | `clear snare` |
| `volume <inst> <0-100>` | Set volume | `volume hihat 60` |
| `tempo <bpm>` | Set tempo (30-300) | `tempo 120` |
| `show` | Display current pattern | |
| `help` | Show help | |
| `quit` | Exit | |

## Instruments

- `kick`
- `snare`
- `hihat`

## Samples

Place your own `.wav` samples in the `samples/` directory:
- `kick.wav`
- `snare.wav`
- `hihat.wav`
