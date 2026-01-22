#!/usr/bin/env python3
"""
Drum Loop Agent - A text-controlled drum machine
"""
from __future__ import annotations

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from pathlib import Path

SAMPLE_RATE = 44100
STEPS = 16


class DrumMachine:
    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = Path(samples_dir)
        self.samples = {}
        self.pattern = {
            "kick": [],
            "snare": [],
            "hihat": [],
        }
        self.volume = {
            "kick": 100,
            "snare": 100,
            "hihat": 80,
        }
        self.tempo = 90
        self.playing = False
        self._stream = None
        self._playback_position = 0
        self._loop_buffer = None
        self._lock = threading.Lock()

        self._load_samples()
        self._set_default_pattern()
        self._rebuild_loop()

    def _load_samples(self):
        """Load wav samples from disk."""
        for name in ["kick", "snare", "hihat"]:
            path = self.samples_dir / f"{name}.wav"
            if path.exists():
                data, sr = sf.read(str(path))
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                # Resample if needed
                if sr != SAMPLE_RATE:
                    ratio = SAMPLE_RATE / sr
                    new_length = int(len(data) * ratio)
                    data = np.interp(
                        np.linspace(0, len(data), new_length),
                        np.arange(len(data)),
                        data
                    )
                self.samples[name] = data.astype(np.float32)
                print(f"Loaded {name}")
            else:
                print(f"Warning: {path} not found")

    def _set_default_pattern(self):
        """Set a basic starting pattern."""
        self.pattern["kick"] = [1, 9]
        self.pattern["snare"] = [5, 13]
        self.pattern["hihat"] = [1, 3, 5, 7, 9, 11, 13, 15]

    def _step_samples(self) -> int:
        """Number of audio samples per step (16th note)."""
        beats_per_second = self.tempo / 60
        steps_per_beat = 4  # 16th notes
        step_duration = 1 / (beats_per_second * steps_per_beat)
        return int(step_duration * SAMPLE_RATE)

    def _rebuild_loop(self):
        """Rebuild the loop buffer from current pattern."""
        step_len = self._step_samples()
        loop_len = step_len * STEPS

        # Find max sample length to ensure we have room
        max_sample_len = max((len(s) for s in self.samples.values()), default=0)
        buffer_len = loop_len + max_sample_len

        new_buffer = np.zeros(buffer_len, dtype=np.float32)

        for instrument, steps in self.pattern.items():
            if instrument not in self.samples:
                continue
            sample = self.samples[instrument]
            vol = self.volume.get(instrument, 100) / 100

            for step in steps:
                if 1 <= step <= STEPS:
                    start = (step - 1) * step_len
                    end = start + len(sample)
                    new_buffer[start:end] += sample * vol

        # Clip to prevent distortion
        new_buffer = np.clip(new_buffer, -1, 1)

        with self._lock:
            self._loop_buffer = new_buffer
            self._loop_length = loop_len

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for continuous audio stream."""
        with self._lock:
            if self._loop_buffer is None:
                outdata.fill(0)
                return

            output = np.zeros(frames, dtype=np.float32)
            pos = self._playback_position
            loop_len = self._loop_length

            # Fill output buffer, wrapping around loop
            remaining = frames
            out_pos = 0
            while remaining > 0:
                chunk_size = min(remaining, loop_len - pos)
                output[out_pos:out_pos + chunk_size] = self._loop_buffer[pos:pos + chunk_size]
                pos = (pos + chunk_size) % loop_len
                out_pos += chunk_size
                remaining -= chunk_size

            self._playback_position = pos
            outdata[:, 0] = output

    def play(self):
        """Start playback."""
        if self.playing:
            print("Already playing")
            return

        self._rebuild_loop()
        self._playback_position = 0

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024
        )
        self._stream.start()
        self.playing = True
        print(f"Playing at {self.tempo} BPM")

    def stop(self):
        """Stop playback."""
        if not self.playing:
            print("Not playing")
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self.playing = False
        print("Stopped")

    def set_pattern(self, instrument: str, steps: list[int]):
        """Set pattern for an instrument."""
        if instrument not in self.pattern:
            print(f"Unknown instrument: {instrument}")
            return
        valid_steps = [s for s in steps if 1 <= s <= STEPS]
        self.pattern[instrument] = valid_steps
        self._rebuild_loop()
        print(f"{instrument}: {valid_steps}")

    def clear(self, instrument: str):
        """Clear pattern for an instrument or all."""
        if instrument == "all":
            for inst in self.pattern:
                self.pattern[inst] = []
            print("Cleared all patterns")
        elif instrument in self.pattern:
            self.pattern[instrument] = []
            print(f"Cleared {instrument}")
        else:
            print(f"Unknown instrument: {instrument}")
            return
        self._rebuild_loop()

    def set_volume(self, instrument: str, level: int):
        """Set volume for an instrument (0-100)."""
        if instrument not in self.volume:
            print(f"Unknown instrument: {instrument}")
            return
        self.volume[instrument] = max(0, min(100, level))
        self._rebuild_loop()
        print(f"{instrument} volume: {self.volume[instrument]}")

    def set_tempo(self, bpm: int):
        """Set tempo in BPM."""
        self.tempo = max(30, min(300, bpm))
        self._rebuild_loop()
        print(f"Tempo: {self.tempo} BPM")

    def show(self):
        """Display current pattern."""
        print(f"\nTempo: {self.tempo} BPM")
        print("Pattern (steps 1-16):")
        print("-" * 40)
        for instrument, steps in self.pattern.items():
            vol = self.volume.get(instrument, 100)
            grid = ["." for _ in range(STEPS)]
            for s in steps:
                if 1 <= s <= STEPS:
                    grid[s - 1] = "X"
            grid_str = " ".join(grid)
            print(f"{instrument:6} [{vol:3}%]: {grid_str}")
        print("-" * 40)


def print_help():
    """Print available commands."""
    print("""
Commands:
  play                    Start playback
  stop                    Stop playback
  set <inst> <steps...>   Set pattern (e.g., set kick 1 5 9 13)
  clear <inst|all>        Clear pattern
  volume <inst> <0-100>   Set volume (e.g., volume snare 80)
  tempo <bpm>             Set tempo (e.g., tempo 120)
  show                    Display current pattern
  help                    Show this help
  quit                    Exit

Instruments: kick, snare, hihat
Steps: 1-16 (16th notes in one bar)
""")


def parse_command(dm: DrumMachine, cmd: str):
    """Parse and execute a command."""
    parts = cmd.strip().lower().split()
    if not parts:
        return True

    command = parts[0]

    if command == "quit" or command == "exit":
        dm.stop()
        return False
    elif command == "play":
        dm.play()
    elif command == "stop":
        dm.stop()
    elif command == "set" and len(parts) >= 2:
        instrument = parts[1]
        steps = []
        for p in parts[2:]:
            try:
                steps.append(int(p))
            except ValueError:
                pass
        dm.set_pattern(instrument, steps)
    elif command == "clear" and len(parts) >= 2:
        dm.clear(parts[1])
    elif command == "volume" and len(parts) >= 3:
        try:
            dm.set_volume(parts[1], int(parts[2]))
        except ValueError:
            print("Invalid volume value")
    elif command == "tempo" and len(parts) >= 2:
        try:
            dm.set_tempo(int(parts[1]))
        except ValueError:
            print("Invalid tempo value")
    elif command == "show":
        dm.show()
    elif command == "help":
        print_help()
    else:
        print(f"Unknown command: {cmd}. Type 'help' for commands.")

    return True


def main():
    print("Drum Loop Agent")
    print("=" * 40)

    dm = DrumMachine()
    dm.show()
    print("\nType 'help' for commands, 'play' to start\n")

    running = True
    while running:
        try:
            cmd = input("> ")
            running = parse_command(dm, cmd)
        except KeyboardInterrupt:
            print("\nStopping...")
            dm.stop()
            break
        except EOFError:
            break

    print("Goodbye!")


if __name__ == "__main__":
    main()
