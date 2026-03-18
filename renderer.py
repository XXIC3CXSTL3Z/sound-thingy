from dataclasses import dataclass, field
import numpy as np
import sounddevice as sd
import pydub

from methods import (
    adsr,
    gen_signal,
    gen_supersaw,
    gen_909_kick,
    fast_dist,
)


def transpose(freq: float, semitones: float) -> float:
    return freq * (2 ** (semitones / 12.0))


def read_mp3(path: str, normalized: bool = True):
    """Load MP3 into numpy array."""
    audio = pydub.AudioSegment.from_mp3(path)
    y = np.array(audio.get_array_of_samples())

    if audio.channels == 2:
        y = y.reshape((-1, 2)).mean(axis=1)  # mono mixdown

    y = y.astype(np.float32)

    if normalized:
        peak = np.max(np.abs(y)) + 1e-12
        y = y / peak

    return audio.frame_rate, y


@dataclass
class RenderConfig:
    fs: int = 48000
    bpm: int = 140
    root_freq: float = 261.63
    num_beats: int = 64
    clap_path: str = "sound.mp3"

    lead_gain_boost: float = 1.50
    master_gain: float = 0.90

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def samples_per_beat(self) -> int:
        return int(self.fs * self.seconds_per_beat)

    @property
    def samples_per_16th(self) -> int:
        return self.samples_per_beat // 4

    @property
    def samples_per_8th(self) -> int:
        return self.samples_per_beat // 2

    @property
    def duration_16th(self) -> float:
        return self.seconds_per_beat / 4.0

    @property
    def duration_8th(self) -> float:
        return self.seconds_per_beat / 2.0

    @property
    def total_samples(self) -> int:
        return self.num_beats * self.samples_per_beat

    @property
    def arp_root(self) -> float:
        return transpose(self.root_freq, -12)

    @property
    def bass_root(self) -> float:
        return transpose(self.root_freq, -24)

    @property
    def lead_root(self) -> float:
        return self.root_freq


@dataclass
class Renderer:
    cfg: RenderConfig
    arp_events: list[dict] = field(default_factory=list)
    supersaw_events: list[dict] = field(default_factory=list)
    bass_events: list[dict] = field(default_factory=list)
    clap_events: list[dict] = field(default_factory=list)
    super_events: list[list[dict] = field(default_factory=list)
    def __post_init__(self):
        if not self.arp_events:
            self.arp_events = [
                {"start_beats": 0.00, "duration_beats": 0.25, "semitones": 0, "gain": 0.55},
                {"start_beats": 0.50, "duration_beats": 0.25, "semitones": 7, "gain": 0.55},
                {"start_beats": 1.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.55},
                {"start_beats": 1.50, "duration_beats": 0.25, "semitones": 3, "gain": 0.55},
                {"start_beats": 2.00, "duration_beats": 0.25, "semitones": 0, "gain": 0.55},
                {"start_beats": 2.50, "duration_beats": 0.25, "semitones": 7, "gain": 0.55},
                {"start_beats": 3.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.55},
                {"start_beats": 3.50, "duration_beats": 0.25, "semitones": 3, "gain": 0.55},
            ]

        if not self.supersaw_events:
            self.supersaw_events = [
                {"start_beats": 0.00, "duration_beats": 0.50, "semitones": 7,  "gain": 0.35},
                {"start_beats": 0.50, "duration_beats": 0.25, "semitones": 10, "gain": 0.35},
                {"start_beats": 0.75, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
                {"start_beats": 1.00, "duration_beats": 0.50, "semitones": 14, "gain": 0.35},
                {"start_beats": 1.50, "duration_beats": 0.50, "semitones": 10, "gain": 0.35},
                {"start_beats": 2.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
                {"start_beats": 2.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
                {"start_beats": 2.50, "duration_beats": 0.50, "semitones": 15, "gain": 0.35},
                {"start_beats": 3.00, "duration_beats": 1.00, "semitones": 17, "gain": 0.35},

                {"start_beats": 4.00, "duration_beats": 0.50, "semitones": 7,  "gain": 0.35},
                {"start_beats": 4.50, "duration_beats": 0.25, "semitones": 10, "gain": 0.35},
                {"start_beats": 4.75, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
                {"start_beats": 5.00, "duration_beats": 0.50, "semitones": 14, "gain": 0.35},
                {"start_beats": 5.50, "duration_beats": 0.50, "semitones": 10, "gain": 0.35},
                {"start_beats": 6.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
                {"start_beats": 6.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
                {"start_beats": 6.50, "duration_beats": 0.50, "semitones": 15, "gain": 0.35},
                {"start_beats": 7.00, "duration_beats": 1.00, "semitones": 19, "gain": 0.35},

                {"start_beats": 8.00, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
                {"start_beats": 8.25, "duration_beats": 0.25, "semitones": 15, "gain": 0.35},
                {"start_beats": 8.50, "duration_beats": 0.50, "semitones": 17, "gain": 0.35},
                {"start_beats": 9.00, "duration_beats": 0.50, "semitones": 19, "gain": 0.35},
                {"start_beats": 9.50, "duration_beats": 0.50, "semitones": 17, "gain": 0.35},
                {"start_beats": 10.00, "duration_beats": 0.25, "semitones": 15, "gain": 0.35},
                {"start_beats": 10.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
                {"start_beats": 10.50, "duration_beats": 0.50, "semitones": 12, "gain": 0.35},
                {"start_beats": 11.00, "duration_beats": 1.00, "semitones": 10, "gain": 0.35},

                {"start_beats": 12.00, "duration_beats": 0.50, "semitones": 14, "gain": 0.35},
                {"start_beats": 12.50, "duration_beats": 0.25, "semitones": 15, "gain": 0.35},
                {"start_beats": 12.75, "duration_beats": 0.25, "semitones": 17, "gain": 0.35},
                {"start_beats": 13.00, "duration_beats": 0.50, "semitones": 19, "gain": 0.35},
                {"start_beats": 13.50, "duration_beats": 0.50, "semitones": 17, "gain": 0.35},
                {"start_beats": 14.00, "duration_beats": 0.25, "semitones": 15, "gain": 0.35},
                {"start_beats": 14.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
                {"start_beats": 14.50, "duration_beats": 0.50, "semitones": 12, "gain": 0.35},
                {"start_beats": 15.00, "duration_beats": 1.00, "semitones": 7,  "gain": 0.40},
            ]

        if not self.bass_events:
            self.bass_events = [
                {"start_beats": 0.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 1.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 2.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 3.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 4.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 5.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 6.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 7.50, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},

                {"start_beats": 8.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 9.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 10.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 11.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 12.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 13.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 14.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
                {"start_beats": 15.50, "duration_beats": 0.5, "semitones": 3, "gain": 0.5},
            ]

        if not self.clap_events:
            self.clap_events = [
                {"start_beats": 1.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 3.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 5.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
                {"start_beats": 7.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
            ]

    def _repeat_pattern(self, events: list[dict], pattern_length_beats: float, repeats: int, gain_boost: float = 0.0):
        out = []
        for i in range(repeats):
            offset = i * pattern_length_beats
            for event in events:
                out.append({
                    "start_beats": event["start_beats"] + offset,
                    "duration_beats": event["duration_beats"],
                    "semitones": event["semitones"],
                    "gain": event["gain"] + gain_boost,
                })
        return out

    def _mix_note(self, timeline: np.ndarray, start_sample: int, note: np.ndarray, gain: float):
        end_sample = min(len(timeline), start_sample + len(note))
        timeline[start_sample:end_sample] += gain * note[: end_sample - start_sample]

    def render(self) -> np.ndarray:
        cfg = self.cfg

        fs = cfg.fs
        seconds_per_beat = cfg.seconds_per_beat
        samples_per_beat = cfg.samples_per_beat
        total_samples = cfg.total_samples

        timeline = np.zeros(total_samples, dtype=np.float32)

        t_8th = np.linspace(0, cfg.duration_8th, cfg.samples_per_8th, endpoint=False)

        lead_root = cfg.lead_root
        arp_root = cfg.arp_root
        bass_root = cfg.bass_root

        supersaw_events_full = self._repeat_pattern(
            self.supersaw_events,
            pattern_length_beats=16.0,
            repeats=int(np.ceil(cfg.num_beats / 16.0)),
            gain_boost=cfg.lead_gain_boost,
        )

        for event in supersaw_events_full:
            start_sample = int(event["start_beats"] * samples_per_beat)
            note_samples = int(event["duration_beats"] * samples_per_beat)
            duration_sec = event["duration_beats"] * seconds_per_beat

            t_note = np.linspace(0, duration_sec, note_samples, endpoint=False)
            freq = transpose(lead_root, event["semitones"])
            env = adsr(fs, t_note, A=0.001, D=0.012, S=0.1, R=0.01)
            note = gen_supersaw(freq, t_note, voices=11, detune_cents=18) * env

            self._mix_note(timeline, start_sample, note.astype(np.float32), event["gain"])

        arp_events_full = self._repeat_pattern(
            self.arp_events,
            pattern_length_beats=4.0,
            repeats=int(np.ceil(cfg.num_beats / 4.0)),
        )

        for event in arp_events_full:
            start_sample = int(event["start_beats"] * samples_per_beat)
            note_samples = int(event["duration_beats"] * samples_per_beat)
            duration_sec = event["duration_beats"] * seconds_per_beat

            t_note = np.linspace(0, duration_sec, note_samples, endpoint=False)
            freq = transpose(arp_root, event["semitones"] - 12)
            env = adsr(fs, t_note, A=0.001, D=0.012, S=0.1, R=0.01)
            note = gen_signal(freq, t_note) * env

            self._mix_note(timeline, start_sample, note.astype(np.float32), event["gain"])

        bass_events_full = []
        kick_events_full = []
        for event in self._repeat_pattern(
            self.bass_events,
            pattern_length_beats=16.0,
            repeats=int(np.ceil(cfg.num_beats / 16.0)),
        ):
            bass_events_full.append(event)
            kick_events_full.append({
                "start_beats": event["start_beats"] - 0.5,
                "duration_beats": event["duration_beats"],
                "semitones": event["semitones"],
                "gain": event["gain"] + 0.5,
            })

        clap_events_full = self._repeat_pattern(
            self.clap_events,
            pattern_length_beats=8.0,
            repeats=int(np.ceil(cfg.num_beats / 8.0)),
            gain_boost=0.75,
        )

        bass_detune_cents = [-6, 0, 6]
        kick = gen_909_kick(fs).astype(np.float32)

        _, clap = read_mp3(cfg.clap_path, normalized=True)
        threshold = 0.02
        start = np.argmax(np.abs(clap) > threshold)
        clap = clap[start:].astype(np.float32)

        for event in bass_events_full:
            start_sample = int(event["start_beats"] * samples_per_beat)
            note_samples = int(event["duration_beats"] * samples_per_beat)
            duration_sec = event["duration_beats"] * seconds_per_beat

            t_note = np.linspace(0, duration_sec, note_samples, endpoint=False)
            env = adsr(fs, t_note, A=0.001, D=0.012, S=0.1, R=0.01)

            base_freq = transpose(bass_root, event["semitones"])
            osc = np.zeros_like(t_note, dtype=np.float32)

            for cents in bass_detune_cents:
                freq = base_freq * (2 ** (cents / 1200.0))
                osc += gen_signal(freq, t_note).astype(np.float32)

            bass_note = osc * env
            bass_note = fast_dist(
                bass_note,
                pre=4.0,
                thresh=4.0,
                mix=0.8,
                post=0.7,
                mode="hard",
            ).astype(np.float32)

            self._mix_note(timeline, start_sample, bass_note, 2.0 * event["gain"])

        for event in kick_events_full:
            start_sample = int(event["start_beats"] * samples_per_beat)
            self._mix_note(timeline, start_sample, kick, event["gain"])

        for event in clap_events_full:
            start_sample = int(event["start_beats"] * samples_per_beat)
            self._mix_note(timeline, start_sample, clap, event["gain"])

        peak = np.max(np.abs(timeline)) + 1e-12
        timeline = cfg.master_gain * (timeline / peak)

        return timeline.astype(np.float32)

    def play(self):
        audio = self.render()
        sd.play(audio, self.cfg.fs)
        sd.wait()


if __name__ == "__main__":
    cfg = RenderConfig(
        fs=48000,
        bpm=140,
        root_freq=261.63,
        num_beats=64,
        clap_path="sound.mp3",
    )

    renderer = Renderer(cfg)
    renderer.play()
