import numpy as np
import sounddevice as sd
from methods import *
# from scipy.io.wavfile import read
import pydub

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float(y) / 2**15
    else:
        return a.frame_rate, y
        


def transpose(freq, semitones):
    return freq * (2 ** (semitones / 12))


# Audio / tempo
fs = 48000
bpm = 140
root_freq = transpose(261.63, 19)

seconds_per_beat = 60 / bpm
samples_per_beat = int(fs * seconds_per_beat)

# Rhythmic subdivisions
samples_per_16th = samples_per_beat // 4
samples_per_8th = samples_per_beat // 2

duration_16th = seconds_per_beat / 4
duration_8th = seconds_per_beat / 2

t_16th = np.linspace(0, duration_16th, samples_per_16th, endpoint=False)
t_8th = np.linspace(0, duration_8th, samples_per_8th, endpoint=False)

# Song length
num_beats = 64
total_samples = num_beats * samples_per_beat
timeline = np.zeros(total_samples, dtype=np.float32)

# Instrument roots
arp_root = transpose(root_freq, -12)
bass_root = transpose(root_freq, -48)


# Arp events
arp_events = [
    {"start_beats": 0.00, "duration_beats": 0.25, "semitones": 0, "gain": 0.55},
    {"start_beats": 0.50, "duration_beats": 0.25, "semitones": 7, "gain": 0.55},
    {"start_beats": 1.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.55},
    {"start_beats": 1.50, "duration_beats": 0.25, "semitones": 3, "gain": 0.55},
    {"start_beats": 2.00, "duration_beats": 0.25, "semitones": 0, "gain": 0.55},
    {"start_beats": 2.50, "duration_beats": 0.25, "semitones": 7, "gain": 0.55},
    {"start_beats": 3.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.55},
    {"start_beats": 3.50, "duration_beats": 0.25, "semitones": 3, "gain": 0.55},
]

supersaw_events = [

    # Bar 1
    {"start_beats": 0.00, "duration_beats": 0.50, "semitones": 7,  "gain": 0.35},
    {"start_beats": 0.50, "duration_beats": 0.25, "semitones": 10, "gain": 0.35},
    {"start_beats": 0.75, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
    {"start_beats": 1.00, "duration_beats": 0.50, "semitones": 14, "gain": 0.35},
    {"start_beats": 1.50, "duration_beats": 0.50, "semitones": 10, "gain": 0.35},
    {"start_beats": 2.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
    {"start_beats": 2.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
    {"start_beats": 2.50, "duration_beats": 0.50, "semitones": 15, "gain": 0.35},
    {"start_beats": 3.00, "duration_beats": 1.00, "semitones": 17, "gain": 0.35},

    # Bar 2
    {"start_beats": 4.00, "duration_beats": 0.50, "semitones": 7,  "gain": 0.35},
    {"start_beats": 4.50, "duration_beats": 0.25, "semitones": 10, "gain": 0.35},
    {"start_beats": 4.75, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
    {"start_beats": 5.00, "duration_beats": 0.50, "semitones": 14, "gain": 0.35},
    {"start_beats": 5.50, "duration_beats": 0.50, "semitones": 10, "gain": 0.35},
    {"start_beats": 6.00, "duration_beats": 0.25, "semitones": 12, "gain": 0.35},
    {"start_beats": 6.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
    {"start_beats": 6.50, "duration_beats": 0.50, "semitones": 15, "gain": 0.35},
    {"start_beats": 7.00, "duration_beats": 1.00, "semitones": 19, "gain": 0.35},

    # Bar 3 (variation)
    {"start_beats": 8.00, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
    {"start_beats": 8.25, "duration_beats": 0.25, "semitones": 15, "gain": 0.35},
    {"start_beats": 8.50, "duration_beats": 0.50, "semitones": 17, "gain": 0.35},
    {"start_beats": 9.00, "duration_beats": 0.50, "semitones": 19, "gain": 0.35},
    {"start_beats": 9.50, "duration_beats": 0.50, "semitones": 17, "gain": 0.35},
    {"start_beats": 10.00, "duration_beats": 0.25, "semitones": 15, "gain": 0.35},
    {"start_beats": 10.25, "duration_beats": 0.25, "semitones": 14, "gain": 0.35},
    {"start_beats": 10.50, "duration_beats": 0.50, "semitones": 12, "gain": 0.35},
    {"start_beats": 11.00, "duration_beats": 1.00, "semitones": 10, "gain": 0.35},

    # Bar 4 (resolution)
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

supersaw_events_full = []
pattern_length_beats = 16.0
num_repeats = int(np.ceil(num_beats / pattern_length_beats))
for i in range(num_repeats):
    offset = i * pattern_length_beats
    for event in supersaw_events:
        supersaw_events_full.append({
            "start_beats": event["start_beats"] + offset,
            "duration_beats": event["duration_beats"],
            "semitones": event["semitones"],
            "gain": event["gain"] + 1.50,
        })
        
lead_root = transpose(root_freq, 0)

for event in supersaw_events_full:
    start_sample = int(event["start_beats"] * samples_per_beat)
    note_samples = int(event["duration_beats"] * samples_per_beat)
    duration_sec = event["duration_beats"] * seconds_per_beat

    t_note = np.linspace(0, duration_sec, note_samples, endpoint=False)
    freq = transpose(lead_root, event["semitones"])
    env = adsr(fs, t_note, A=0.001, D=0.012, S=0.1, R=0.01)
    note = gen_supersaw(freq, t_note, voices=11, detune_cents=18) * env

    end_sample = min(len(timeline), start_sample + len(note))    
    timeline[start_sample:end_sample] += event["gain"] * note[: end_sample - start_sample]

arp_events_full = []
pattern_length_beats = 4.0
num_repeats = int(np.ceil(num_beats / pattern_length_beats))
for i in range(num_repeats):
    offset = i * pattern_length_beats
    for event in arp_events:
        arp_events_full.append({
            "start_beats": event["start_beats"] + offset,
            "duration_beats": event["duration_beats"],
            "semitones": event["semitones"],
            "gain": event["gain"],
        })

for event in arp_events_full:
    start_sample = int(event["start_beats"] * samples_per_beat)
    note_samples = int(event["duration_beats"] * samples_per_beat)
    duration_sec = event["duration_beats"] * seconds_per_beat

    t_note = np.linspace(0, duration_sec, note_samples, endpoint=False)
    freq = transpose(arp_root, event["semitones"] - 12)
    env = adsr(fs, t_note, A=0.001, D=0.012, S=0.1, R=0.01)

    note = gen_signal(freq, t_note) * env

    end_sample = min(len(timeline), start_sample + note_samples)
    timeline[start_sample:end_sample] += event["gain"] * note[: end_sample - start_sample]

# Bass / kick step patterns
bass_events = [
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
clap_events = [
    {"start_beats": 1.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
    {"start_beats": 3.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
    {"start_beats": 5.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
    {"start_beats": 7.00, "duration_beats": 0.5, "semitones": 0, "gain": 0.5},
]

bass_events_full = []
kick_events_full = []
clap_events_full = []
pattern_length_beats = 16.0
num_repeats = int(np.ceil(num_beats / pattern_length_beats))
for i in range(num_repeats):
    offset = i * pattern_length_beats
    for event in bass_events:
        bass_events_full.append({
            "start_beats": event["start_beats"] + offset,
            "duration_beats": event["duration_beats"],
            "semitones": event["semitones"],
            "gain": event["gain"],
        })
        kick_events_full.append({
            "start_beats": event["start_beats"] - 0.5 + offset,
            "duration_beats": event["duration_beats"],
            "semitones": event["semitones"],
            "gain": event["gain"] + 0.5,
        })
pattern_length_beats = 8.0
num_repeats = int(np.ceil(num_beats / pattern_length_beats))
for i in range(num_repeats):
    offset = i * pattern_length_beats
    for event in clap_events:
        clap_events_full.append({
            "start_beats": event["start_beats"] + offset,
            "duration_beats": event["duration_beats"],
            "semitones": event["semitones"],
            "gain": event["gain"] + 0.75,
        })
    
bass_env = adsr(fs, t_8th, A=0.001, D=0.012, S=0.1, R=0.01)
kick = gen_909_kick(fs)
print(len(kick))
clap = read("sound.mp3")[1].astype(np.float32)
clap /= np.max(np.abs(clap)) + 1e-12
threshold = 0.02
start = np.argmax(np.abs(clap) > threshold)
clap = clap[start:]
print(len(clap))

bass_detune_cents = [-6, 0, 6]

for event in bass_events_full:
    start_sample = int(event["start_beats"] * samples_per_beat)
    note_samples = int(event["duration_beats"] * samples_per_beat)
    duration_sec = event["duration_beats"] * seconds_per_beat

    t_note = np.linspace(0, duration_sec, note_samples, endpoint=False)
    env = adsr(fs, t_note, A=0.001, D=0.012, S=0.1, R=0.01)

    osc = 0.0
    base_freq = transpose(bass_root, event["semitones"])

    for cents in bass_detune_cents:
        freq = base_freq * (2 ** (cents / 1200))
        osc += gen_signal(freq, t_note)

    bass_note = osc * env
    bass_note = fast_dist(bass_note, pre=4.0, thresh=4.0, mix=0.8, post=0.7, mode="hard")

    end_sample = min(len(timeline), start_sample + note_samples)
    timeline[start_sample:end_sample] += 2* event["gain"] * bass_note[: end_sample - start_sample]


for event in kick_events_full:
    start_sample = int(event["start_beats"] * samples_per_beat)
    note_samples = int(event["duration_beats"] * samples_per_beat)
    duration_sec = event["duration_beats"] * seconds_per_beat
    end_sample = min(len(timeline), start_sample + len(kick))    
    timeline[start_sample:end_sample] += kick[: end_sample - start_sample] * event["gain"]

for event in clap_events_full:
    start_sample = int(event["start_beats"] * samples_per_beat)
    end_sample = min(len(timeline), start_sample + len(clap))
    timeline[start_sample:end_sample] += event["gain"] * clap[: end_sample - start_sample]

# Normalize
peak = np.max(np.abs(timeline)) + 1e-12
timeline = 0.9 * (timeline / peak)

# Playback
sd.play(timeline, fs)
sd.wait()
