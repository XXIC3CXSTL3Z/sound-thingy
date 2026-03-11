import numpy as np

def gen_supersaw(
    freq,
    t,
    voices=7,
    detune_cents=12.0,
    stereo_spread=False,
    phase_random=True,
):
    """
    Generate a simple supersaw by stacking multiple detuned saw waves.

    Parameters
    ----------
    freq : float
        Base frequency in Hz.
    t : np.ndarray
        Time vector.
    voices : int
        Number of saw voices. Best results with odd numbers like 5, 7, 9.
    detune_cents : float
        Maximum detune from center voice, in cents.
    stereo_spread : bool
        If True, returns a 2-channel array [N, 2]. Otherwise mono [N].
    phase_random : bool
        If True, each voice gets a random phase offset.

    Returns
    -------
    np.ndarray
        Mono or stereo supersaw signal.
    """
    if voices < 1:
        raise ValueError("voices must be >= 1")

    if voices == 1:
        detune_offsets = np.array([0.0])
    else:
        detune_offsets = np.linspace(-detune_cents, detune_cents, voices)

    # Saw wave: 2 * frac(x) - 1
    def saw_wave(f, time, phase=0.0):
        x = f * time + phase
        return 2.0 * (x - np.floor(x)) - 1.0

    if stereo_spread:
        left = np.zeros_like(t, dtype=np.float32)
        right = np.zeros_like(t, dtype=np.float32)

        pan_positions = np.linspace(-1.0, 1.0, voices) if voices > 1 else np.array([0.0])

        for i, cents in enumerate(detune_offsets):
            detuned_freq = freq * (2 ** (cents / 1200.0))
            phase = np.random.rand() if phase_random else 0.0
            voice = saw_wave(detuned_freq, t, phase=phase)

            pan = pan_positions[i]
            left_gain = np.sqrt((1.0 - pan) / 2.0)
            right_gain = np.sqrt((1.0 + pan) / 2.0)

            left += voice * left_gain
            right += voice * right_gain

        stereo = np.stack([left, right], axis=-1)
        stereo /= max(np.max(np.abs(stereo)), 1e-12)
        return stereo.astype(np.float32)

    else:
        out = np.zeros_like(t, dtype=np.float32)

        for cents in detune_offsets:
            detuned_freq = freq * (2 ** (cents / 1200.0))
            phase = np.random.rand() if phase_random else 0.0
            out += saw_wave(detuned_freq, t, phase=phase)

        out /= max(np.max(np.abs(out)), 1e-12)
        return out.astype(np.float32)

def gen_909_kick(
    fs: int,
    length_s: float = 0.6,      # 909/trance kicks are often ~300–600ms
    f_start: float = 180.0,      # initial pitch for the "knock"
    f_end: float = 55.0,         # tail pitch (tune this to your track key-ish)
    pitch_drop_s: float = 0.045, # how fast it falls from f_start to f_end
    amp_attack_s: float = 0.001, # fast attack
    amp_decay_s: float = 0.35,   # long-ish decay for trance
    click_s: float = 0.003,      # short click length
    click_level: float = 0.25,   # click mix
    drive: float = 1.8,          # soft saturation amount
) -> np.ndarray:
    """
    909-ish kick synthesis:
      - Exponential pitch drop sine (body)
      - Exponential amp envelope
      - Short high-frequency click transient
      - Gentle tanh saturation
    Returns mono float32 array.
    """
    n = int(fs * length_s)
    t = np.arange(n, dtype=np.float32) / fs

    # --- Pitch envelope (exponential glide) ---
    # frequency(t) transitions quickly, then settles near f_end
    # use exp(-t/tau) style curve
    tau = max(pitch_drop_s, 1e-4)
    freq = f_end + (f_start - f_end) * np.exp(-t / tau)

    # Phase accumulation for time-varying frequency
    phase = 2.0 * np.pi * np.cumsum(freq) / fs
    body = np.sin(phase).astype(np.float32)

    # --- Amp envelope (fast attack, exponential decay) ---
    attack_n = max(1, int(fs * amp_attack_s))
    env = np.ones(n, dtype=np.float32)
    env[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False, dtype=np.float32)

    # exponential decay after attack
    decay_tau = max(amp_decay_s, 1e-4)
    env *= np.exp(-t / decay_tau).astype(np.float32)

    kick = body * env

    # --- Click (short, bright transient) ---
    click_n = max(1, int(fs * click_s))
    ct = np.arange(click_n, dtype=np.float32) / fs
    # a bright pip + a touch of noise is a decent "beater" click
    click = (np.sin(2*np.pi*4000*ct) + 0.3*np.random.randn(click_n).astype(np.float32))
    click *= np.exp(-ct / max(click_s/3, 1e-4)).astype(np.float32)  # super fast decay
    kick[:click_n] += click_level * click

    # --- Gentle saturation (adds harmonics, makes it loud) ---
    kick = np.tanh(drive * kick).astype(np.float32)

    # --- Normalize to safe peak ---
    peak = float(np.max(np.abs(kick)) + 1e-12)
    kick = (0.95 * kick / peak).astype(np.float32)
    return kick



def gen_signal(freq, t_local):
    signal = np.zeros_like(t_local)
    harmonics = 17
    for n in range(1, harmonics + 1):
        signal += (1/n) * np.sin(2 * np.pi * freq * n * t_local)
    signal /= np.max(np.abs(signal)) + 1e-12
    return signal



def fast_dist(
    x: np.ndarray,
    pre: float = 2.0,        # input drive (like PRE)
    thresh: float = 2.5,     # curve intensity (like THRESH)
    mix: float = 0.7,        # 0=dry, 1=wet (like MIX)
    post: float = 0.8,       # output trim (like POST)
    mode: str = "tanh",      # "tanh", "atan", "hard"
    dc_block: bool = True,   # helps bass stay clean
    eps: float = 1e-12
) -> np.ndarray:
    """
    Fast distortion / waveshaper in the style of FL 'Fruity Fast Dist':
      PRE  -> pre
      THRESH -> thresh (controls how quickly it saturates)
      MIX  -> mix
      POST -> post

    x should be float audio, typically in [-1, 1].
    Returns float32 array.
    """
    x = np.asarray(x, dtype=np.float32)

    # Optional DC blocker (simple 1st-order highpass)
    if dc_block:
        # y[n] = x[n] - x[n-1] + R*y[n-1]
        R = 0.995
        y = np.empty_like(x)
        y_prev = 0.0
        x_prev = 0.0
        for n in range(len(x)):
            y_curr = x[n] - x_prev + R * y_prev
            y[n] = y_curr
            y_prev = y_curr
            x_prev = x[n]
        x_in = y
    else:
        x_in = x

    # PRE gain
    x_pre = pre * x_in

    # Waveshaper (THRESH controls curvature)
    if mode == "tanh":
        wet = np.tanh(thresh * x_pre)
    elif mode == "atan":
        wet = (2.0 / np.pi) * np.arctan(thresh * x_pre)
    elif mode == "hard":
        # thresh here acts like inverse threshold: bigger thresh => harder clip
        T = 1.0 / max(thresh, eps)
        wet = np.clip(x_pre, -T, T) / max(T, eps)
    else:
        raise ValueError("mode must be one of: 'tanh', 'atan', 'hard'")

    # MIX (dry/wet)
    m = float(np.clip(mix, 0.0, 1.0))
    out = (1.0 - m) * x_in + m * wet

    # POST gain
    out *= post

    # Safety normalize if it gets too hot (optional but practical)
    peak = float(np.max(np.abs(out)) + eps)
    if peak > 1.0:
        out = out / peak

    return out.astype(np.float32)
    

def adsr(fs, t_local, A=0.03, D=0.03, S=0.3, R=0.1):
    note_duration = len(t_local) / fs
    assert A + D + R <= note_duration

    env = np.zeros_like(t_local)
    a = int(A * fs)
    d = int(D * fs)
    r = int(R * fs)
    s = len(t_local) - (a + d + r)

    if a > 0:
        env[:a] = np.linspace(0, 1, a, endpoint=False)
    if d > 0:
        env[a:a+d] = np.linspace(1, S, d, endpoint=False)
    env[a+d:a+d+s] = S
    if r > 0:
        env[a+d+s:] = np.linspace(S, 0, r, endpoint=False)
    return env
