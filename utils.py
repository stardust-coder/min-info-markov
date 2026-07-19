import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Butterworth band-pass filter.

    Parameters
    ----------
    x:
        Input signal.
    fs:
        Sampling frequency in Hz.
    f_low:
        Lower cutoff frequency in Hz.
    f_high:
        Upper cutoff frequency in Hz.
    order:
        Filter order.
    axis:
        Axis along which to filter.
    """
    nyq = fs / 2.0

    if not 0 < f_low < f_high < nyq:
        raise ValueError(
            f"Invalid bandpass range: f_low={f_low}, f_high={f_high}, "
            f"but Nyquist frequency is {nyq} Hz."
        )

    b, a = butter(
        order,
        [f_low / nyq, f_high / nyq],
        btype="band",
    )

    return filtfilt(b, a, x, axis=axis)


def lowpass_filter(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Butterworth low-pass filter.
    """
    nyq = fs / 2.0

    if not 0 < cutoff < nyq:
        raise ValueError(
            f"Invalid lowpass cutoff: cutoff={cutoff}, "
            f"but Nyquist frequency is {nyq} Hz."
        )

    b, a = butter(
        order,
        cutoff / nyq,
        btype="low",
    )

    return filtfilt(b, a, x, axis=axis)


def get_bandpass(
    data: np.ndarray,
    start: float,
    end: float,
    *,
    samplerate: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Backward-compatible wrapper for bandpass_filter.
    """
    import traceback

    print(
        "NEW get_bandpass:",
        "start =", start,
        "end =", end,
        "samplerate =", samplerate,
        "axis =", axis,
    )

    traceback.print_stack(limit=10)
    
    return bandpass_filter(
        data,
        fs=samplerate,
        f_low=start,
        f_high=end,
        order=order,
        axis=axis,
    )


def hilbert_transform(
    signal: np.ndarray,
    *,
    axis: int = -1,
    verbose: bool = False,
):
    """
    Returns analytic signal, envelope, phase, and instantaneous frequency placeholder.

    Returns
    -------
    analytic_signal
    envelope
    phase
    instantaneous_frequency
    """
    analytic_signal = hilbert(signal, axis=axis)
    envelope = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)

    # 必要なら後で np.diff(np.unwrap(phase)) から計算する
    instantaneous_frequency = None

    if verbose:
        print("analytic_signal shape:", analytic_signal.shape)
        print("envelope shape:", envelope.shape)
        print("phase shape:", phase.shape)

    return analytic_signal, envelope, phase, instantaneous_frequency


def tort_modulation_index(A_t, phi_t, n_bins=18):
    """
    Tort et al. modulation index (PAC)
    
    Parameters
    ----------
    A_t : array-like
        振幅（高周波成分のエンベロープなど）
    phi_t : array-like
        位相（低周波成分）
    n_bins : int
        位相ビン数
    
    Returns
    -------
    MI : float
        modulation index
    """
    
    # 位相を -pi ~ pi にラップ
    phi_t = np.angle(np.exp(1j * phi_t))
    
    # ビン作成
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    # 各ビンの平均振幅
    mean_amp = np.zeros(n_bins)
    
    for i in range(n_bins):
        idx = (phi_t >= bins[i]) & (phi_t < bins[i+1])
        if np.any(idx):
            mean_amp[i] = np.mean(A_t[idx])
        else:
            mean_amp[i] = 0.0
    
    # 確率分布に正規化
    P = mean_amp / np.sum(mean_amp)
    
    # ゼロ回避
    P = np.where(P == 0, 1e-10, P)
    
    # 一様分布
    U = np.ones(n_bins) / n_bins
    
    # KL divergence
    D_kl = np.sum(P * np.log(P / U))
    
    # 正規化
    MI = D_kl / np.log(n_bins)
    
    return MI

def tort_mi_surrogate_test(A_t, phi_t, n_bins=18, n_surrogates=200):
    A_t = np.asarray(A_t)
    phi_t = np.asarray(phi_t)

    mi_obs = tort_modulation_index(A_t, phi_t, n_bins=n_bins)

    mi_surr = np.zeros(n_surrogates)
    N = len(A_t)

    for k in range(n_surrogates):
        shift = np.random.randint(N)
        A_shifted = np.roll(A_t, shift)
        mi_surr[k] = tort_modulation_index(A_shifted, phi_t, n_bins=n_bins)

    z = (mi_obs - np.mean(mi_surr)) / np.std(mi_surr)
    p = (np.sum(mi_surr >= mi_obs) + 1) / (n_surrogates + 1)

    return {
        "mi_obs": mi_obs,
        "mi_surr_mean": np.mean(mi_surr),
        "mi_surr_std": np.std(mi_surr),
        "zscore": z,
        "pvalue": p,
        # "mi_surr": mi_surr,
    }

def mean_vector_length(A_t, phi_t):
    """
    Mean Vector Length (MVL) for phase-amplitude coupling
    
    Parameters
    ----------
    A_t : array-like
        振幅（高周波エンベロープ）
    phi_t : array-like
        位相（低周波）
    
    Returns
    -------
    mvl : float
        mean vector length
    """
    
    # 位相を -pi ~ pi にラップ
    phi_t = np.angle(np.exp(1j * phi_t))
    
    # 複素ベクトル
    complex_vec = A_t * np.exp(1j * phi_t)
    
    # 平均ベクトル長
    mvl = np.abs(np.mean(complex_vec))
    
    return mvl

def mvl_surrogate_test(A_t, phi_t, n_surrogates=200):
    A_t = np.asarray(A_t)
    phi_t = np.asarray(phi_t)

    obs = mean_vector_length(A_t, phi_t)
    N = len(A_t)
    surr = np.zeros(n_surrogates)

    for k in range(n_surrogates):
        shift = np.random.randint(N)
        A_shifted = np.roll(A_t, shift)
        surr[k] = mean_vector_length(A_shifted, phi_t)

    z = (obs - np.mean(surr)) / np.std(surr)
    p = (np.sum(surr >= obs) + 1) / (n_surrogates + 1)

    return {
        "mvl_obs": obs,
        "mvl_surr_mean": np.mean(surr),
        "mvl_surr_std": np.std(surr),
        "zscore": z,
        "pvalue": p,
    }




def extract_phase_and_amplitude(
    x,
    fs,
    amp_band=(30, 50),
    phase_band=(5, 12),
    order=4,
):
    """
    x: shape (n, num_channel) or (n,)

    return:
        raw: shape (n, 2 * num_channel)
            [phi_0, A_0, phi_1, A_1, ...]
    """
    x = np.asarray(x)

    if x.ndim == 1:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(f"x must have shape (n, num_channel), got {x.shape}")

    # 高周波帯振幅
    s_amp = bandpass_filter(x, fs, amp_band[0], amp_band[1], order=order)
    analytic_amp = hilbert(s_amp, axis=0)
    A_t = np.abs(analytic_amp)

    # 低周波帯位相
    s_phase = bandpass_filter(x, fs, phase_band[0], phase_band[1], order=order)
    analytic_phase = hilbert(s_phase, axis=0)
    phi_t = np.angle(analytic_phase)

    n, num_channel = x.shape
    raw = np.empty((n, 2 * num_channel), dtype=np.float64)

    raw[:, 0::2] = phi_t
    raw[:, 1::2] = A_t

    return raw