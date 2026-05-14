import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.special import digamma

def bandpass_filter(x, fs, f_low, f_high, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype='band')
    return filtfilt(b, a, x, axis=0)


def lowpass_filter(x, fs, cutoff, order=4):
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, x)


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