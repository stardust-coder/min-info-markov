import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import random
from itertools import combinations

def simulate_Gaussian_VAR(dim,order,steps):
    #Model parameters
    phi = [0.5*np.identity(dim) for _ in range(order)]
    if (dim,order) == (1,1):
        phi = [0.5*np.identity(dim)]
    elif (dim,order) == (1,2):
        phi = [0.5*np.identity(dim),0.3*np.identity(dim)]
    elif (dim,order) == (1,3):
        phi = [0.5*np.identity(dim),0.3*np.identity(dim),0.1*np.identity(dim)]
    elif dim == 1:
        phi = [0.5*np.identity(dim) for _ in range(order)]
    elif (dim,order) == (2,1):
        phi[0][0][0] = 0.5
        phi[0][1][0] = 0.1
        phi[0][0][1] = 0.1
        phi[0][1][1] = 0.5
    elif (dim,order) == (2,2):
        phi[0][0][0] = 0.5
        phi[0][1][0] = 0.1
        phi[0][0][1] = 0.1
        phi[0][1][1] = 0.5
        phi[1][0][0] = 0.3
        phi[1][1][0] = 0.1
        phi[1][0][1] = 0.1
        phi[1][1][1] = 0.3
    else:
        raise ValueError("dim, orderを見直してください.")

    for item in phi:
        assert item.shape == (dim,dim)
    assert len(phi) == order

    sigma = np.identity(dim) * 0.5 ### noise variance
    assert sigma.shape == (dim,dim)

    mean = np.zeros((dim))
    assert mean.shape == (dim,)

    ### Stationarity check
    check_stationarity = False
    if check_stationarity:
        coeffs = [1] + [-phi_.item() for phi_ in phi]  # 特性方程式：1 - φ₁z - φ₂z² - φ₃z³ = 0
        coeffs.reverse()
        roots = np.roots(coeffs)
        is_stationary = np.all(np.abs(roots) > 1)
        print("Is stationary?", is_stationary)
    ### Data generation
    var_data = []
    for _ in range(order):
        var_data.append(np.zeros((1,dim))) # initial value
    for _ in range(order,steps+order):
        v = np.zeros((dim,1))
        for k in range(1,order+1):
            v += phi[k-1]@var_data[-k].T #A^k x_{t-k}            
        v += np.random.multivariate_normal(mean, sigma, 1).T
        var_data.append(v.T)
    var_data = np.array(var_data[order:])
    
    assert (dim,order) in [(1,1),(1,2),(1,3),(2,1)] #AR(1),AR(2),AR(3),VAR(1)
    if order == 1:
        Theta = [phi[0].T@np.linalg.inv(sigma)]
        Theta = np.concatenate(Theta)
    if (dim, order) == (1,2):
        phi1 = phi[0].item()
        phi2 = phi[1].item()
        sigma2 = sigma[0][0].item()
        Theta = [(phi1 - phi1*phi2)/sigma2, phi2/sigma2]
        Theta = np.array([Theta])
    if (dim, order) == (1,3):
        phi1 = phi[0].item()
        phi2 = phi[1].item()
        phi3 = phi[2].item()
        sigma2 = sigma[0][0].item()
        Theta = [(phi1 - phi1*phi2 - phi2*phi3)/sigma2, (phi2 - phi1*phi3)/sigma2, phi3/sigma2]
        Theta = np.array([Theta])
    
    Theta = [phi[p].T@np.linalg.inv(sigma) for p in range(order)]
    Theta = np.concatenate(Theta)
    print("True Parameter (estimation target):", Theta.flatten())
    return var_data[:,0,:], Theta.flatten()

def simulate_t_VAR(dim,order,steps):
    from student import sample_from_mininfo_markov
    var_data, true_param = sample_from_mininfo_markov(steps)
    return var_data, true_param.flatten()



def erdos_renyi_edges(n, p, directed=False, self_loop=False, seed=None):
    """
    n: ノード数。ノードは 1, 2, ..., n
    p: 各 edge を採用する確率
    directed: True なら有向グラフ
    self_loop: True なら (i, i) も許す
    seed: 乱数シード
    """
    rng = random.Random(seed)
    nodes = range(1, n + 1)

    if directed:
        candidates = [
            (i, j)
            for i in nodes
            for j in nodes
            if self_loop or i != j
        ]
    else:
        if self_loop:
            candidates = [(i, j) for i in nodes for j in nodes if i <= j]
        else:
            candidates = list(combinations(nodes, 2))

    edges = [
        (i, j)
        for i, j in candidates
        if rng.random() < p
    ]

    return edges

def Kuramoto_Model(N, seed=None, verbose=False):
    rng = np.random.default_rng(seed)
    # edge = [(1, 2), (2, 3), (3, 4), (4, 5)]
    edge = erdos_renyi_edges(n=N, p=0.2, seed=seed)
    print("True edges in Kuramoto model:")
    adj = {i: [] for i in range(1, N + 1)}

    for i, j in edge:
        adj[i].append(j)
        adj[j].append(i) 

    K = np.zeros((N, N))  # 結合強度
    for (i, j) in edge:
        K[i - 1][j - 1] = 25 * 0.9
        K[j - 1][i - 1] = 25 * 0.9  # 非対称Kのときはこちらのみ.

    if N <= 5:
        print("K in Kuramoto model:")
        print(K)
    else:
        with open("25dim", "w", encoding="utf-8") as f:
            for i in range(1, N + 1):
                neighbors = sorted(adj[i])
                line = f"{i}: " + " ".join(map(str, neighbors))

                print(line)
                f.write(line + "\n")
    
    T = 15
    dt = 0.01
    steps = int(T / dt)
    print("#Time steps:", steps)

    # 初期位相と自然周波数
    theta = rng.uniform(0, 2 * np.pi, N)
    omega = rng.normal(11.5, 2, N)

    # 結果保存用
    theta_history = np.zeros((steps, N))
    R_history = []

    # 時間発展
    for t in range(steps):
        theta = np.mod(theta, 2 * np.pi)
        theta_history[t] = theta

        dtheta = np.zeros(N)
        for i in range(N):
            coupling = np.sum(K[i, :] * np.sin(theta - theta[i]))
            dtheta[i] = omega[i] + (1.0 / N) * coupling

        theta += dtheta * dt

        noise = rng.vonmises(0, 300, N)
        theta += noise

        real = np.sum(np.cos(theta)) / N
        imag = np.sum(np.sin(theta)) / N
        R = (real**2 + imag**2) ** 0.5
        R_history.append(R)

    if verbose:
        plt.clf()

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(steps) * dt, R_history, label="R", color="red")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Order Parameter R")
        plt.title("Kuramoto Model")
        plt.savefig("./output/kuramoto_R.png")
        plt.show()
        plt.clf()

        plt.figure(figsize=(10, 6))
        for i in range(N):
            plt.plot(
                np.arange(steps)[-100:] * dt,
                theta_history[-100:, i],
                alpha=0.5,
                label=f"{i + 1}",
            )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Phase θ")
        plt.title("Kuramoto Model")
        plt.savefig("./output/kuramoto.png")
        plt.show()
        plt.clf()

    return theta_history


def artificial_PAC_data_Tort():
    from tensorpac.signals import pac_signals_tort
    n_epochs = 20    # number of trials
    sf = 1000.        # sampling frequency
    T = 0.5          # one trials time (sec)
    n_times = sf * T # number of time points

    # Create artificially coupled signals using Tort method :
    data, time = pac_signals_tort(f_pha=10, f_amp=100, noise=2, n_epochs=n_epochs, 
                                dpha=10, damp=10, sf=sf, n_times=n_times)
    return data


def make_pink_noise(alpha: float, L: int, dt: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    MATLAB: make_pink_noise(alpha,L,dt)
    FFT の振幅スペクトルを |f|^{-alpha} で整形してから逆FFTして 1/f^alpha ノイズを作る。
    """
    if rng is None:
        rng = np.random.default_rng()

    x = rng.standard_normal(L)
    xf = np.fft.fft(x)
    A = np.abs(xf)
    phase = np.angle(xf)

    # 周波数軸（両側）。絶対値にして |f|^{-alpha} を適用。
    f = np.abs(np.fft.fftfreq(L, d=dt))
    # f=0 の発散回避
    with np.errstate(divide="ignore"):
        one_over_f = 1.0 / (f ** alpha)
    one_over_f[0] = 0.0

    Anew = A * one_over_f
    xf_new = Anew * np.exp(1j * phase)
    x_new = np.fft.ifft(xf_new).real
    return x_new


def make_nadalin_signal(pac_mod: float,
           aac_mod: float,
           sim_method: str,
           rng: np.random.Generator | None = None):
    """
    MATLAB: [XX,P,Vlo,Vhi,t] = simfun(pac_mod,aac_mod,sim_method,pval,ci,AIC[,q])

    pac_mod : PAC 強度（高周波包絡を低周波ピーク同期のハン窓で変調）
    aac_mod : AAC 強度（高/低の振幅–振幅結合の強さ）
    sim_method : 'GLM' | 'pink' | 'spiking'
    pval, ci, AIC : GLM 推定側のオプション（ここでは受け渡しだけ）
    q : 省略可。GLM 実装側にそのまま渡す
    rng : numpy.random.Generator（再現性が欲しいときに指定）

    戻り値:
      XX, P : GLM 出力（ここではスタブ）。実装があれば差し替えてください。
      Vlo, Vhi : 低/高周波バンド信号（最終的に観測信号から再抽出したもの）
      t : 時間軸（秒）
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- シミュレーション条件
    dt = 0.002
    Fs = 1.0 / dt
    fNQ = Fs / 2.0
    N = int(20 / dt + 4000)  # 端のフィルタ歪みを後で捨て、正味 20s にする

    # 共通因子
    rho_ = 0.0
    common = make_pink_noise(1.0, N, dt, rng)

    # ---- 低周波バンドを作る（ピンクノイズ→FIRLS→filtfilt）
    Vpink = make_pink_noise(1.0, N, dt, rng) * (1-rho_) + common * rho_
    Vpink -= Vpink.mean()

    def _firls_bandpass(locut, hicut, order_scale):
        # MATLAB の firls(order, f, m) は「次数」指定。SciPy の firls は「タップ数」指定なので +1。
        filtorder = int(order_scale * (Fs / locut))
        if filtorder % 2 != 0:
            filtorder += 1
        numtaps = filtorder + 1
        trans = 0.15
        bands = [0,
                 (1 - trans) * (locut / fNQ),
                 (locut / fNQ),
                 (hicut / fNQ),
                 (1 + trans) * (hicut / fNQ),
                 1.0]
        desired = [0, 0, 1, 1, 0, 0]
        b = signal.firls(numtaps, bands, desired)
        a = 1.0
        return b, a

    # Low: 
    bL, aL = _firls_bandpass(4.0, 7.0, order_scale=3)
    Vlo = signal.filtfilt(bL, aL, Vpink)

    # ---- 高周波バンド用のピンクノイズを作り直し
    Vpink = make_pink_noise(1.0, N, dt, rng)  * (1-rho_) + common * rho_
    Vpink -= Vpink.mean()

    # High: 
    bH, aH = _firls_bandpass(100.0, 140.0, order_scale=10)
    Vhi = signal.filtfilt(bH, aH, Vpink)

    # ---- フィルタ端の 4 s を両端で捨てる（MATLAB: 2001:end-2000）
    Vlo = Vlo[2000:-2000]
    Vhi = Vhi[2000:-2000]
    t = np.arange(1, Vlo.size + 1) * dt
    N = Vlo.size

    # ---- 低周波ピークに同期した変調窓 s(t) を作る（ハン窓 21 点）
    peaks, _ = signal.find_peaks(Vlo)
    AmpLo = np.abs(signal.hilbert(Vlo))  # 低周波の包絡
    s = np.zeros_like(Vhi)
    for idx in peaks:
        if 10 < idx < (Vhi.size - 10):
            s[idx - 10: idx + 11] = np.hanning(21)
    s /= (s.max() + np.finfo(float).eps)  # 0–1 正規化

    # ---- sim_method 切り替え
    if sim_method == 'GLM':
        # Ahi(t) = 1 + pac_mod * s(t)
        Ahi = 1.0 + pac_mod * s
        # φ_hi は元の高周波から Hilbert 位相を抽出し、振幅 Ahi でコサイン化
        phi_hi = np.angle(signal.hilbert(Vhi))
        Vhi = 0.01 * Ahi * np.cos(phi_hi)
        # AAC: 高/低の振幅相関（AmpLo の正規化でスケーリング）
        Vhi = Vhi * (1.0 + aac_mod * AmpLo / AmpLo.max())

    elif sim_method == 'pink':
        # 高周波の包絡をピーク同期窓で AM し、さらに低周波包絡で AM
        Vhi = Vhi * (1.0 + pac_mod * s)
        Vhi = Vhi * (1.0 + aac_mod * AmpLo / AmpLo.max())

    else:
        raise ValueError("sim_method must be 'GLM' or 'pink'.")

    Vpink2 = make_pink_noise(1.0, N, dt, rng)
    noise_level = 0.01

    ###rare spike factor
    n_spikes = 0                 # まずは 20–50 程度に
    spike_halfw = int(0.015*Fs)   # 半幅 ~15ms（合計 ~30ms）; Low も少し拾うならもう少し大きく
    theta_spk = np.zeros(N)

    for idx in rng.choice(N, n_spikes, replace=False):
        i0 = max(0, idx - spike_halfw)
        i1 = min(N, idx + spike_halfw)
        theta_spk[i0:i1] += np.hanning(i1 - i0)
    
    Vlo += theta_spk
    Vhi += theta_spk

    V1 = Vlo + Vhi + noise_level * Vpink2

    # Low:（再び抽出）
    Vlo = signal.filtfilt(bL, aL, V1)
    # High: （再び抽出）
    Vhi = signal.filtfilt(bH, aH, V1)

    return V1, Vlo, Vhi, t


def make_two_burst_phase_locked_pac_signal(
    fs=1000,
    duration=5.0,
    f_phase=6.0,
    f_amp=40.0,
    burst_centers=(1.5, 3.5),
    burst_width=0.20,
    burst_gain=1.5,
    baseline_gain=0.02,
    phase_lock_strength=0.9,
    preferred_phase=0.0,
    low_amp=1.0,
    noise_std=0.01,
    random_state=0,
    biphasic=False
):
    rng = np.random.default_rng(random_state)
    t = np.arange(0, duration, 1/fs)

    phi = 2 * np.pi * f_phase * t
    low_component = low_amp * np.sin(phi)

    burst_env = np.full_like(t, baseline_gain, dtype=float)
    for c in burst_centers:
        burst_env += burst_gain * np.exp(-0.5 * ((t - c) / burst_width) ** 2)

    # 位相依存の振幅変調
    if biphasic:
        phase_mod = 1 + phase_lock_strength * np.cos(2*(phi - preferred_phase))
    else:
        phase_mod = 1 + phase_lock_strength * np.cos(phi - preferred_phase)
    # phase_mod = np.clip(phase_mod, 0, None)

    envelope = burst_env * phase_mod
    high_component = envelope * np.sin(2 * np.pi * f_amp * t)

    noise = rng.normal(0, noise_std, size=t.shape)
    x = low_component + high_component + noise

    return t, x, low_component, high_component, envelope


from utils import bandpass_filter
from scipy.signal import butter, filtfilt, hilbert
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


def simulated_data(mode="burst_pac"):
    meta = {}
    if mode == "burst_pac":
        fs = 1000
        t, x, low_component, high_component, envelope_true = make_two_burst_phase_locked_pac_signal(
            fs=fs,
            duration=5.0,
            f_phase=6.0,
            f_amp=40.0,
            burst_centers=(1.5, 3.5),
            burst_width=0.20,
            burst_gain=1.8,
            baseline_gain=0.03,
            low_amp=1.0,
            noise_std=0.01,
            random_state=0,
            biphasic=False
        )
        amp_band = (30, 50)
        phase_band = (5, 12)

    else:
        fs = 500
        x, _, _, t = make_nadalin_signal(pac_mod=1,aac_mod=0,sim_method="pink",rng=np.random.default_rng(0))
        amp_band = (100, 140)
        phase_band = (4, 7)

    meta["fs"] = fs
    meta["amp_band"] = amp_band
    meta["phase_band"] = phase_band

    return x, meta