"""Analysis and plotting for Year 2 Earth-field NMR & 2D spin-echo MRI lab data."""

from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from prospa_io import read_par, read_prospa_binary, read_t1_txt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

# Constants from lab notebook
GAMMA_1H = 2.675e8     # rad s^-1 T^-1
GAMMA_19F = 2.518e8    # rad s^-1 T^-1
L_COIL = 0.3685        # H (AnalyseCoil fit: 0.3685 +/- 0.0006 H)
C0_COIL = 4.30e-9      # F (AnalyseCoil fit: 4.30 +/- 0.03 nF)


def compute_fft(signal: np.ndarray, acq_time: float, zero_fill: int = 1, tau_exp: float | None = None):
    """Return time axis, frequency axis (Hz), and spectral magnitude scaled by dt."""
    y = signal - np.mean(signal)
    n = len(y)
    dt = acq_time / n
    t = np.arange(n) * dt
    if tau_exp is not None:
        y = y * np.exp(-t / tau_exp)
    n_fft = n * zero_fill
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    mag = np.abs(np.fft.rfft(y, n=n_fft)) * dt
    return t, freqs, mag


def find_peak_and_fwhm(freqs: np.ndarray, mag: np.ndarray, f_lo: float, f_hi: float) -> tuple[float, float, float]:
    """Find peak frequency, peak magnitude, and FWHM in [f_lo, f_hi]."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    f_sub, m_sub = freqs[mask], mag[mask]
    idx = int(np.argmax(m_sub))
    peak_f, peak_m = float(f_sub[idx]), float(m_sub[idx])
    half = 0.5 * peak_m

    left = idx
    while left > 0 and m_sub[left] > half:
        left -= 1
    right = idx
    while right < len(m_sub) - 1 and m_sub[right] > half:
        right += 1

    fwhm = float(f_sub[right] - f_sub[left])
    return peak_f, peak_m, fwhm


def plot_shimming_and_noise() -> dict:
    """Compare unshimmed vs shimmed 1H FIDs, autoshim convergence, and mains noise."""
    runs = [
        ("Unshimmed (0, 0, 0)", DATA / "1d/hydrogen_unshimmed", "#d95f02"),
        ("Manual shim (-1.5, 1.75, 0.25)", DATA / "1d/hydrogen_manual_shim", "#7570b3"),
        ("Auto-shimmed (-1.5, 2.28, 0.55)", DATA / "1d/hydrogen_autoshim", "#1b9e77"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=160)
    stats = {}

    for label, folder, color in runs:
        par = read_par(folder / "acqu.par")
        sig = read_prospa_binary(folder / "data.1d")
        t, freqs, mag = compute_fft(sig, float(par["acqTime"]), zero_fill=2)
        f0, pk, fwhm = find_peak_and_fwhm(freqs, mag, 1950, 1995)
        stats[label] = (f0, pk, fwhm)

        if "Manual" not in label:
            axes[0, 0].plot(t, sig, lw=0.45, alpha=0.8, color=color, label=label)
        band = (freqs >= 1955) & (freqs <= 1990)
        axes[0, 1].plot(freqs[band], mag[band], lw=1.4, color=color,
                        label=f"{label.split(' (')[0]} (FWHM {fwhm:.2f} Hz)")

    axes[0, 0].set_title("1H FID time-domain coherence")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Signal (uV)")
    axes[0, 0].set_xlim(0, 3.0)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].set_title("1H Larmor peak sharpening with shimming")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Spectral amplitude (uV s)")
    axes[0, 1].legend(loc="upper left", fontsize=8)

    shim_trace = read_prospa_binary(DATA / "1d/autoshim_steps/data.1d")
    axes[1, 0].plot(np.arange(1, len(shim_trace) + 1), shim_trace, "-o", ms=3, lw=1.2, color="#2b5c8f")
    axes[1, 0].set_title("3-axis bisection auto-shim convergence")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Peak signal (a.u.)")

    bad_par = read_par(DATA / "1d/hydrogen_overshim/acqu.par")
    bad_sig = read_prospa_binary(DATA / "1d/hydrogen_overshim/data.1d")
    _, bad_freqs, bad_mag = compute_fft(bad_sig, float(bad_par["acqTime"]))
    low = (bad_freqs >= 10) & (bad_freqs <= 600)
    axes[1, 1].plot(bad_freqs[low], bad_mag[low], lw=1.0, color="#b2182b")
    for h in (50, 150, 450):
        idx = int(np.argmin(np.abs(bad_freqs - h)))
        axes[1, 1].annotate(f"{h} Hz", (bad_freqs[idx], bad_mag[idx]), xytext=(5, 2),
                            textcoords="offset points", fontsize=8)
    axes[1, 1].set_title("Over-shimmed failure: 50 Hz mains harmonics")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Spectral amplitude (uV s)")
    axes[1, 1].set_ylim(0, bad_mag[low].max() * 1.18)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_shimming_and_noise.png")
    plt.close(fig)
    return stats


def _t1_buildup(t_ms: np.ndarray, s_inf: float, t1_ms: float) -> np.ndarray:
    return s_inf * (1.0 - np.exp(-t_ms / t1_ms))


def plot_t1_relaxation() -> list[tuple[str, float, float]]:
    """Fit T1 polarisation build-up across three water runs."""
    runs = [
        ("Run 1 (6 A, 1 scan)", DATA / "t1/water_run1/data.txt", "#1b9e77", "o"),
        ("Run 2 (6 A, 1 scan)", DATA / "t1/water_run2/data.txt", "#7570b3", "s"),
        ("Run 3 (7 A, 4 scans)", DATA / "t1/water_run3/data.txt", "#d95f02", "^"),
    ]

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    t_grid = np.linspace(0, 5200, 200)
    results = []

    for label, path, color, marker in runs:
        t_ms, sig = read_t1_txt(path)
        popt, pcov = curve_fit(_t1_buildup, t_ms, sig, p0=(sig.max() * 1.2, 2500.0))
        s_inf, t1_ms = popt
        t1_err = float(np.sqrt(np.diag(pcov))[1])
        results.append((label, float(t1_ms), t1_err))

        ax_raw.plot(t_ms, sig, marker, color=color, ms=5,
                    label=f"{label}: T1 = {t1_ms / 1000:.2f} +/- {t1_err / 1000:.2f} s")
        ax_raw.plot(t_grid, _t1_buildup(t_grid, *popt), "-", color=color, lw=1.3, alpha=0.85)
        ax_norm.plot(t_ms, sig / s_inf, marker, color=color, ms=5, label=label)

    weights = np.array([1.0 / (err ** 2) for _, _, err in results])
    t1_vals = np.array([t1 for _, t1, _ in results])
    t1_mean = float(np.sum(weights * t1_vals) / np.sum(weights))
    ax_norm.plot(t_grid, 1.0 - np.exp(-t_grid / t1_mean), "k--", lw=1.3,
                 label=f"Weighted mean: T1 = {t1_mean / 1000:.2f} s")

    ax_raw.set_title("Water T1 polarisation build-up (raw)")
    ax_raw.set_xlabel("Polarising duration t (ms)")
    ax_raw.set_ylabel("Signal amplitude (a.u.)")
    ax_raw.legend(fontsize=8)

    ax_norm.set_title("Normalised build-up S(t) / S_inf")
    ax_norm.set_xlabel("Polarising duration t (ms)")
    ax_norm.set_ylabel("Normalised magnetisation")
    ax_norm.legend(fontsize=8)

    for ax in (ax_raw, ax_norm):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_t1_relaxation.png")
    plt.close(fig)
    return results


def plot_fluorine_spectrum(f_proton_ref: float) -> tuple[float, float]:
    """Plot 80-average C2F6 spectrum resolving 19F alongside residual 1H."""
    folder = DATA / "1d/fluorine_c2f6"
    par = read_par(folder / "acqu.par")
    sig = read_prospa_binary(folder / "data.1d")

    _, freqs, mag_apod = compute_fft(sig, float(par["acqTime"]), zero_fill=2, tau_exp=0.25)
    _, _, mag_raw = compute_fft(sig, float(par["acqTime"]), zero_fill=2)

    f_19f, p_19f, _ = find_peak_and_fwhm(freqs, mag_apod, 1845, 1875)
    f_1h, p_1h, _ = find_peak_and_fwhm(freqs, mag_apod, 1965, 1985)
    f_19f_pred = f_proton_ref * (GAMMA_19F / GAMMA_1H)

    band = (freqs >= 1810) & (freqs <= 2010)
    scale_apod = mag_apod[band].max()
    scale_raw = mag_raw[band].max()

    fig, ax = plt.subplots(figsize=(8.8, 4.1), dpi=160)
    ax.plot(freqs[band], mag_raw[band] / scale_raw, lw=0.8, color="#bdbdbd", label="Unwindowed FFT")
    ax.plot(freqs[band], mag_apod[band] / scale_apod, lw=1.5, color="#d95f02",
            label="Matched exp window (tau = 0.25 s)")
    ax.axvline(f_19f_pred, color="#2b5c8f", ls="--", lw=1.1,
               label=f"Expected 19F from 1H ref ({f_19f_pred:.1f} Hz)")

    ax.annotate(f"19F (~{f_19f:.1f} Hz)", xy=(f_19f, p_19f / scale_apod),
                xytext=(f_19f - 36, 0.85), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    ax.annotate("39x50 Hz\nmains spike", xy=(1950.0, 0.98),
                xytext=(1913, 0.84), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=8)
    ax.annotate(f"Residual 1H\n(~{f_1h:.1f} Hz)", xy=(f_1h, p_1h / scale_apod),
                xytext=(f_1h + 8, 0.84), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)

    ax.set_title("Perfluoroethane (C2F6), 80 averages: 19F vs residual 1H")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_xlim(1810, 2010)
    ax.set_ylim(0, 1.14)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_fluorine_spectrum.png")
    plt.close(fig)
    return f_19f, f_19f_pred


def reconstruct_2d_mri(kspace: np.ndarray, remove_zipper: bool = True, apodise_echo: bool = False) -> np.ndarray:
    """Reconstruct magnitude MRI image from Prospa 2D spin-echo k-space."""
    k = kspace.copy()
    if remove_zipper:
        # Remove phase-independent stimulated echo / DC baseline across phase-encode steps
        k = k - (np.median(k.real, axis=0, keepdims=True) + 1j * np.median(k.imag, axis=0, keepdims=True))
    if apodise_echo:
        ny, nx = k.shape
        echo_col = int(np.argmax(np.mean(np.abs(k), axis=0)))
        rr, cc = np.ogrid[:ny, :nx]
        win = np.exp(-0.5 * (((rr - ny // 2) / (0.25 * ny)) ** 2 + ((cc - echo_col) / (0.22 * nx)) ** 2))
        k = k * win
    return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(k))))


def plot_2d_mri() -> None:
    """Plot raw vs cleaned water phantom (32x32) and overnight easy-peelers scan (128x128)."""
    tubes_par = read_par(DATA / "2d/water_air_tubes_yz/acqu.par")
    tubes_k = read_prospa_binary(DATA / "2d/water_air_tubes_yz/data.2d")
    tubes_raw = reconstruct_2d_mri(tubes_k, remove_zipper=False)
    tubes_clean = reconstruct_2d_mri(tubes_k, remove_zipper=True)

    or_par = read_par(DATA / "2d/easy_peelers_overnight/acqu.par")
    or_k = read_prospa_binary(DATA / "2d/easy_peelers_overnight/data.2d")
    or_img = np.rot90(reconstruct_2d_mri(or_k, remove_zipper=True, apodise_echo=True), 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), dpi=160)

    ext_tubes = [0, float(tubes_par["FOVr"]), 0, float(tubes_par["FOVp1"])]
    axes[0].imshow(tubes_raw, cmap="inferno", origin="lower", extent=ext_tubes)
    axes[0].set_title("Water + 3 air tubes (raw 2D FFT)")
    axes[0].set_xlabel("Readout (mm)")
    axes[0].set_ylabel("Phase encode (mm)")

    axes[1].imshow(tubes_clean, cmap="inferno", origin="lower", extent=ext_tubes)
    axes[1].set_title("Water + 3 air tubes (zipper removed)")
    axes[1].set_xlabel("Readout (mm)")
    axes[1].set_ylabel("Phase encode (mm)")

    ext_or = [0, float(or_par["FOVp1"]), 0, float(or_par["FOVr"])]
    axes[2].imshow(
        or_img,
        cmap="inferno",
        origin="lower",
        extent=ext_or,
        vmin=np.percentile(or_img, 45),
        vmax=np.percentile(or_img, 99.7),
    )
    axes[2].set_title("3 easy peelers (128x128, 17 h, 10 scans)")
    axes[2].set_xlabel("Z (mm)")
    axes[2].set_ylabel("X (mm)")

    fig.tight_layout()
    fig.savefig(FIGURES / "04_spin_echo_mri.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    coil_sig = read_prospa_binary(DATA / "1d/analyse_coil/data.1d")
    _, c_freqs, c_mag = compute_fft(coil_sig, 0.1)
    ring_f, _, _ = find_peak_and_fwhm(c_freqs, c_mag, 500, 3500)
    f_self = 1.0 / (2.0 * np.pi * np.sqrt(L_COIL * C0_COIL))

    shim_stats = plot_shimming_and_noise()
    f_proton = shim_stats["Auto-shimmed (-1.5, 2.28, 0.55)"][0]
    b0_uT = (2.0 * np.pi * f_proton / GAMMA_1H) * 1e6
    c_ext_nF = (1.0 / (L_COIL * (2.0 * np.pi * f_proton) ** 2) - C0_COIL) * 1e9

    t1_results = plot_t1_relaxation()
    f_19f, f_19f_pred = plot_fluorine_spectrum(f_proton)
    plot_2d_mri()

    print("=== Earth-Field NMR Summary ===")
    print(f"B1 coil self-resonance (L={L_COIL} H, C0={C0_COIL*1e9:.2f} nF): {f_self:.0f} Hz (ring-down peak: {ring_f:.0f} Hz)")
    print(f"Auto-shimmed 1H frequency: {f_proton:.2f} Hz -> B0 = {b0_uT:.2f} uT (phone sensor: ~45 uT)")
    print(f"External capacitance for {f_proton:.1f} Hz: {c_ext_nF:.2f} nF (used 13.35-13.45 nF)")
    for label, (f0, _, fwhm) in shim_stats.items():
        print(f"  {label}: f0 = {f0:.2f} Hz, FWHM = {fwhm:.2f} Hz")
    print("Water T1 polarisation build-up fits:")
    for label, t1_ms, err in t1_results:
        print(f"  {label}: T1 = ({t1_ms:.0f} +/- {err:.0f}) ms")
    print(f"19F peak in C2F6: {f_19f:.1f} Hz (expected {f_19f_pred:.1f} Hz from 1H reference)")
    print(f"Saved 4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
