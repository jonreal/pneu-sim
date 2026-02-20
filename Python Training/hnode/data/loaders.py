import os
import numpy as np
import jax.numpy as jnp
from scipy.signal import butter, filtfilt
import scipy.io
from scipy.optimize import minimize

# ------------------------ File Path Generator ------------------------
def generate_file_paths(base_dir, prefix="nn_", suffix="", codes=None, ranges=None):

    file_paths = []

    if codes is not None:
        for code in codes:
            fname = f"{prefix}{code}{suffix}"
            fp = os.path.join(base_dir, fname)
            if os.path.exists(fp):
                file_paths.append(fp)

    elif ranges is not None:
        for r in ranges:
            # allow ("10","80"), (10,80), or "10-80"
            if isinstance(r, (tuple, list)) and len(r) == 2:
                start, end = int(r[0]), int(r[1])
            else:
                start, end = map(int, str(r).split("-"))

            values = list(range(start, end + 1, 5))

            for xx in values:
                for yy in values:
                    fname = f"{prefix}{xx:02d}{yy:02d}{suffix}"
                    fp = os.path.join(base_dir, fname)
                    if os.path.exists(fp):
                        file_paths.append(fp)

    else:
        # Raw-code default set
        default_codes = ["1010", "4545", "8080", "1045", "1080", "8010", "8045"]
        for code in default_codes:
            fname = f"{prefix}{code}{suffix}"
            fp = os.path.join(base_dir, fname)
            if os.path.exists(fp):
                file_paths.append(fp)

    return file_paths

# ------------------------ Load Poly44 Fit Models ------------------------
_module_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_module_dir))
_poly44_path = os.path.join(_project_root, 'poly44_all_fits.mat')

data = scipy.io.loadmat(_poly44_path)
powers = data['powers']
fF_loadingP    = data['coeffs_fF_loadingP'].flatten()
fF_unloadingP  = data['coeffs_fF_unloadingP'].flatten()
fP_loadingP    = data['coeffs_fP_loadingP'].flatten()
fP_unloadingP  = data['coeffs_fP_unloadingP'].flatten()

def make_poly_fn(coeffs, powers):
    def poly_fn(x, y):
        return sum(c * x**i * y**j for c, (i, j) in zip(coeffs, powers))
    return poly_fn

fF_loadingP_fn    = make_poly_fn(fF_loadingP, powers)
fF_unloadingP_fn  = make_poly_fn(fF_unloadingP, powers)
fP_loadingP_fn    = make_poly_fn(fP_loadingP, powers)
fP_unloadingP_fn  = make_poly_fn(fP_unloadingP, powers)

# ------------------------ EXACT raw compute_me_mf (with m_atm) ------------------------
def compute_me_mf(Pf, Pe, preload, x0_1, x0_2):
    # Case A: loading flexion, unloading extension
    FP_flexion_A = lambda m, th: fP_loadingP_fn(m, -th + preload)
    FF_flexion_A = lambda m, th: fF_loadingP_fn(m, -th + preload)
    FP_extension_A = lambda m, th: fP_unloadingP_fn(m, th + preload)
    FF_extension_A = lambda m, th: fF_unloadingP_fn(m, th + preload)

    # Case B: unloading flexion, loading extension
    FP_flexion_B = lambda m, th: fP_unloadingP_fn(m, -th + preload)
    FF_flexion_B = lambda m, th: fF_unloadingP_fn(m, -th + preload)
    FP_extension_B = lambda m, th: fP_loadingP_fn(m, th + preload)
    FF_extension_B = lambda m, th: fF_loadingP_fn(m, th + preload)

    # Objective and constraint builder
    def make_obj(FP_flex, FP_ext, Pf, Pe):
        return lambda x: np.linalg.norm([
            FP_flex(x[0], x[2]) - Pf,
            FP_ext(x[1], x[2]) - Pe
        ])

    def make_eq_con(FF_flex, FF_ext):
        return {'type': 'eq', 'fun': lambda x: FF_flex(x[0], x[2]) - FF_ext(x[1], x[2])}

    # Optimization setup
    bounds1 = [(0, 0.2), (0, 0.2), (x0_1, x0_1)]
    bounds2 = [(0, 0.2), (0, 0.2), (x0_2, x0_2)]

    # Solve Case A
    res_A = minimize(
        make_obj(FP_flexion_A, FP_extension_A, Pf, Pe),
        [0.1, 0.1, x0_1], method='SLSQP',
        bounds=bounds1,
        constraints=[make_eq_con(FF_flexion_A, FF_extension_A)],
        options={'maxiter': 10000, 'ftol': 1e-8}
    )

    # Solve Case B
    res_B = minimize(
        make_obj(FP_flexion_B, FP_extension_B, Pf, Pe),
        [0.1, 0.1, x0_2], method='SLSQP',
        bounds=bounds2,
        constraints=[make_eq_con(FF_flexion_B, FF_extension_B)],
        options={'maxiter': 10000, 'ftol': 1e-8}
    )

    # Final output (unchanged)
    mf = 0.5 * (res_A.x[0] + res_B.x[0])
    me = 0.5 * (res_A.x[1] + res_B.x[1])
    xeq = [res_B.x[2], res_A.x[2]]  # xeq for +vel and -vel
    return mf, me, xeq

# ------------------------ EXACT raw readfile ------------------------
def readfile(file_path, min_idx=2000, thr=0.01):
    cols = [0,1,2,3,6,9]; M = []
    with open(file_path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'): continue
            v = [float(x) for x in s.split()]
            if len(v) > cols[-1]: M.append([v[i] for i in cols])
    if not M: raise ValueError("No data.")
    A = np.array(M)
    g = np.abs(np.gradient(A[:,2]))
    nz = np.flatnonzero(g[min_idx:] > thr)
    start = min_idx + int(nz[0]) if nz.size else min_idx
    pre = slice(max(start-100, 0), start)
    pf_i = float(np.mean(A[pre,4])) if start > 0 else float('nan')
    pe_i = float(np.mean(A[pre,5])) if start > 0 else float('nan')
    A = A[start:]

    t, ang, _, cur, pf, pe = A.T
    radius = 0.006875                                   # m
    TorqueConstant = 25.5                               # mNm/A
    GearRatio = 36
    Time = (t - t[0]) / 1000.0                          # s
    Displacement = jnp.deg2rad(-ang) * radius * 1000    # mm
    Force = (-cur) *TorqueConstant * GearRatio / radius # mN
    return Time, Displacement, Force, pf, pe, pf_i, pe_i

# ------------------------ EXACT raw loader ------------------------
def load_training_data_from_file(file_path):
    Time, Displacement, Force, pf, pe, pf_i, pe_i = readfile(file_path)
    dt = Time[1] - Time[0]
    fs = 1.0 / dt
    b, a = butter(1, 20, fs=fs, btype='low')
    Displacement = jnp.array(filtfilt(b, a, Displacement))          # mm
    Force = jnp.array(filtfilt(b, a, Force))                        # mN
    pressure_rate = 6.89476                                         # psi to kPa
    P_atm = 101.325                                                 # kPa
    pf = jnp.array(filtfilt(b, a, pf)) * pressure_rate + P_atm      # kPa
    pe = jnp.array(filtfilt(b, a, pe)) * pressure_rate + P_atm      # kPa
    pf_i = pf_i * pressure_rate + P_atm                             # kPa
    pe_i = pe_i * pressure_rate + P_atm                             # kPa
    Velocity = jnp.gradient(Displacement, dt)                       # mm/s
    y_data = jnp.stack([Displacement, Velocity, pf, pe], axis=-1)  # [T, 4]
    x0 = y_data[0]
    ts = jnp.array(Time)
    Force_array = jnp.array(Force)

    def u_fn(t):
        idx = jnp.searchsorted(ts, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(Force_array)-1)
        return Force_array[idx]
    
    # Calculate mf, me
    preload = 0.0
    dF = np.gradient(Force)
    mask = np.abs(Force) <= 1000
    idx1 = np.where(mask & (dF < 0))[0]
    idx2 = np.where(mask & (dF > 0))[0]
    x0_1 = np.mean(Displacement[idx1]) if len(idx1) > 0 else 0.0
    x0_2 = np.mean(Displacement[idx2]) if len(idx2) > 0 else 0.0
    mf, me, xeq = compute_me_mf(pf_i, pe_i, preload, x0_1/1e3, x0_2/1e3)

    return x0, ts, y_data, u_fn, mf, me, xeq, Force_array
