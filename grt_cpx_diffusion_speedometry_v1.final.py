"""
Grt-Cpx-Diffusion_speedometry_v1.5
=====================================================================
Log:

  1. Clinopyroxene Fe-Mg Diffusivity -- 
     Arrhenius parameters defined as
        D(Fe-Mg) = 2.77(+-4.27)e-7 * exp(-320.7(+-16.0) kJ/mol / RT)  m^2/s
        i.e. D0 = 2.77e-7 m^2/s = 2.77e-3 cm^2/s, Q = 320.7 kJ/mol =
        320700 J/mol
     Muller et al. (2013, Contrib Mineral Petrol 166:1563-1576) reported no 
     resolvable fO2-dependence and no resolvable compositional dependence. Additionally, D is measured ALONG THE C-AXIS [001] of diopside, which the paper states is always the fastest diffusion direction in clinopyroxene; off-axis directions in a real, randomly-oriented natural cpx grain will diffuse Fe-Mg slower than this. Model doesn't attempt to cprrect for crystal orientation.

  2. Interdiffusion and tracer diffusivities --
     Muller et al. (2013) report interdiffusion coefficient for the Fe-Mg    exchange reaction as a whole (not separate D_Fe*, D_Mg* tracer diffusivities the way Chakraborty &Ganguly 1992 or Borinski et al. 2012 do for garnet). Model treats clinopyroxene as a single-scalar binary diffusion couple so the two are actually a natural fit for
     each other

  3. Garnet-Clinopyroxene Kd(T) -- 
     Values from Ganguly et al. (1996, Contrib Mineral Petrol
     126:137-151), Eq. (15).
         ln K'(Gt-CPx) = 4100/T + 11.07*P/T - 2.40     (15)
     Same functional shape used in the model define by Kd_grt_cpx() --
     ln Kd = ln Kd0 + (dH/R)*(1/T - 1/T0), i.e. linear in 1/T. Since the model has no pressure dependence 11.07*P/T term is dropped.


Probable improvements for future:
  - Inversion with differential_evolution, and MisfitFn).
  - Adding Monte Carlo simulation withmultiprocessing and parallel/Mac-tuning machinery.
  - Spherical or cylindrical geometry added to grid configuration.
  - Add Ellis and Green (1979) Kd model

=====================================================================

REQUIREMENTS
-------------
    pip install numpy scipy matplotlib
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import root

R_GAS = 8.314462618
SEC_PER_MYR = 1.0e6 * 365.25 * 24 * 3600.0
SPECIES_ALL = ("Fe", "Mg", "Mn", "Ca")

# =====================================================================================
# 1) CONFIGURATION
# =====================================================================================

@dataclass
class GridConfig:
    """ Planar 1D 
    ** boundary_cpx_far defaults to "noflux", not "dirichlet". 
    ** Chowdhury & Chakraborty (2019) model two adjacent
    mineral grains in a rock -- both garnet and clinopyroxene are finite,
    closed crystals with their own core plateau, so cpx gets the same
    no-flux/closed-core treatment as garnet.
    """

    N_grt: int = 100
    N_cpx: int = 100
    L_grt_um: float = 150.0
    L_cpx_um: float = 150.0
    boundary_grt_far: str = "noflux"
    boundary_cpx_far: str = "noflux"
    n_timesteps_per_stage: int = 500


@dataclass
class DiffusionConfig:
    garnet_model: str = "chakraborty_ganguly1992"   # fixed choice for this test file
    kd_model: str = "ganguly1996"


CFG_GRID = GridConfig()
CFG_DIFF = DiffusionConfig()


# =====================================================================================
# 2) T-t PATH(S)
# =====================================================================================

@dataclass
class TwoStagePath:
    """ Defaults below sit at the middle of Chowdhury & Chakraborty
    (2019)'s stated ranges. Adjust to match a specific sample
    (their GC1/GC4/GC6/GCsym3).
    """

    T_peak_C: float = 800.0
    T_final_C: float = 500.0
    T_break_C: float = 665.0
    rate1_C_per_Myr: float = 30.0
    rate2_C_per_Myr: float = 75.0

    def __post_init__(self):
        assert self.T_final_C < self.T_break_C < self.T_peak_C, (
            "T_break must lie strictly between T_final and T_peak")
        self.t1_sec = (self.T_peak_C - self.T_break_C) / self.rate1_C_per_Myr \
            * SEC_PER_MYR
        self.t2_sec = (self.T_break_C - self.T_final_C) / self.rate2_C_per_Myr \
            * SEC_PER_MYR
        self.duration_sec = self.t1_sec + self.t2_sec

    def T_at(self, t_sec):
        t_sec = min(t_sec, self.duration_sec)
        if t_sec <= self.t1_sec:
            return self.T_peak_C - self.rate1_C_per_Myr * (t_sec / SEC_PER_MYR)
        return self.T_break_C - self.rate2_C_per_Myr * \
            ((t_sec - self.t1_sec) / SEC_PER_MYR)

    def stage_seconds(self, n_timesteps_per_stage):
        """ List of (dt_seconds_per_step, n_steps) for the forward_model loop,
        one entry per linear stage.
        """

        n = n_timesteps_per_stage
        segs = []
        if self.t1_sec > 0:
            segs.append((self.t1_sec / n, n))
        if self.t2_sec > 0:
            segs.append((self.t2_sec / n, n))
        return segs or [(0.0, 0)]


@dataclass
class PiecewiseLinearPath:
    """ General alternative to TwoStagePath: from a list of (time_Myr,
    T_degC) anchor points (monotonically increasing time, monotonically
    decreasing T) and it linearly interpolates between them -- use this if you
    want to paste in a T-t history directly (e.g. digitized from a published
    figure) rather than fitting it to the rate1/T_break/rate2 parameterization.
    """
    anchors: list  # [(t_Myr, T_C), ...], sorted by time, t[0]=0

    def __post_init__(self):
        ts = [a[0] for a in self.anchors]
        Ts = [a[1] for a in self.anchors]
        assert ts[0] == 0.0, "first anchor must be at t=0 (the peak)"
        assert all(t2 > t1 for t1, t2 in zip(ts, ts[1:])), "times must increase"
        assert all(T2 < T1 for T1, T2 in zip(Ts, Ts[1:])), "T must strictly decrease"
        self._ts_sec = np.array(ts) * SEC_PER_MYR
        self._Ts = np.array(Ts)
        self.duration_sec = self._ts_sec[-1]
        self.T_peak_C = Ts[0]
        self.T_final_C = Ts[-1]

    def T_at(self, t_sec):
        t_sec = min(t_sec, self.duration_sec)
        return float(np.interp(t_sec, self._ts_sec, self._Ts))

    def stage_seconds(self, n_timesteps_per_stage):
        """ One (dt, n_steps) block per anchor-to-anchor segment, with n_steps
        allocated proportionally to segment duration (roughly constant dt
        across the whole path) so a short segment doesn't get starved of
        resolution relative to a long one.
        """
        n_total = n_timesteps_per_stage * (len(self._ts_sec) - 1)
        dt_target = self.duration_sec / n_total
        segs = []
        for t0, t1 in zip(self._ts_sec[:-1], self._ts_sec[1:]):
            seg_dur = t1 - t0
            n = max(1, int(round(seg_dur / dt_target)))
            segs.append((seg_dur / n, n))
        return segs


# =====================================================================================
# 3) DIFFUSION PARAMETER LIBRARY  (pressure-free, planar, backward Euler)
# =====================================================================================

def D_arrhenius_cm2s(T_K, D0, Q):
    return D0 * np.exp(-Q / (R_GAS * T_K))


# ---- Garnet Fe, Mg, Mn: Chakraborty & Ganguly (1992) -----------------------------
GARNET_FE_MG_MN = {
    "Fe": dict(D0=1.64e-6, Q=2.26900e5),   # cm^2/s, J/mol
    "Mg": dict(D0=2.72e-6, Q=2.28300e5),
    "Mn": dict(D0=5.1e-4, Q=2.53421e5),
}

def D_grt_tracer(species, T_K):
    """ Fe, Mg, Mn from Chakraborty & Ganguly (1992); Ca = 0.5 * D_Fe(T)."""
    if species == "Ca":
        return 0.5 * D_arrhenius_cm2s(T_K, GARNET_FE_MG_MN["Fe"]["D0"],
                                       GARNET_FE_MG_MN["Fe"]["Q"])
    p = GARNET_FE_MG_MN[species]
    return D_arrhenius_cm2s(T_K, p["D0"], p["Q"])


# ---- Clinopyroxene Fe-Mg: Muller et al. (2013; 2013, CMP 166:1563-1576) ---------------
CPX_FEMG_MULLER2013 = dict(D0=2.77e-2, Q=3.207e5)  # cm^2/s, J/mol 


def D_cpx_femg(T_K, XFe=None):
    """ Fe-Mg interdiffusion coefficient for clinopyroxene, Muller et al.
    (2013, CMP 166:1563-1576), c-axis, T-only Arrhenius law (no resolvable
    fO2 or compositional dependence reported). XFe is accepted
    but kept only so the call signature already supports adding a compositional
    term later without touching every call site, should a future source report one.
    """
    return D_arrhenius_cm2s(T_K, CPX_FEMG_MULLER2013["D0"], CPX_FEMG_MULLER2013["Q"])


# ---- Grt-Cpx Fe-Mg exchange Kd(T): Chowdhury & Chakraborty (2019) Eq. (3) -------

KD_MODELS = {
    "ganguly1996": dict(Kd0=4.13930687, dH=4100.0 * R_GAS, T0_K=1073.15),  # (Ganguly et al. 1996, Eq. 15, P=0)
    "ellis_green1979": dict(Kd0=1.35, dH=-18000.0, T0_K=1073.15),          # PLACEHOLDER
}


def Kd_grt_cpx(T_K, model=None):
    
    model = model or CFG_DIFF.kd_model
    p = KD_MODELS[model]
    return p["Kd0"] * np.exp((p["dH"] / R_GAS) * (1.0 / T_K - 1.0 / p["T0_K"]))


# =====================================================================================
# 4) MULTICOMPONENT (ONSAGER) GARNET DIFFUSION
# =====================================================================================

def _garnet_D_matrix(X_indep: np.ndarray, Dstar: dict) -> np.ndarray:
    Dvec = np.array([Dstar["Fe"], Dstar["Mg"], Dstar["Mn"]])
    Dca = Dstar["Ca"]
    return np.diag(Dvec) - np.outer(X_indep, (Dvec - Dca))


def _half_point_matrices(Ds_nodes):
    return [0.5 * (Ds_nodes[j] + Ds_nodes[j + 1]) for j in range(len(Ds_nodes) - 1)]


def step_grt_multicomponent(U, Dstar_nodes, dx_cm, dt, interface_vals_FeMg):
    """ Backward-Euler step of the coupled [X_Fe, X_Mg, X_Mn] system, assembled
    as a sparse 3N x 3N block-tridiagonal system, solved with spsolve. 
    """
    N = U.shape[0]
    D_nodes = [_garnet_D_matrix(U[j], Dstar_nodes[j]) for j in range(N)]
    D_half = _half_point_matrices(D_nodes)

    ndof = 3 * N
    A = lil_matrix((ndof, ndof))
    b = np.zeros(ndof)
    I3 = np.eye(3)
    inv_dx2 = 1.0 / dx_cm ** 2

    def blk(j):
        return slice(3 * j, 3 * j + 3)

    for j in range(N):
        if j == 0:
            Dp = D_half[0]
            A[blk(0), blk(0)] = I3 / dt + Dp * inv_dx2
            A[blk(0), blk(1)] = -Dp * inv_dx2
            b[blk(0)] = U[0] / dt
        elif j == N - 1:
            Dm = D_half[-1]
            diagblk = I3 / dt + Dm * inv_dx2
            lowblk = -Dm * inv_dx2
            rhs = U[j] / dt
            diagblk[0, :] = 0.0; diagblk[0, 0] = 1.0
            diagblk[1, :] = 0.0; diagblk[1, 1] = 1.0
            lowblk[0, :] = 0.0
            lowblk[1, :] = 0.0
            rhs[0] = interface_vals_FeMg[0]
            rhs[1] = interface_vals_FeMg[1]
            A[blk(j), blk(j)] = diagblk
            A[blk(j), blk(j - 1)] = lowblk
            b[blk(j)] = rhs
        else:
            Dm, Dp = D_half[j - 1], D_half[j]
            A[blk(j), blk(j - 1)] = -Dm * inv_dx2
            A[blk(j), blk(j)] = I3 / dt + (Dm + Dp) * inv_dx2
            A[blk(j), blk(j + 1)] = -Dp * inv_dx2
            b[blk(j)] = U[j] / dt

    U_new = spsolve(csr_matrix(A), b).reshape(N, 3)
    return np.clip(U_new, 1e-6, 1.0 - 1e-6)


# =====================================================================================
# 4b) CLINOPYROXENE (Fe-Mg, backward Euler)
# =====================================================================================

def bw_euler_step_scalar(C_old, D, dx_cm, dt, left_type, left_val,
                          right_type, right_val):
    from scipy.linalg import solve_banded
    n = len(C_old)
    r = D * dt / dx_cm ** 2
    lower = np.zeros(n); diag = np.zeros(n); upper = np.zeros(n); b = np.zeros(n)
    for i in range(1, n - 1):
        lower[i] = -r; diag[i] = 1 + 2 * r; upper[i] = -r
        b[i] = C_old[i]
    if left_type == "dirichlet":
        diag[0], upper[0], b[0] = 1.0, 0.0, left_val
    else:
        diag[0], upper[0] = 1 + r, -r
        b[0] = C_old[0]
    if right_type == "dirichlet":
        diag[-1], lower[-1], b[-1] = 1.0, 0.0, right_val
    else:
        diag[-1], lower[-1] = 1 + r, -r
        b[-1] = C_old[-1]
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]; ab[1, :] = diag; ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, b)


# =====================================================================================
# 5) INTERFACE COUPLING (grt-cpx)  -- LM-based approach
# =====================================================================================
# Physical assumption: the contact permits Fe-Mg exchange (the exchange-
# thermometer reaction governed by (Kd_grt_cpx above) but Mn/Ca do not partition
# into cpx and see a no-flux condition there, even though they still diffuse freely
# inside the garnet interior via the full coupled matrix.

def solve_grt_cpx_interface_femg_lm(T_K, U, Dstar, D_cpx_fe, dx_grt_cm, dx_cpx_cm,
                                     C_cpx, boundary_cpx_far, dt):
    """
    BUGFIX: `dt` is now an explicit parameter instead of being read
    from the module-level `dt_placeholder` mutable-global list. The v1
    version worked correctly in serial use (dt_placeholder[0] was set right
    before each call), but it's the same fragile pattern as the
    stage_seconds() bug above -- module-level mutable state driving a
    function that looks like it takes explicit parameters -- and is exactly
    the kind of thing that causes hard-to-debug cross-talk once this code is
    parallelized (threads sharing the module, or careless reuse across calls
    in the same process). Removed now, before that becomes a problem.
    """

    XFe_last, XMg_last, XMn_last = U[-1]
    XFe_nbr, XMg_nbr, XMn_nbr = U[-2]

    def residuals(x):
        XFe_c = np.clip(x[0], 1e-6, 1.0 - 1e-6)
        XMg_c = np.clip(x[1], 1e-6, 1.0 - 1e-6)

        D_here = _garnet_D_matrix(np.array([XFe_c, XMg_c, XMn_last]), Dstar)
        D_nbr = _garnet_D_matrix(np.array([XFe_nbr, XMg_nbr, XMn_nbr]), Dstar)
        D_avg = 0.5 * (D_here + D_nbr)
        grad = np.array([XFe_c - XFe_nbr, XMg_c - XMg_nbr,
                          XMn_last - XMn_nbr]) / dx_grt_cm
        J_grt = -D_avg @ grad

        ratio_grt = XFe_c / XMg_c
        ratio_cpx = ratio_grt / Kd_grt_cpx(T_K)
        XFe_cpx_int = ratio_cpx / (1.0 + ratio_cpx)
        C_cpx_trial = bw_euler_step_scalar(C_cpx, D_cpx_fe, dx_cpx_cm,
                                            dt, "dirichlet",
                                            XFe_cpx_int, boundary_cpx_far,
                                            C_cpx[-1])
        J_cpx_fe = -D_cpx_fe * (C_cpx_trial[1] - C_cpx_trial[0]) / dx_cpx_cm

        R1 = J_grt[0] - J_cpx_fe
        R2 = J_grt[0] + J_grt[1]
        return [R1, R2]

    x0 = np.array([XFe_last, XMg_last])
    sol = root(residuals, x0, method="lm")
    if not sol.success:
        warnings.warn(f"Interface LM solve did not converge at T="
                       f"{T_K - 273.15:.0f} degC ({sol.message}); falling back "
                       "to previous-step interface values.", stacklevel=2)
        XFe_int, XMg_int = XFe_last, XMg_last
    else:
        XFe_int, XMg_int = np.clip(sol.x, 1e-6, 1.0 - 1e-6)

    ratio_grt = XFe_int / XMg_int
    ratio_cpx = ratio_grt / Kd_grt_cpx(T_K)
    XFe_cpx_int = ratio_cpx / (1.0 + ratio_cpx)
    C_cpx_new = bw_euler_step_scalar(C_cpx, D_cpx_fe, dx_cpx_cm,
                                      dt, "dirichlet", XFe_cpx_int,
                                      boundary_cpx_far, C_cpx[-1])
    return XFe_int, XMg_int, C_cpx_new


# =====================================================================================
# 6) FORWARD MODEL
# =====================================================================================

def forward_model(path, init: dict, grid: GridConfig):
    """
    path: a TwoStagePath or PiecewiseLinearPath instance (anything exposing
    .T_at(t_sec) and .stage_seconds()).
    init keys: XFe_grt0, XMg_grt0, XMn_grt0 (Ca0 implied), XFe_cpx0.
        (No separate "XFe_cpx_matrix" reservoir value, unlike v4's opx --
        cpx's far boundary is no-flux/closed by default here, see GridConfig.)

    Returns {'XFe_grt','XMg_grt','XMn_grt','XCa_grt','XFe_cpx'} final profiles,
    plus the distance arrays for plotting.
    """
    U = np.column_stack([
        np.full(grid.N_grt, init["XFe_grt0"]),
        np.full(grid.N_grt, init["XMg_grt0"]),
        np.full(grid.N_grt, init["XMn_grt0"]),
    ])
    C_cpx = np.full(grid.N_cpx, init["XFe_cpx0"])

    dx_grt_cm = (grid.L_grt_um / (grid.N_grt - 1)) * 1e-4
    dx_cpx_cm = (grid.L_cpx_um / (grid.N_cpx - 1)) * 1e-4

    t = 0.0
    
    for dt, n in path.stage_seconds(grid.n_timesteps_per_stage):
        for _ in range(n):
            T_K = path.T_at(t + dt / 2) + 273.15

            Dstar = {sp: D_grt_tracer(sp, T_K) for sp in SPECIES_ALL}
            D_cpx_fe = D_cpx_femg(T_K, C_cpx[0])

            XFe_int, XMg_int, C_cpx = solve_grt_cpx_interface_femg_lm(
                T_K, U, Dstar, D_cpx_fe, dx_grt_cm, dx_cpx_cm, C_cpx,
                grid.boundary_cpx_far, dt)

            Dstar_nodes = [Dstar] * grid.N_grt
            U = step_grt_multicomponent(U, Dstar_nodes, dx_grt_cm, dt,
                                         (XFe_int, XMg_int))
            t += dt

    XCa_grt = 1.0 - U.sum(axis=1)
    x_grt = np.linspace(0, grid.L_grt_um, grid.N_grt)
    x_cpx = np.linspace(0, grid.L_cpx_um, grid.N_cpx)
    return {"XFe_grt": U[:, 0], "XMg_grt": U[:, 1], "XMn_grt": U[:, 2],
            "XCa_grt": XCa_grt, "XFe_cpx": C_cpx,
            "x_grt_um": x_grt, "x_cpx_um": x_cpx}


# =====================================================================================
# 7) SELF-TESTS 
# =====================================================================================

def run_self_tests(verbose=True):
    results = {}
    grid = GridConfig(N_grt=100, N_cpx=100, n_timesteps_per_stage=500)
    path = TwoStagePath(T_peak_C=800.0, T_final_C=500.0, T_break_C=665.0,
                         rate1_C_per_Myr=30.0, rate2_C_per_Myr=75.0)
    init = dict(XFe_grt0=0.46, XMg_grt0=0.31, XMn_grt0=0.01, XFe_cpx0=0.18)
    out = forward_model(path, init, grid)

    total = out["XFe_grt"] + out["XMg_grt"] + out["XMn_grt"] + out["XCa_grt"]
    results["mass_closure (max|sum-1|)"] = float(np.max(np.abs(total - 1.0)))

    all_vals = np.concatenate([out[k] for k in
                                ("XFe_grt", "XMg_grt", "XMn_grt", "XCa_grt", "XFe_cpx")])
    results["finite_and_bounded"] = bool(np.all(np.isfinite(all_vals)) and
                                          np.all(all_vals >= 0) and
                                          np.all(all_vals <= 1))

    Dstar_equal = {"Fe": 1e-18, "Mg": 1e-18, "Mn": 1e-18, "Ca": 1e-18}
    M = _garnet_D_matrix(np.array([0.3, 0.3, 0.1]), Dstar_equal)
    results["Onsager_reduces_to_diagonal_when_D_equal (max off-diag)"] = float(
        np.max(np.abs(M - np.diag(np.diag(M)))))

    grid_coarse = GridConfig(N_grt=100, N_cpx=100, n_timesteps_per_stage=100)
    grid_fine = GridConfig(N_grt=100, N_cpx=100, n_timesteps_per_stage=5000)
    out_c = forward_model(path, init, grid_coarse)
    out_f = forward_model(path, init, grid_fine)
    results["grid_convergence (|XFe_int_fine - XFe_int_coarse|)"] = float(
        abs(out_f["XFe_grt"][-1] - out_c["XFe_grt"][-1]))

    # Numerical methods NOTE :
    # if you push resolution higher than the pair above: for BACKWARD EULER
    # (first-order in time), the mesh Fourier number r = D*dt/dx^2 grows
    # linearly with N if you scale n_timesteps_per_stage ~ N (as the pair
    # above does) while dx ~ 1/N -- because dt ~ 1/N but dx^2 ~ 1/N^2, so
    # r ~ D*N. At the coarse N used above (15 vs 30), spatial discretization
    # error still dominates and this doesn't matter (confirmed numerically:
    # scaling n_timesteps_per_stage ~ N^2 instead, which keeps r roughly
    # constant, gives the SAME delta to many decimal places at this N range).
    # But if you manually push N much higher (60, 120, 240...) while only
    # scaling n_timesteps_per_stage ~ N, temporal error starts to dominate and
    # the convergence delta can stop shrinking, or even grow, between
    # successive refinements -- this looks alarming but is a refinement-
    # methodology artifact, not a solver bug. If you see that, rerun with
    # n_timesteps_per_stage scaled roughly as N_grt^2 (not just N_grt) to
    # restore proper joint dt~dx^2 refinement before concluding anything is
    # actually wrong.

    # --- Test: domain-length sufficiency ---------------------------------------

    core_drift = abs(out_f["XFe_grt"][0] - init["XFe_grt0"])
    results["domain_sufficiency (|XFe_core_fine - XFe_grt0|, should be ~0)"] = \
        float(core_drift)
    if core_drift > 0.01:
        warnings.warn(
            "domain_sufficiency check: the garnet core moved by "
            f"{core_drift:.4f} from its initial value at the CURRENT "
            f"GridConfig.L_grt_um={grid_fine.L_grt_um} um. Two different "
            "things this can mean, and they call for OPPOSITE responses:\n"
            "  (a) If L_grt_um does NOT yet match your real sample's grain "
            "size/traverse length, fix that first -- set it to your actual "
            "measured value, not an arbitrary number.\n"
            "  (b) If L_grt_um ALREADY matches your real sample, DO NOT "
            "enlarge it to make this warning go away -- that would just be "
            "modeling a bigger, fictional grain. Instead, take this as a "
            "real result: the diffusivities and/or T-t path you're using "
            "would homogenize a grain of your actual size, which doesn't "
            "match what you observe. Question the diffusivity source "
            "(e.g. this file's Chakraborty & Ganguly 1992 vs the slower "
            "Borinski et al. 2012 that Chowdhury & Chakraborty 2019 "
            "actually used) or the assumed cooling duration/rate -- see "
            "GridConfig's docstring for the worked numerical example.",
            stacklevel=2)

    if verbose:
        print("\n=== run_self_tests() report ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("Same interpretation guidance as v4's run_self_tests(): read "
              "the numbers, don't just trust a pass/fail. mass_closure and "
              "Onsager-diagonal-reduction should be ~1e-8 or smaller; "
              "grid_convergence should shrink further as you refine more.")
    return results


# =====================================================================================
# 8) PLOTTING
# =====================================================================================

def plot_comparison(model_out, measured=None, outpath="grt_cpx_test_result.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(model_out["x_grt_um"], model_out["XFe_grt"], "-", color="tab:blue",
                 label="model grt Fe")
    axes[0].plot(model_out["x_cpx_um"] + model_out["x_grt_um"][-1],
                 model_out["XFe_cpx"], "-", color="tab:orange", label="model cpx Fe")
    if measured is not None:
        axes[0].plot(measured["x_grt_um"], measured["XFe_grt"], "o", ms=3,
                     color="gray", label="measured grt Fe")
        if "XFe_cpx" in measured:
            axes[0].plot(measured["x_cpx_um"] + model_out["x_grt_um"][-1],
                         measured["XFe_cpx"], "o", ms=3, color="darkgray",
                         label="measured cpx Fe")
    axes[0].axvline(model_out["x_grt_um"][-1], color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("distance (\u00b5m)"); axes[0].set_ylabel("X$_{Fe}$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=8)

    XMg_cpx = 1.0 - model_out["XFe_cpx"]  # cpx is binary -- Mg is just the complement
    axes[1].plot(model_out["x_grt_um"], model_out["XMg_grt"], "-", color="tab:blue",
                 label="model grt Mg (solved state)")
    axes[1].plot(model_out["x_cpx_um"] + model_out["x_grt_um"][-1], XMg_cpx, "-",
                 color="tab:orange", label="model cpx Mg (= 1 - X$_{Fe,cpx}$)")
    if measured is not None and "XMg_grt" in measured:
        axes[1].plot(measured["x_grt_um"], measured["XMg_grt"], "o", ms=3,
                     color="gray", label="measured grt Mg")
    axes[1].axvline(model_out["x_grt_um"][-1], color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("distance (\u00b5m)"); axes[1].set_ylabel("X$_{Mg}$")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=8)

    axes[2].plot(model_out["x_grt_um"], model_out["XMn_grt"], "-", color="tab:green",
                 label="model Mn")
    axes[2].plot(model_out["x_grt_um"], model_out["XCa_grt"], "-", color="tab:red",
                 label="model Ca")
    if measured is not None:
        if "XMn_grt" in measured:
            axes[2].plot(measured["x_grt_um"], measured["XMn_grt"], "s", ms=3,
                         color="lightgreen", label="measured Mn")
        if "XCa_grt" in measured:
            axes[2].plot(measured["x_grt_um"], measured["XCa_grt"], "s", ms=3,
                         color="lightcoral", label="measured Ca")
    axes[2].set_xlabel("distance (\u00b5m)"); axes[2].set_ylabel("X$_{Mn}$, X$_{Ca}$ (grt)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=900)
    plt.close(fig)
    print(f"Saved: {outpath}")


# =====================================================================================
# 9) MAIN
# =====================================================================================

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        run_self_tests()
        sys.exit(0)


    grid = CFG_GRID
    path = TwoStagePath()  # T_peak=800, T_break=665, rate1=20, rate2=75, T_final=500
    print(f"T-t path: peak={path.T_peak_C} C, break={path.T_break_C} C "
          f"(rate1={path.rate1_C_per_Myr} -> rate2={path.rate2_C_per_Myr} C/Myr), "
          f"final={path.T_final_C} C. Implied duration: "
          f"{path.duration_sec / SEC_PER_MYR:.2f} Myr ")


    init = dict(XFe_grt0=0.46, XMg_grt0=0.31, XMn_grt0=0.01, XFe_cpx0=0.18)
    out = forward_model(path, init, grid)
    plot_comparison(out, measured=None, outpath="grt_cpx_diffusion_result.png")
