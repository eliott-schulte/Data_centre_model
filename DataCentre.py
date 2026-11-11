"""
Data Centre Thermal & Performance Simulation
=============================================
Based on: Gupta et al., Applied Energy 299 (2021) 117050

Steady-state model of one pack of server racks cooled by in-row coolers (IRCs)
and a chiller, used to optimise the cooling operation (chilled-water
temperature, airflow and workload distribution) for COP, CPR or η_ex.

HOW TO RUN
    Edit the USER SETTINGS at the bottom of this file (section 11, inside
    `if __name__ == "__main__":`) and run the file.  Nothing else needs to be
    changed.  CODE_GUIDE.md explains every option, setting, function and
    output file.

Simplifications applied (vs. full model):
  • Steady-state only (no thermal capacitance / time-marching)
"""

from __future__ import annotations

import os
import time
import warnings
import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, brentq
from scipy.optimize import root
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, basinhopping
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – registers 3-D projection
from pymoo.core.problem import Problem as MooProblem
from pymoo.core.repair  import Repair  as MooRepair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as moo_minimize


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DCParams:
    """
    All physical and model constants for the simplified DC model.
    Frozen so no function can accidentally mutate a constant mid-run.
    """

    # ── Geometry & counts ─────────────────────────────────────────────────
    N_s:   int   = 3               # Servers per zone
    N_z:   int   = 5               # Zones per rack
    N_r:   int   = 10              # Racks
    N_cu:  int   = 3               # IRC cooling units
    is_irc: np.ndarray = field(default_factory=lambda: np.array([True, False, False, False, False, False, True, False, False, False, False, False, True]))  # position of the IRC units
    def __post_init__(self):
        # We must use post_init to define is_rack because it depends on is_irc
        object.__setattr__(self, 'is_rack', ~self.is_irc)

    # ── Fluid properties ──────────────────────────────────────────────────
    rho_a: float = 1.18            # Air density             [kg m⁻³]
    Cp_a:  float = 1006.0          # Air specific heat       [J kg⁻¹ K⁻¹]
    rho_w: float = 997.0           # Water density           [kg m⁻³]
    Cp_w:  float = 4180.0          # Water specific heat     [J kg⁻¹ K⁻¹]

    # ── Flow resistances ──────────────────────────────────────────────────
    R_fH:  float = 6.0             # Horizontal flow resistance for the front chamber
    R_bH:  float = 31.0            # Horizontal flow resistance for the back chamber
    R_fV:  float = 6.0             # Vertical flow resistance for the front chamber
    R_bV:  float = 31.0            # Vertical flow resistance for the back chamber
    R_s:   float = 2415.3          # Flow resistance inside a passive server
    R_br:  float = 3977.7          # Flow resistance of the brushes

    # ── Heat exchanger ────────────────────────────────────────────────────
    UA_ref_per_unit:float = 2070.0              # Nominal UA per Rittal 30kW IRC                [W K⁻¹]
    Q_a_ref_per_unit:float= 1.389               # Nominal airflow per IRC unit (5000 m³/h)      [m³ s⁻¹]
    #Q_w:            float = 15.0 * 6.309e-5     # Water flow (15 GPM) for Trane CGA120         [m³ s⁻¹] --> must be between 12 and 36 GPM
    Q_w_min:        float = 30.0 * 6.309e-5     # Minimal evaporator flow rate for Trane CGAM20 [m³ s⁻¹]
    Q_w_max:        float = 69.0 * 6.309e-5     # Maximal evaporator flow rate for Trane CGAM20 [m³ s⁻¹]
    Q_w:            float = 40.0 * 6.309e-5     # [m³ s⁻¹]

    # ── Environment & fans ────────────────────────────────────────────────
    T_amb:   float = 25.0 + 273.15 # Ambient temperature    [K]
    eta_fan: float = 0.604         # Fan efficiency         [-]

    # ── Chiller condenser fan flow ─────────────────────────────────────────
    # Q_cf: float = 8120 * 0.000471947  # Condenser fan total flow for Trane CGA120 [m³ s⁻¹]
    Q_cf: float = 2 * 9413 * 0.000471947  # Condenser fan total flow for Trane CGAM20 [m³ s⁻¹]

    # ── Server power model (HP ProLiant DL360 G5) — Eq. 11 ───────────────
    A1: float = 223.4              # Idle power per server  [W]
    A2: float = 154.5              # Dynamic power slope    [W]

    # ── Fan power model (RITTAL IRC) — Eq. 13 ────────────────────────────
    B1: float =  480.0
    B2: float = -3073.0
    B3: float =  6031.0

    # ── Gordon-Ng chiller model — Eq. 14 (re-fitted for scaled system) ───
    # NOTE: Paper reports C1=0.0092, C2=8.953, C3=0.649 for TRANE CGA120.
    C1: float = 0.0
    C2: float = 38.1422
    C3: float = 0.8439

    # ── Chiller model based on EIR-fPLR method (EnergyPlus) ───
    # General data
    Q_ch_ref:   float = 68930.0     # reference chiller cooling capacity    [W]
    Q_ch_cap:   float | None = None # optional hard upper limit on the available cooling capacity [W]:
                                    # Q_ch,max = min(Q_ch_ref × CAPFT, Q_ch_cap).  None = no cap:
                                    # the capacity follows CAPFT, which matches the datasheet as
                                    # long as T_cw stays within its rated range (≤ 10 °C).
    COP_ref:    float = 2.96        # reference chiller COP                 [-]
    T_cw_min:   float = 5.56        # minimum chilled water temperature     [°C]
    T_cw_max:   float = 10.0        # maximum chilled water temperature     [°C]
    T_amb_min:  float = 4.44        # minimum ambient air temperature       [°C]
    T_amb_max:  float = 46.11       # maximum ambient air temperature       [°C]
    PLR_min:    float = 0.25        # minimum part load ratio               [-]
    PLR_max:    float = 1.0         # maximum part load ratio               [-]

    # ── Multi-pack configuration ───────────────────────────────────────────
    n_packs: int  = 1               # 1 = single pack,  2 = double pack (symmetric twin)
                                    # The physics simulation runs on ONE pack only.
                                    # KPIs (COP, η_ex) are then re-derived for the full DC
                                    # where n_packs identical packs share a SINGLE chiller:
                                    #   PLR_total = n_packs × PLR_single_pack
                                    # CPR is a spatial ratio → identical for each pack.

    # ── Thermal limit (ASHRAE A2 class) ──────────────────────────────────────
    T_server_red: float = 27.0      # [°C] maximum allowed server inlet temperature
                                    # (soft penalty on max T_f in run_simulation)
    T_server_min: float | None = 18.0   # [°C] minimum allowed server inlet temperature
                                    # (ASHRAE recommended range 18–27 °C): soft penalty on
                                    # min T_f over the rack zones in run_simulation, so that
                                    # CPR / η_ex cannot push the inlet air arbitrarily cold.
                                    # None = no lower limit
    T_server_min_for: tuple = ("CPR", "eta_ex")   # objectives it applies to.  Not COP by
                                    # default: it would bind in cold months at high LF
                                    # (e.g. LF 0.8, T_amb −4 °C: min T_f ≈ 16.7 °C at the COP
                                    # optimum) and change the climate-sweep results

    # Coefficients for the Cooling Capacity Function of Temperature Curve (CAPFT)
    cc1: float = 1.0020652868       # constant coefficient
    cc2: float = 0.0290829382       # coefficient for T_cw
    cc3: float = 0.0004981323       # coefficient for T_cw^2
    cc4: float = 0.0009088235       # coefficient for T_amb
    cc5: float = -0.0001680612      # coefficient for T_amb^2
    cc6: float = -0.0001752731      # coefficient for T_cw*T_amb

    # Coefficients for the Electric Input to Cooling Capacity Ratio Function of Temperature Curve (EIRFT)
    dd1: float = 0.609429307        # constant coefficient
    dd2: float = -0.0010745365      # coefficient for T_cw
    dd3: float = -0.0000580474      # coefficient for T_cw^2
    dd4: float = -0.0003768051      # coefficient for T_amb
    dd5: float = 0.0004576827       # coefficient for T_amb^2
    dd6: float = -0.0005942491      # coefficient for T_cw*T_amb

    # Coefficients for the Electric Input to Cooling Capacity Ratio Function of Part Load Ratio (EIRFPLR)
    ee1: float = 0.0546             # constant coefficient
    ee2: float = 0.7604             # coefficient for PLR
    ee3: float = 0.1871             # coefficient for PLR^2

    # ── CPU temperature regression — Eq. 25 ──────────────────────────────
    E1: float =  20.2
    E2: float =  59.1
    E3: float =   0.76
    E4: float =  -8.1
    E5: float =  -0.56

    # ── CPR linear model — Eq. 26 ─────────────────────────────────────────
    F1: float =  0.75
    F2: float = -1.72e-3


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA CENTRE STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DCState:
    """
    Holds the 2D matrices of the state of the entire 7x5 data centre grid.
    There are 5x5 zones in front of the servers, and 2x5 zones in front of 
    the cooling units. 
    """

    # ── Server utilisation ────────────────────────────────────────────────
    u:      np.ndarray  = field(default_factory=lambda: np.zeros((10,5)))

    # ── Pressures ─────────────────────────────────────────────────────────
    p_f:    np.ndarray  = field(default_factory=lambda: np.zeros((13,5)))         # Pressure inside the front chambers         [Pa]
    p_b:    np.ndarray  = field(default_factory=lambda: np.zeros((13,5)))         # Pressure inside the back chambers          [Pa]

    # ── Temperatures ──────────────────────────────────────────────────────
    T_f:    np.ndarray  = field(default_factory=lambda: np.zeros((13,5)))         # Temperature inside the front chambers      [K]
    T_b:    np.ndarray  = field(default_factory=lambda: np.zeros((13,5)))         # Temperature inside the back chambers       [K]
    T_e:    np.ndarray  = field(default_factory=lambda: np.zeros((10,5)))                # Temperature at the servers exhaust         [K]
    T_ha:   float       = 0.0
    T_ca:   float       = 0.0
    T_hw:   float       = 0.0
    T_cw:   float       = 0.0
    # T_cw:   float # uncomment if you want T_cw to be inside the data frame

    # ── Airflow rates ─────────────────────────────────────────────────────
    Q_z:    np.ndarray   = field(default_factory=lambda: np.zeros((10,5)))   # Airflow rate inside each servers' zone    [m³/s]
    Q_a:    float        = 0.0                                                      # Total airflow rate inside the DC          [m³/s]

    # ── Powers ────────────────────────────────────────────────────────────
    P_z:    np.ndarray   = field(default_factory=lambda: np.zeros((10,5)))   # Power consumption of a zone's servers     [W]
    P_ch:   float        = 0.0                                                      # Power consumption of the chiller          [W]
    P_f:    float        = 0.0                                                      # Power consumption of the fans             [W]
    P_cool: float        = 0.0                                                      # Total cooling power consumption           [W]
    
    # ── Performances ──────────────────────────────────────────────────────
    COP:    float        = 0.0                                                      # Coefficient of performance                [-]
    CPR:    float        = 0.0                                                      # Computing performance ratio               [-]
    eta_ex: float        = 0.0                                                      # Exergy efficiency                         [-]
    PLR:    float        = 0.0                                                      # Chiller part-load ratio (unclamped)       [-]
    Q_coil: float        = 0.0                                                      # Cooling load on the chiller (all packs)  [W]
    Q_ch_max: float      = 0.0                                                      # Available chiller cooling capacity       [W]

    # ── Exergy destruction (all packs) ────────────────────────────────────
    psi_ch: float        = 0.0                                                      # in the chiller            (Eq. 17)        [W]
    psi_hx: float        = 0.0                                                      # in the IRC heat exchangers (Eq. 18)       [W]
    psi_f:  float        = 0.0                                                      # in the fans               (Eq. 19)        [W]
    psi_in: float        = 0.0                                                      # exergy input to the cooling cycle (Eq. 21) [W]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PHYSICS FUNCTIONS  (all pure — no globals)
# ─────────────────────────────────────────────────────────────────────────────

def pressure_residuals(P_1D: np.ndarray, p: DCParams, s: DCState) -> np.ndarray:
    """
    Calculates the mass balance error at every node using Vectorized Net Flows.
    """
    Q_z = s.Q_z
    Q_irc_total = s.Q_a
    total_cols = p.N_r + p.N_cu

    # 1. Reshape the 1D arrays back into 2D grids
    P_f = P_1D[:(total_cols * p.N_z)].reshape((total_cols, p.N_z))
    P_b = P_1D[(total_cols * p.N_z):].reshape((total_cols, p.N_z))
    
    res_f = np.zeros((total_cols, p.N_z))
    res_b = np.zeros((total_cols, p.N_z))
    
    # 2. CROSS LEAKAGE (Flow from Front to Back)
    Q_leak = (P_f - P_b) / p.R_br
    
    # 3. HORIZONTAL FLOWS (Flow from Left to Right)
    # Calculate flow crossing the boundary between col i and col i+1
    Q_fH_cross = (P_f[:-1, :] - P_f[1:, :]) / p.R_fH
    Q_bH_cross = (P_b[:-1, :] - P_b[1:, :]) / p.R_bH
    
    # Pad with 0s on the left and right (the physical walls of the data center)
    Q_fH_pad = np.pad(Q_fH_cross, pad_width=((1, 1), (0, 0)), mode='constant')
    Q_bH_pad = np.pad(Q_bH_cross, pad_width=((1, 1), (0, 0)), mode='constant')
    
    # Net horizontal flow ENTERING each node (Flow in from left - Flow out to right)
    Net_Q_fH = Q_fH_pad[:-1, :] - Q_fH_pad[1:, :]
    Net_Q_bH = Q_bH_pad[:-1, :] - Q_bH_pad[1:, :]
    
    # 4. VERTICAL FLOWS (Flow from Bottom to Top)
    Q_fV_cross = (P_f[:, :-1] - P_f[:, 1:]) / p.R_fV
    Q_bV_cross = (P_b[:, :-1] - P_b[:, 1:]) / p.R_bV
    
    # Pad with 0s on the bottom and top (Floor and Ceiling)
    Q_fV_pad = np.pad(Q_fV_cross, pad_width=((0, 0), (1, 1)), mode='constant')
    Q_bV_pad = np.pad(Q_bV_cross, pad_width=((0, 0), (1, 1)), mode='constant')
    
    # Net vertical flow ENTERING each node (Flow in from below - Flow out to above)
    Net_Q_fV = Q_fV_pad[:, :-1] - Q_fV_pad[:, 1:]
    Net_Q_bV = Q_bV_pad[:, :-1] - Q_bV_pad[:, 1:]

    # ---------------------------------------------------------
    # APPLY MASS BALANCES USING BOOLEAN MASKS
    # ---------------------------------------------------------
    
    # --- RACKS ---
    # Front: Net Horizontal In + Net Vertical In = Server Intake Out + Leakage Out
    res_f[p.is_rack, :] = Net_Q_fH[p.is_rack, :] + Net_Q_fV[p.is_rack, :] - Q_z[p.is_rack, :] - Q_leak[p.is_rack, :]
    
    # Back: Net Horizontal In + Net Vertical In + Server Exhaust In + Leakage In = 0
    res_b[p.is_rack, :] = Net_Q_bH[p.is_rack, :] + Net_Q_bV[p.is_rack, :] + Q_z[p.is_rack, :] + Q_leak[p.is_rack, :]

    # --- IN-ROW COOLERS ---
    Q_irc_single = Q_irc_total / p.N_cu
    
    # Front Chamber (Bottom Zone j=0 injects the total air into the room)
    # Sum of Net Horizontal Flow will naturally handle left, right, or center directions!
    res_f[p.is_irc, 0] = Q_irc_single + np.sum(Net_Q_fH[p.is_irc, :] - Q_leak[p.is_irc, :], axis=1)
    
    # Equalize pressure vertically (The "Open Grille" assumption)
    res_f[p.is_irc, 1:] = P_f[p.is_irc, 1:] - P_f[p.is_irc, 0:1]
    
    # Back Chamber (Extracts air from the room)
    res_b[p.is_irc, 0] = -Q_irc_single + np.sum(Net_Q_bH[p.is_irc, :] + Q_leak[p.is_irc, :], axis=1)
    res_b[p.is_irc, 1:] = P_b[p.is_irc, 1:] - P_b[p.is_irc, 0:1]

    # 5. THE REFERENCE NODE FIX
    # We force the top-right back chamber (last column, last row) to be exactly 0 Pa.
    # This anchors the floating matrix.
    #res_b[-1, -1] = P_b[-1, -1] - 0.0  

    # 6. RE-FLATTEN AND RETURN
    return np.concatenate([res_f.flatten(), res_b.flatten()])

def pressure_network(p:  DCParams, s:  DCState) -> list[float]:
    T_f_racks = s.T_f[p.is_rack, :]
    Q_z_racks = np.zeros((p.N_r, p.N_z))

    mask_low = T_f_racks <= 25+273.15
    Q_z_racks[mask_low] = p.N_s*0.01415
    mask_mid = (T_f_racks > 25+273.15) & (T_f_racks < 35+273.15)
    Q_z_racks[mask_mid] = p.N_s*(0.01415 + (T_f_racks[mask_mid] - (25+273.15))*0.00142)
    mask_high = T_f_racks >= 35+273.15
    Q_z_racks[mask_high] = p.N_s * (0.01415 + 10 * 0.00142)

    s.Q_z[p.is_rack, :] = Q_z_racks

    # Setup initial guess
    total_cols = p.N_r + p.N_cu
    P_guess_1D = np.zeros(total_cols * p.N_z * 2)
    
    # Call the SciPy Solver
    # 'root' with method 'hybr' is highly optimized for this exact type of physics problem
    # maxfev cap prevents infinite loops when Q_a is physically infeasible (supply < demand).
    # 20×n gives ample budget for well-conditioned cases while failing fast otherwise.
    n_eq = total_cols * p.N_z * 2
    solution = root(pressure_residuals, P_guess_1D, args=(p, s), method='hybr',
                    options={'maxfev': 20 * n_eq})

    if not solution.success:
        pass   # caller (run_simulation) handles convergence failures
        
    # 4. Unpack the solved pressures back into your state object!
    P_sol_1D = solution.x
    s.p_f = P_sol_1D[:(total_cols*p.N_z)].reshape((total_cols, p.N_z))
    s.p_b = P_sol_1D[(total_cols*p.N_z):].reshape((total_cols, p.N_z))


def temperature_residuals(T_1D: np.ndarray, s: DCState, p: DCParams) -> np.ndarray:
    """
    Calculates the energy balance error at every node. 
    fsolve/root will try to drive all returned values to 0.
    """
    P_f = s.p_f
    P_b = s.p_b
    Q_z = s.Q_z
    P_it_zonal = s.P_z
    T_ca = s.T_ca
    Q_irc_total = s.Q_a
    total_cols = p.N_r + p.N_cu

    T_f = T_1D[:(total_cols * p.N_z)].reshape((total_cols, p.N_z))     # front chamber temperatures [K]
    T_b = T_1D[(total_cols * p.N_z):(2 * total_cols * p.N_z)].reshape((total_cols, p.N_z))   # back chamber temperatures [K]
    T_e = T_1D[(2 * total_cols * p.N_z):].reshape((p.N_r, p.N_z))      # servers exhaust temperatures [K]
    
    res_f = np.zeros((total_cols, p.N_z))            # residuals for the front chambers
    res_b = np.zeros((total_cols, p.N_z))            # residuals for the back chambers
    res_e = np.zeros((p.N_r, p.N_z))             # residuals for the servers exhaust
    
    # Pressure difference between adjacent columns
    Delta_P_fV = P_f[:, :-1] - P_f[:, 1:] 
    Delta_P_bV = P_b[:, :-1] - P_b[:, 1:]
    Delta_P_fH = P_f[:-1, :] - P_f[1:, :] 
    Delta_P_bH = P_b[:-1, :] - P_b[1:, :]
    Delta_P_cross = P_f - P_b

    # Temperature of the donor zones
    T_donor_fH = np.where(Delta_P_fH > 0, T_f[:-1, :], T_f[1:, :])
    T_donor_bH = np.where(Delta_P_bH > 0, T_b[:-1, :], T_b[1:, :])
    T_donor_fV = np.where(Delta_P_fV > 0, T_f[:, :-1], T_f[:, 1:])
    T_donor_bV = np.where(Delta_P_bV > 0, T_b[:, :-1], T_b[:, 1:])
    T_donor_leak = np.where(Delta_P_cross > 0, T_f, T_b)

    # Heat flows (no rho, C_pa because they cancel out in the equations)
    Phi_fH = Delta_P_fH / p.R_fH * T_donor_fH
    Phi_bH = Delta_P_bH / p.R_bH * T_donor_bH
    Phi_fV = Delta_P_fV / p.R_fV * T_donor_fV
    Phi_bV = Delta_P_bV / p.R_bV * T_donor_bV
    Phi_leak = Delta_P_cross / p.R_br * T_donor_leak

    # Padding the enthalpy matrices --> we fix the heat flows from the extremities = 0 
    # Extremities : roof, ground, left of left IRC unit, right of right IRC unit
    Phi_fH_pad = np.pad(Phi_fH, pad_width=((1, 1), (0, 0)), mode='constant')
    Phi_bH_pad = np.pad(Phi_bH, pad_width=((1, 1), (0, 0)), mode='constant')
    Phi_fV_pad = np.pad(Phi_fV, pad_width=((0, 0), (1, 1)), mode='constant')
    Phi_bV_pad = np.pad(Phi_bV, pad_width=((0, 0), (1, 1)), mode='constant')

    # Net Enthalpy entering any given node
    Net_Phi_fH = Phi_fH_pad[:-1, :] - Phi_fH_pad[1:, :]
    Net_Phi_bH = Phi_bH_pad[:-1, :] - Phi_bH_pad[1:, :]
    Net_Phi_fV = Phi_fV_pad[:, :-1] - Phi_fV_pad[:, 1:]
    Net_Phi_bV = Phi_bV_pad[:, :-1] - Phi_bV_pad[:, 1:]
    
    # --- RACKS (Columns 1 to 5) ---
    # Front Chamber: Gains Net Horizontal, Net Vertical. Loses Leakage, Loses Server Flow (at T_f)
    res_f[p.is_rack, :] = Net_Phi_fH[p.is_rack, :] + Net_Phi_fV[p.is_rack, :] - Phi_leak[p.is_rack, :] - (Q_z[p.is_rack, :] * T_f[p.is_rack, :])
    
    # Back Chamber: Gains Net Horizontal, Net Vertical, Leakage, and Server EXHAUST Flow (at T_e)
    res_b[p.is_rack, :] = Net_Phi_bH[p.is_rack, :] + Net_Phi_bV[p.is_rack, :] + Phi_leak[p.is_rack, :] + (Q_z[p.is_rack, :] * T_e)
    
    # Server Chassis: Exhaust Temp = Intake Temp + IT Power Heating
    res_e = T_e - (T_f[p.is_rack, :] + P_it_zonal[:, :] / (p.rho_a * p.Cp_a * Q_z[p.is_rack, :]))

    # --- IN-ROW COOLERS (Columns 0 and 6) ---
    Q_leak = Delta_P_cross / p.R_br

    # Front Chamber: the coil supplies air at T_ca, which mixes with any air
    # entering the node (e.g. hot air leaking back→front through the brushes).
    # Supply to each node from its mass balance: Q_sup + Net_Q_fH + Net_Q_fV − Q_leak = 0
    Q_fH_pad = np.pad(Delta_P_fH / p.R_fH, pad_width=((1, 1), (0, 0)), mode='constant')
    Q_fV_pad = np.pad(Delta_P_fV / p.R_fV, pad_width=((0, 0), (1, 1)), mode='constant')
    Net_Q_f  = (Q_fH_pad[:-1, :] - Q_fH_pad[1:, :]) + (Q_fV_pad[:, :-1] - Q_fV_pad[:, 1:])
    Q_sup    = Q_leak - Net_Q_f
    res_f[p.is_irc, :] = (Q_sup[p.is_irc, :] * T_ca + Net_Phi_fH[p.is_irc, :]
                          + Net_Phi_fV[p.is_irc, :] - Phi_leak[p.is_irc, :])

    # Back Chamber: Calculate the EXACT volumetric flow entering each node to guarantee mass balance
    Q_bH = Delta_P_bH / p.R_bH
    Q_bV = Delta_P_bV / p.R_bV
    
    Q_bH_pad = np.pad(Q_bH, pad_width=((1, 1), (0, 0)), mode='constant')
    Q_bV_pad = np.pad(Q_bV, pad_width=((0, 0), (1, 1)), mode='constant')
    
    Net_Q_bH = Q_bH_pad[:-1, :] - Q_bH_pad[1:, :]
    Net_Q_bV = Q_bV_pad[:, :-1] - Q_bV_pad[:, 1:]
    
    # Total actual air entering the Back Chamber nodes
    Q_in_b = Net_Q_bH + Net_Q_bV + Q_leak

    res_b[p.is_irc, :] = Net_Phi_bH[p.is_irc, :] + Net_Phi_bV[p.is_irc, :] + Phi_leak[p.is_irc, :] - (Q_in_b[p.is_irc, :] * T_b[p.is_irc, :])

    return np.concatenate([res_f.flatten(), res_b.flatten(), res_e.flatten()])

def temperature_network(p:  DCParams, s:  DCState) -> list[float]:
    # Setup initial guess (90 elements, all 0 K)
    total_cols = p.N_r + p.N_cu
    total_nodes = (total_cols * p.N_z * 2) + (p.N_r * p.N_z)
    # Warm start for finding the solution when T_b has not been calculated yet
    if s.T_b.max() < 100.0:          # 100 K is safely below any physical temperature
        T_guess_1D = np.full(total_nodes, p.T_amb)
    else:
        T_guess_1D = np.concatenate([
            s.T_f.flatten(),
            s.T_b.flatten(),
            s.T_e.flatten()
        ])
    
    # Call the SciPy Solver
    # 'root' with method 'hybr' is highly optimized for this exact type of physics problem
    # maxfev cap prevents infinite loops when Q_a is physically infeasible (supply < demand)
    n_eq = total_cols * p.N_z * 2 + p.N_r * p.N_z
    solution = root(temperature_residuals, T_guess_1D, args=(s, p), method='hybr',
                    options={'maxfev': 20 * n_eq})

    if not solution.success:
        pass   # caller (run_simulation) handles convergence failures
        
    # 4. Unpack the solved pressures back into your state object!
    T_sol_1D = solution.x
    s.T_f = T_sol_1D[:(total_cols*p.N_z)].reshape((total_cols, p.N_z))
    s.T_b = T_sol_1D[(total_cols*p.N_z):(2*total_cols*p.N_z)].reshape((total_cols, p.N_z))
    s.T_e = T_sol_1D[(2*total_cols*p.N_z):].reshape((p.N_r, p.N_z))


def calc_T_ha(s: DCState, p: DCParams) -> float:
    """
    Flow-weighted (mixed) temperature of the air extracted by the IRC units:
    T_ha = Σ Q_in,k · T_b,k / Σ Q_in,k  over the IRC back-chamber nodes k,
    with Q_in,k the net airflow entering node k (same expression as in
    temperature_residuals).  Falls back to the plain mean if Σ Q_in ≤ 0.
    """
    Q_bH = np.pad((s.p_b[:-1, :] - s.p_b[1:, :]) / p.R_bH, pad_width=((1, 1), (0, 0)), mode='constant')
    Q_bV = np.pad((s.p_b[:, :-1] - s.p_b[:, 1:]) / p.R_bV, pad_width=((0, 0), (1, 1)), mode='constant')
    Q_leak = (s.p_f - s.p_b) / p.R_br
    Q_in = ((Q_bH[:-1, :] - Q_bH[1:, :]) + (Q_bV[:, :-1] - Q_bV[:, 1:]) + Q_leak)[p.is_irc, :]
    T_b  = s.T_b[p.is_irc, :]
    Q_sum = np.sum(Q_in)
    return float(np.sum(Q_in * T_b) / Q_sum) if Q_sum > 0 else float(np.mean(T_b))


def chiller_residuals(vars: list[float], s: DCState, p: DCParams) -> list[float]:
    """
    Steady-state energy balance for the simplified single-zone DC (Eqs. 7–8).

    Solves for [T_ca, T_hw] in °C.

    Parameters
    ----------
    vars : [T_ca, T_hw]  current estimate  [°C]
    s    : data centre state values
    p    : DCParams instance

    Returns
    -------
    Residuals [eq8, eq7]  (all zero at the solution)
    """
    T_ca, T_hw = vars

    mCp_a = p.rho_a * p.Cp_a * s.Q_a          # air-side  capacity rate [W K⁻¹]
    mCp_w = p.rho_w * p.Cp_w * p.Q_w          # water-side capacity rate [W K⁻¹]

    # --- Dynamic UA calculation ---
    flow_ratio = s.Q_a / (p.N_cu * p.Q_a_ref_per_unit)
    exponent = 0.8                            # Dittus-Boelter coefficient [84f]
    R_tot_ratio = 0.05 + 0.15 + (0.80 / (flow_ratio**exponent)) # total heat resistance
    UA_tot = p.N_cu * p.UA_ref_per_unit / R_tot_ratio

    # AMTD-based coil heat transfer
    Q_coil = (UA_tot / 2.0) * (s.T_ha + T_ca - s.T_cw - T_hw)

    # Eq. 8 — air-side HX balance
    eq8 = Q_coil - mCp_a * (s.T_ha - T_ca)

    # Eq. 7 — water-side HX balance
    eq7 = Q_coil - mCp_w * (T_hw - s.T_cw)

    return [eq8, eq7]


def calc_Pch(s: DCState, p: DCParams) -> float:
    """
    Gordon-Ng chiller power model (Eq. 14).

    Parameters
    ----------
    s      : DCState instance
    p      : DCParams instance

    Returns
    -------
    Chiller electrical power               [W]
    """
    Q_coil = p.rho_w * p.Cp_w * p.Q_w * (s.T_hw - s.T_cw)
    Q_ch_kW = Q_coil / 1000.0

    P_ch_kW = (
        p.T_amb * Q_ch_kW
        + p.C1 * s.T_cw * p.T_amb
        + p.C2 * (p.T_amb - s.T_cw)
    ) / (s.T_cw - p.C3 * Q_ch_kW) - Q_ch_kW

    # Gordon-Ng has no inherent capacity limit, so set PLR to a clamped value
    # that will not trigger the PLR penalty in run_simulation.
    # (The penalty was designed for EIR-fPLR's polynomial validity range.)
    s.PLR = float(np.clip(Q_ch_kW / 35.0, p.PLR_min, p.PLR_max))

    s.P_ch = P_ch_kW * 1000.0   # [W]

    return s.P_ch

def chiller_full_load(p: DCParams, T_cw_C: float, T_amb_C: float) -> tuple:
    """
    EIR-fPLR chiller at full load for the current temperatures.

    Returns
    -------
    (Q_ch_max, P_ch_full_load)   available cooling capacity and electrical
                                 power at that capacity                [W]

    Q_ch_max = Q_ch_ref × CAPFT, limited to p.Q_ch_cap.  The full-load power is
    tied to the available capacity, P_full = Q_ch_max × EIRFT / COP_ref, so the
    full-load COP (= COP_ref / EIRFT) is unchanged by the cap; without the cap
    this equals the original (Q_ch_ref / COP_ref) × CAPFT × EIRFT.
    """
    T_cw, T_amb = T_cw_C, T_amb_C

    # Influence of temperature on maximum cooling capacity
    CAPFT = (p.cc1 + p.cc2*T_cw + p.cc3*T_cw**2 +
                p.cc4*T_amb + p.cc5*T_amb**2 +
                p.cc6*T_cw*T_amb)

    # Influence of temperature on chiller power
    EIRFT = (p.dd1 + p.dd2*T_cw + p.dd3*T_cw**2 +
                p.dd4*T_amb + p.dd5*T_amb**2 +
                p.dd6*T_cw*T_amb)

    Q_ch_max = p.Q_ch_ref * CAPFT
    if p.Q_ch_cap is not None:
        Q_ch_max = min(Q_ch_max, p.Q_ch_cap)

    P_ch_full_load = Q_ch_max * EIRFT / p.COP_ref
    return Q_ch_max, P_ch_full_load


def calc_Pch_EIRfPLR(s: DCState, p: DCParams) -> float:
    """
    Chiller performance calculation based on the EIR-fPLR method

    Parameters
    ----------
    s      : DCState instance
    p      : DCParams instance

    Returns
    -------
    Chiller electrical power               [W]
    """
    Q_ch_max_current, P_ch_full_load = chiller_full_load(p, s.T_cw - 273.15,
                                                         p.T_amb - 273.15)

    # Cooling load required by the servers
    Q_coil = p.rho_w * p.Cp_w * p.Q_w * (s.T_hw - s.T_cw)
    PLR = Q_coil / Q_ch_max_current
    s.Q_coil   = float(Q_coil)
    s.Q_ch_max = float(Q_ch_max_current)

    s.PLR = float(PLR)                                          # store actual (unclamped) PLR for penalty
    PLR_clamped = float(np.clip(PLR, p.PLR_min, p.PLR_max))     # clamp for EIRFPLR evaluation

    # Part Load Penalty
    EIRFPLR = p.ee1 + p.ee2*PLR_clamped + p.ee3*PLR_clamped**2
    
    s.P_ch = P_ch_full_load * EIRFPLR

    return s.P_ch


def calc_Pf(s: DCState, p: DCParams) -> float:
    """
    IRC fan power (Eq. 13).

    Parameters
    ----------
    s   : DCState instance
    p   : DCParams instance

    Returns
    -------
    Total fan electrical power  [W]
    """
    q_per_unit = s.Q_a / p.N_cu
    s.P_f = p.N_cu * (p.B1 + p.B2 * q_per_unit + p.B3 * q_per_unit ** 2)

    return s.P_f


def calc_exergy(s: DCState, p: DCParams) -> float:
    """
    Exergy efficiency of the DC cooling cycle (Eqs. 16–21).

    Returns
    -------
    ηex  [%], clipped to [0, 100]
    """
    Q_coil = p.rho_w * p.Cp_w * p.Q_w * (s.T_hw - s.T_cw)   # heat pulled out of the indoor water (heat coming from the servers)

    # Guard: all log arguments must be strictly positive.
    nan = float('nan')
    if s.T_hw <= 0 or s.T_cw <= 0 or s.T_ha <= 0 or s.T_ca <= 0:
        s.eta_ex = s.psi_ch = s.psi_hx = s.psi_f = s.psi_in = nan
        return s.eta_ex

    # Hot air exhausted through chiller condenser  (Eq. 17)
    T_hamb = p.T_amb + (Q_coil + s.P_ch) / (p.rho_a * p.Cp_a * p.Q_cf)

    if T_hamb <= 0:
        s.eta_ex = s.psi_ch = s.psi_hx = s.psi_f = s.psi_in = nan
        return s.eta_ex

    # Exergy destruction in chiller  (Eq. 17)
    psi_ch = (
        p.rho_w * p.Cp_w * p.Q_w * p.T_amb * np.log(s.T_cw / s.T_hw)
        + p.rho_a * p.Cp_a * p.Q_cf * p.T_amb * np.log(T_hamb / p.T_amb)
    )

    # Exergy destruction in IRC heat exchangers  (Eq. 18)
    psi_hx = (
        p.rho_a * p.Cp_a * s.Q_a
        * (s.T_ha - s.T_ca - p.T_amb * np.log(s.T_ha / s.T_ca))
        + p.rho_w * p.Cp_w * p.Q_w
        * (s.T_cw - s.T_hw - p.T_amb * np.log(s.T_cw / s.T_hw))
    )

    # Exergy destruction in fans  (Eq. 19)
    psi_f = (1.0 - p.eta_fan) * s.P_f

    psi_cool = psi_ch + psi_hx + psi_f

    # Exergy input to cooling cycle  (Eq. 21)
    psi_in = (
        p.rho_a * p.Cp_a * s.Q_a
        * (s.T_ha - p.T_amb - p.T_amb * np.log(s.T_ha / p.T_amb))
        + s.P_f + s.P_ch
    )

    s.psi_ch, s.psi_hx, s.psi_f, s.psi_in = (float(psi_ch), float(psi_hx),
                                             float(psi_f),  float(psi_in))
    s.eta_ex = (1.0 - psi_cool / psi_in) * 100.0

    return s.eta_ex


def calc_CPR(s: DCState, p: DCParams) -> float:
    """
    Computing Performance Ratio (Eqs. 25–26).

    Parameters
    ----------
    s    : DCState instance
    p    : DCParams instance

    Returns
    -------
    CPR  [-], clipped to [0, 1]
    """
    T_f_celsius = s.T_f[p.is_rack, :] - 273.15    # temperatures in Celsius
    T_CPU = (
        p.E1
        + p.E2 * s.u
        + p.E3 * T_f_celsius
        + p.E4 * s.u ** 2
        + p.E5 * s.u * T_f_celsius
    )

    s.CPR = p.F1 + p.F2 * np.sum(T_CPU)/(p.N_z*p.N_r)

    return s.CPR


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_physics(s: DCState, p: DCParams, tol:  float = 1e-6) -> None:
    """Raise ValueError if any state variable violates physical bounds."""
    errors: list[str] = []

    if (s.T_cw-273.15) > p.T_cw_max or (s.T_cw-273.15) < p.T_cw_min:
        errors.append(f"Chilled water temperature out of range : T_cw={s.T_cw} should be in [{p.T_cw_min},{p.T_cw_max}]")
    if s.T_ca >= s.T_ha - tol:
        errors.append(f"Air not heated:    T_ca={s.T_ca:.2f} ≥ T_ha={s.T_ha:.2f} °C")
    if s.T_cw >= s.T_hw - tol:
        errors.append(f"Water not heated:  T_cw={s.T_cw:.2f} ≥ T_hw={s.T_hw:.2f} °C")
    if s.T_ca <= s.T_cw + tol:
        errors.append(f"HX infeasible:     T_ca={s.T_ca:.2f} ≤ T_cw={s.T_cw:.2f} °C")
    if not (0.0 < s.T_ca < 30.0):
        errors.append(f"T_ca={s.T_ca:.2f} °C outside (0, 30)")
    if not (0.0 < s.T_ha < 60.0):
        errors.append(f"T_ha={s.T_ha:.2f} °C outside (0, 60)")
    if not (0.0 < s.T_hw < 40.0):
        errors.append(f"T_hw={s.T_hw:.2f} °C outside (0, 40)")
    if s.P_ch <= 0.0:
        errors.append(f"Non-positive chiller power: P_ch={s.P_ch:.1f} W")
    if s.P_f  <= 0.0:
        errors.append(f"Non-positive fan power:      P_f={s.P_f:.1f} W")

    if errors:
        raise ValueError(
            "\n!!! PHYSICS VIOLATION !!!\n"
            + "\n".join(f"  → {e}" for e in errors)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _scale_kpis_for_n_packs(s: DCState, p: DCParams) -> tuple:
    """
    Given a converged single-pack DCState *s*, compute the KPIs (COP, CPR,
    η_ex) for the full DC consisting of p.n_packs identical packs that share
    ONE chiller.

    Physics of the scaling
    ──────────────────────
    • Each pack contributes the same thermal load: Q_coil_pack = ρ_w Cp_w Q_w (T_hw−T_cw).
    • The shared chiller handles the TOTAL load: Q_coil_total = n × Q_coil_pack,
      so its PLR rises from PLR_single to PLR_total = n × PLR_single.
      Because EIRFPLR is non-linear in PLR, P_ch_total ≠ n × P_ch_single.
    • Fan and server powers scale linearly: P_f_total = n × P_f, P_IT_total = n × P_IT.
    • COP_total = P_IT_total / (P_ch_total + P_f_total).
    • CPR is a spatial temperature ratio → identical for each pack → CPR_total = CPR_single.
    • Exergy: most terms scale by n; the condenser exit temperature T_hamb is
      recomputed with the total load and the single chiller's condenser fan.

    Returns
    ───────
    (COP_total, CPR_total, eta_ex_total, P_ch_total, P_f_total, PLR_total)
    """
    n = p.n_packs

    # ── Single-pack thermal quantities ────────────────────────────────────
    Q_coil_pack = p.rho_w * p.Cp_w * p.Q_w * (s.T_hw - s.T_cw)   # [W]
    P_IT_pack   = float(np.sum(s.P_z))                              # [W]
    P_f_pack    = s.P_f                                             # [W] (already computed)

    # ── Totals ────────────────────────────────────────────────────────────
    Q_coil_total = n * Q_coil_pack
    P_IT_total   = n * P_IT_pack
    P_f_total    = n * P_f_pack          # n packs × same fans at same flow-per-unit

    # ── Chiller power for the total load ──────────────────────────────────
    T_cw_C  = s.T_cw  - 273.15
    T_amb_C = p.T_amb - 273.15

    # one shared chiller → same capacity (and cap) as for a single pack
    Q_ch_max_total, P_ch_full_load = chiller_full_load(p, T_cw_C, T_amb_C)

    PLR_total   = Q_coil_total / Q_ch_max_total
    s.Q_coil, s.Q_ch_max = float(Q_coil_total), float(Q_ch_max_total)
    PLR_clamped = float(np.clip(PLR_total, p.PLR_min, p.PLR_max))
    EIRFPLR     = p.ee1 + p.ee2*PLR_clamped + p.ee3*PLR_clamped**2
    P_ch_total  = P_ch_full_load * EIRFPLR

    # ── COP ───────────────────────────────────────────────────────────────
    P_cool_total = P_ch_total + P_f_total
    COP_total    = P_IT_total / P_cool_total if P_cool_total > 0 else 0.0

    # ── CPR: purely spatial — same distribution per pack ─────────────────
    CPR_total = s.CPR

    # ── Exergy efficiency for n_packs sharing one chiller ─────────────────
    #    Nomenclature matches calc_exergy (psi = exergy destruction [W])
    T_ha    = s.T_ha
    T_ca    = s.T_ca
    T_cw_K  = s.T_cw
    T_hw_K  = s.T_hw
    T_amb_K = p.T_amb

    nan = float('nan')
    if T_hw_K <= 0 or T_cw_K <= 0 or T_ha <= 0 or T_ca <= 0:
        s.psi_ch = s.psi_hx = s.psi_f = s.psi_in = nan
        return COP_total, CPR_total, nan, P_ch_total, P_f_total, PLR_total

    # Condenser exit temp: one chiller handles total (n-pack) heat rejection
    T_hamb = T_amb_K + (Q_coil_total + P_ch_total) / (p.rho_a * p.Cp_a * p.Q_cf)
    if T_hamb <= 0:
        s.psi_ch = s.psi_hx = s.psi_f = s.psi_in = nan
        return COP_total, CPR_total, nan, P_ch_total, P_f_total, PLR_total

    # psi_ch: exergy destruction in the shared chiller
    #   water side  ∝ Q_w_total = n × Q_w_pack  (each pack has its own water circuit)
    #   air side    uses the single condenser fan (Q_cf unchanged)
    psi_ch = (
        p.rho_w * p.Cp_w * (n * p.Q_w) * T_amb_K * np.log(T_cw_K / T_hw_K)
        + p.rho_a * p.Cp_a * p.Q_cf * T_amb_K * np.log(T_hamb / T_amb_K)
    )

    # psi_hx: exergy destruction in IRC heat exchangers — scales linearly
    psi_hx_pack = (
        p.rho_a * p.Cp_a * s.Q_a
        * (T_ha - T_ca - T_amb_K * np.log(T_ha / T_ca))
        + p.rho_w * p.Cp_w * p.Q_w
        * (T_cw_K - T_hw_K - T_amb_K * np.log(T_cw_K / T_hw_K))
    )
    psi_hx = n * psi_hx_pack

    # psi_f: fan exergy destruction — scales linearly
    psi_f = (1.0 - p.eta_fan) * P_f_total

    psi_cool = psi_ch + psi_hx + psi_f

    # psi_in: total exergy input to the cooling cycle
    psi_in = (
        p.rho_a * p.Cp_a * (n * s.Q_a)
        * (T_ha - T_amb_K - T_amb_K * np.log(T_ha / T_amb_K))
        + P_f_total + P_ch_total
    )

    eta_ex_total = (1.0 - psi_cool / psi_in) * 100.0 if psi_in > 0 else float('nan')
    s.psi_ch, s.psi_hx, s.psi_f, s.psi_in = (float(psi_ch), float(psi_hx),
                                             float(psi_f),  float(psi_in))

    return COP_total, CPR_total, eta_ex_total, P_ch_total, P_f_total, PLR_total


def run_simulation(x: list[float], p: DCParams, answer: str, plot: bool,
                   return_all: bool = False,
                   save_dir: str | None = None,
                   run_tag: str | None = None,
                   allow_infeasible: bool = False,
                   show: bool = True) -> float:
    """
    Simulates the data centre operation for a certain utilisation,

    Parameters
    ----------
    x           : list containing T_cw [°C], Q_a [m³ s⁻¹] and u_set
    p           : DCParams (uses defaults if None)
    answer      : choice of output (COP, eta_ex or CPR)
    plot        : plots the pressure and temperature maps if true
    allow_infeasible : with return_all, also return the KPIs of a point that
                  violates the T_f / PLR limits (instead of None)
    show        : with plot, call plt.show() (False = leave the figures open,
                  e.g. to show them all at once after a loop)

    Returns
    -------
    float: either COP, eta_ex or CPR
    """
    if p is None:
        p = DCParams()
    if answer != "COP" and answer != "eta_ex" and answer != "CPR":
        raise(ValueError(f"The answer requested is not possible. Please select either 'COP', 'eta_ex' or 'CPR'.\n"))
    
    T_cw_C = x[0]
    Q_a = x[1]
    u_racks = x[2:]

    U_SET = np.zeros((p.N_r, p.N_z))
    if len(u_racks) == p.N_r:
        # Per-rack mode: broadcast the same utilisation to every zone in that rack
        for i in range(p.N_r):
            U_SET[i, :] = u_racks[i]
    elif len(u_racks) == p.N_r * p.N_z:
        # Per-zone mode: each rack×zone pair has its own utilisation
        for i in range(p.N_r):
            for j in range(p.N_z):
                U_SET[i, j] = u_racks[i*p.N_z+j]
    else:
        raise ValueError(
            f"x[2:] has {len(u_racks)} elements but N_r={p.N_r} (per-rack) "
            f"or N_r×N_z={p.N_r*p.N_z} (per-zone) are the only valid lengths."
        )
    
    s = DCState(
        p_f=np.zeros((p.N_r + p.N_cu, p.N_z)),
        p_b=np.zeros((p.N_r + p.N_cu, p.N_z)),
        Q_z=np.zeros((p.N_r + p.N_cu, p.N_z)),
        T_f=np.full((p.N_r + p.N_cu, p.N_z), 22.0 + 273.15),
        T_b=np.zeros((p.N_r + p.N_cu, p.N_z)),
        T_e=np.zeros((p.N_r, p.N_z)),
        T_ca=25.0 + 273.15,
        T_hw=15.0 + 273.15,
        T_cw=T_cw_C + 273.15,
        Q_a=Q_a,
        u=U_SET,
        P_z=np.ones((p.N_r, p.N_z)) * p.N_s * (p.A1 + p.A2 * U_SET)
    )

    cooler_guess = [s.T_ca, s.T_hw]

    for iteration in range(5):
        try:
            pressure_network(p, s)
            temperature_network(p, s)

            s.T_ha = calc_T_ha(s, p)
            sol, info, ier, msg = fsolve(chiller_residuals, cooler_guess, args=(s, p), full_output=True)
            
            s.T_ca = sol[0]
            s.T_hw = sol[1]
            cooler_guess = sol
        except Exception:
            return None if return_all else 1000.0   # reject the cases when no solutions were found
    
    try:
        # ── Single-pack physics → COP / η_ex / CPR ───────────────────────
        calc_Pch_EIRfPLR(s, p)
        calc_Pf(s, p)
        s.P_cool = s.P_ch + s.P_f
        P_it_pack = float(np.sum(s.P_z))

        s.COP = P_it_pack / s.P_cool if s.P_cool > 0 else 0.0

        calc_exergy(s, p)
        calc_CPR(s, p)

        # ── Multi-pack scaling: overwrite KPIs for full DC ────────────────
        if p.n_packs > 1:
            (s.COP, s.CPR, s.eta_ex,
             s.P_ch, s.P_f, s.PLR) = _scale_kpis_for_n_packs(s, p)
            s.P_cool = s.P_ch + s.P_f

        T_excess  = max(0, np.max(s.T_f) - 273.15 - p.T_server_red)     # soft penalty when T_f > T_server_red
        PLR_excess = max(0.0, s.PLR - p.PLR_max) + max(0.0, p.PLR_min - s.PLR)  # soft penalty for PLR out of range
        T_f_min   = float(np.min(s.T_f[p.is_rack, :])) - 273.15                 # coldest server inlet [°C]
        T_deficit = (max(0.0, p.T_server_min - T_f_min)                         # soft penalty when T_f < T_server_min
                     if p.T_server_min is not None and answer in p.T_server_min_for else 0.0)

    except Exception:
        return None if return_all else 1000.0

    # ── Early return for multi-objective / logging mode ──────────────────
    if return_all:
        if (T_excess < 0 or PLR_excess > 0 or T_deficit > 0) and not allow_infeasible:
            return None   # infeasible point — exclude from Pareto chart
        P_it_total = float(np.sum(s.P_z)) * p.n_packs
        return (
            s.COP, s.CPR, s.eta_ex,                     # [0] [1] [2]
            s.P_ch, s.P_f, P_it_total,                   # [3] [4] [5]
            s.T_hw - 273.15, s.T_ca - 273.15,            # [6] [7]   °C
            s.T_ha - 273.15,                              # [8]       °C
            float(np.max(s.T_f)) - 273.15,               # [9]       °C
            s.PLR,                                        # [10]
            s.Q_coil, s.Q_ch_max,                         # [11] [12] W  (chiller load / capacity)
            s.psi_ch, s.psi_hx, s.psi_f, s.psi_in,        # [13]–[16] W  (exergy destruction / input)
            T_f_min,                                      # [17]      °C  (coldest server inlet)
        )

    if plot == True:
        n      = p.n_packs
        P_IT_total = float(np.sum(s.P_z)) * n     # single-pack P_z, scaled
        tag    = f"  [{n}-pack DC: optimised 1 pack, KPIs for {n} packs + 1 shared chiller]" if n > 1 else ""
        print(f"---- Performances{tag} ----")
        print(f"  COP    = {s.COP:.4f}  |  eta_ex = {s.eta_ex:.2f} %  |  CPR = {s.CPR:.4f}")
        print(f"  P_IT   = {P_IT_total/1000:.2f} kW  (full DC: {n} pack(s))")
        print(f"  P_ch   = {s.P_ch/1000:.2f} kW  (shared chiller, PLR={s.PLR:.3f})")
        print(f"  P_f    = {s.P_f/1000:.2f} kW  ({n} × fan array)")
        print(f"  P_cool = {s.P_cool/1000:.2f} kW  (P_ch + P_f)")
        print(f"  T_ca   = {s.T_ca-273.15:.2f} °C  |  T_ha = {s.T_ha-273.15:.2f} °C")
        print(f"  T_cw   = {s.T_cw-273.15:.2f} °C  |  T_hw = {s.T_hw-273.15:.2f} °C")
        print(f"  Q_a    = {s.Q_a*n:.3f} m³/s total  ({s.Q_a:.3f} per pack)  |  max T_f = {np.max(s.T_f)-273.15:.2f} °C")
        print(f"  PLR    = {s.PLR:.4f}  (valid range [{p.PLR_min}, {p.PLR_max}])")
        print(f"  Q_ch   = {s.Q_coil/1000:.2f} kW load  /  {s.Q_ch_max/1000:.2f} kW available capacity")
        print(f"  Ex. destruction: chiller {s.psi_ch/1000:.2f} kW | IRC HX {s.psi_hx/1000:.2f} kW"
              f" | fans {s.psi_f/1000:.2f} kW  (exergy input {s.psi_in/1000:.2f} kW)\n")
        figs = {
            "utilisation" : plot_utilisation_grids(s, p),
            "pressure"    : plot_pressure_grids(s, p),
            "temperature" : plot_temperature_grids(s, p),
        }

        if save_dir is not None and run_tag is not None:
            _ensure_dir(save_dir)
            for fig_type, fig in figs.items():
                fpath = os.path.join(save_dir, f"{fig_type}_{run_tag}.png")
                fig.savefig(fpath, dpi=150, bbox_inches="tight")
                print(f"  → saved {fpath}")

        if show:
            plt.show()

    penalty = 1000.0 * T_excess**2 + 1000.0 * PLR_excess**2 + 1000.0 * T_deficit**2

    if answer == "COP":
        return -s.COP + penalty
    elif answer == "eta_ex":
        return -s.eta_ex + penalty
    else:
        return -s.CPR + penalty


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of Pareto-non-dominated rows (all objectives to MAXIMISE).

    Point i is dominated when at least one other point j satisfies
        values[j] >= values[i]  in every column
        values[j] >  values[i]  in at least one column.

    Parameters
    ----------
    values : (N, M) float array — each row is one solution, each column one objective.

    Returns
    -------
    mask : (N,) bool array — True where the point is NOT dominated.
    """
    N = len(values)
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        # Build a view of all OTHER points
        others = np.concatenate([values[:i], values[i+1:]], axis=0)   # (N-1, M)
        dominated_by = (np.all(others >= values[i], axis=1) &
                        np.any(others >  values[i], axis=1))
        if np.any(dominated_by):
            is_efficient[i] = False
    return is_efficient


# ─────────────────────────────────────────────────────────────────────────────
# 6.  OUTPUT HELPERS  (file naming, directory creation, CSV logging)
# ─────────────────────────────────────────────────────────────────────────────

#: Root folder that contains the per-objective sub-directories
#: (= the folder of this script).
_BASE_OUT = os.path.dirname(os.path.abspath(__file__))

#: Mapping from objective name → sub-directory name.
_OBJ_DIR = {
    "COP"    : "optimisation_COP",
    "CPR"    : "optimisation_CPR",
    "eta_ex" : "optimisation_eta_ex",
}


def _run_tag(p: DCParams, LF: float) -> str:
    """
    Build a compact filename stem that encodes the key run parameters.

    Format:  LF{x}e-{k}_Ncu{N_cu}_Nr{N_r}_Np{n_packs}
    Example: LF6e-1_Ncu2_Nr5_Np2  for LF=0.6, 2 IRCs, 5 racks, 2 packs.

    The LF part uses the smallest power of 10 that makes LF an integer:
      0.6 → 6e-1,  0.5 → 5e-1,  0.75 → 75e-2.
    """
    for k in range(1, 6):
        val = round(LF * 10 ** k)
        if abs(val - LF * 10 ** k) < 1e-9:
            lf_tag = f"{val}e-{k}"
            break
    else:
        lf_tag = f"{LF:.4g}".replace(".", "p")   # fallback: "0p6"
    return f"LF{lf_tag}_Ncu{p.N_cu}_Nr{p.N_r}_Np{p.n_packs}"


def _ensure_dir(path: str) -> str:
    """Create *path* (and any parents) if it does not already exist. Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


def _fig_dir(answer: str) -> str:
    """Return the output directory for *answer* objective, creating it if needed."""
    sub = _OBJ_DIR.get(answer, f"optimisation_{answer}")
    return _ensure_dir(os.path.join(_BASE_OUT, sub))


def _perf_extra(triple) -> dict:
    """
    Chiller load/capacity and exergy-destruction columns from the
    run_simulation(..., return_all=True) tuple (NaN when it is None).
    All values are for the full DC (all packs).
    """
    nan = float('nan')
    if triple is None or len(triple) < 17:
        Q_coil = Q_max = psi_ch = psi_hx = psi_f = psi_in = nan
    else:
        Q_coil, Q_max, psi_ch, psi_hx, psi_f, psi_in = triple[11:17]
    psi_tot = psi_ch + psi_hx + psi_f
    pct = lambda v: 100 * v / psi_tot if psi_tot > 0 else nan
    return {
        "Q_cool_load_W"      : Q_coil,
        "Q_ch_max_W"         : Q_max,
        "Q_unmet_W"          : max(0.0, Q_coil - Q_max) if Q_coil == Q_coil else nan,
        "Ex_in_W"            : psi_in,
        "Ex_dest_chiller_W"  : psi_ch,
        "Ex_dest_HX_W"       : psi_hx,
        "Ex_dest_fans_W"     : psi_f,
        "Ex_dest_total_W"    : psi_tot,
        "Ex_dest_chiller_pct": pct(psi_ch),
        "Ex_dest_HX_pct"     : pct(psi_hx),
        "Ex_dest_fans_pct"   : pct(psi_f),
    }


def save_performances(answer: str, data: dict) -> str:
    """
    Append one result row to the accumulating performances CSV for *answer*.

    The CSV lives at  <fig_dir>/performances.csv  and uses pandas for I/O so
    every row is labelled with column headers — trivial to reload later:

        import pandas as pd
        df = pd.read_csv(".../optimisation_COP/performances.csv")

    Returns the full path to the CSV file.
    """
    folder   = _fig_dir(answer)
    csv_path = os.path.join(folder, "performances.csv")

    df_row = pd.DataFrame([data])
    if os.path.exists(csv_path):
        old_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if old_cols == df_row.columns.tolist():
            df_row.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            # new/different columns (e.g. added KPIs): rewrite the file with the
            # union of the columns so older rows stay aligned (missing → empty)
            df_all = pd.concat([pd.read_csv(csv_path), df_row], ignore_index=True)
            df_all.to_csv(csv_path, mode="w", header=True, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True,  index=False)

    print(f"  → performances appended to {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _col_labels(p: DCParams) -> list:
    """Return one x-axis label per column, derived from p.is_irc.

    IRC columns are named Left / Centre [N] / Right in order of appearance.
    Rack columns are numbered sequentially from 1.
    Works for any layout produced by make_layout().
    """
    n_irc = int(np.sum(p.is_irc))
    irc_seen = 0
    rack_seen = 0
    labels = []
    for is_irc_col in p.is_irc:
        if is_irc_col:
            if n_irc == 1:
                labels.append("IRC")
            elif irc_seen == 0:
                labels.append("IRC Left")
            elif irc_seen == n_irc - 1:
                labels.append("IRC Right")
            else:
                n_centre = n_irc - 2
                labels.append("IRC Centre" if n_centre == 1 else f"IRC Centre {irc_seen}")
            irc_seen += 1
        else:
            rack_seen += 1
            labels.append(f"Rack {rack_seen}")
    return labels


def plot_utilisation_grids(s: DCState, p: DCParams):
    """
    Generates a colour-coded heatmap of the server utilisation distribution in the data centre grid, 

    Parameters
    ----------
    s           : DCState instance
    p           : DCParams instance
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    c = ax.imshow((s.u).T, origin='lower', cmap='coolwarm', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_title("Zonal Server Utilization Distribution", fontsize=16, fontweight='bold')
    fig.colorbar(c, ax=ax, label="Utilization (u)")
    
    for i in range(p.N_r):
        for j in range(p.N_z):
            text_color = 'white' if s.u[i, j] < 0.5 else 'black'
            ax.text(i, j, f"{s.u[i, j]:.2f}", ha='center', va='center', fontweight='bold', color=text_color)

    x_labels = [lbl for lbl, irc in zip(_col_labels(p), p.is_irc) if not irc]

    ax.set_xticks(range(p.N_r))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(p.N_z))
    ax.set_ylabel('Vertical Zones (Floor to Ceiling)', fontsize=12)

    plt.tight_layout()
    return fig


def plot_pressure_grids(s: DCState, p: DCParams):
    """
    Generates a colour-coded heatmap of the pressure distribution in the data centre grid, 

    Parameters
    ----------
    s           : DCState instance
    p           : DCParams instance
    """
    P_f = s.p_f
    P_b = s.p_b

    p_min = min(np.min(P_f), np.min(P_b))
    p_max = max(np.max(P_f), np.max(P_b))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Zonal Pressure Distribution Across the Data Center', fontsize=18, fontweight='bold')

    # ---------------------------------------------------------
    # FRONT CHAMBER (Cold Aisle)
    # ---------------------------------------------------------
    # We transpose (.T) the array so the 7 columns are on the X-axis and 5 rows on the Y-axis.
    # origin='lower' ensures index j=0 is at the bottom (the physical floor).
    c1 = ax1.imshow(P_f.T, origin='lower', cmap='coolwarm', aspect='auto', vmin=p_min, vmax=p_max)
    fig.colorbar(c1, ax=ax1, label='Pressure [Pa]')
    ax1.set_title('Front Chamber Pressure ($P^f$)', fontsize=14)
    
    # Overlay the exact numerical values inside the colored boxes
    for i in range(p.N_r+p.N_cu):
        for j in range(p.N_z):
            # Calculate a text color that contrasts with the background
            #text_color = 'white' if P_f[i, j] > (np.max(P_f) + np.min(P_f))/2 else 'black'
            ax1.text(i, j, f"{P_f[i, j]:.1f}", ha='center', va='center', fontweight='bold')

    # ---------------------------------------------------------
    # BACK CHAMBER (Hot Aisle)
    # ---------------------------------------------------------
    c2 = ax2.imshow(P_b.T, origin='lower', cmap='coolwarm', aspect='auto', vmin=p_min, vmax=p_max)
    fig.colorbar(c2, ax=ax2, label='Pressure [Pa]')
    ax2.set_title('Back Chamber Pressure ($P^b$)', fontsize=14)
    
    for i in range(p.N_r+p.N_cu):
        for j in range(p.N_z):
            #text_color = 'white' if P_b[i, j] > (np.max(P_b) + np.min(P_b))/2 else 'black'
            ax2.text(i, j, f"{P_b[i, j]:.1f}", ha='center', va='center', fontweight='bold')

    x_labels = _col_labels(p)
    
    for ax in [ax1, ax2]:
        # Set X-axis labels to match the physical layout
        ax.set_xticks(range(p.N_r+p.N_cu))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
        
        # Set Y-axis labels
        ax.set_yticks(range(p.N_z))
        ax.set_ylabel('Vertical Zones (Floor to Ceiling)', fontsize=12)

    plt.tight_layout()
    return fig


def plot_temperature_grids(s: DCState, p: DCParams):
    """
    Generates a colour-coded heatmap of the temperature distribution in the data centre grid, 

    Parameters
    ----------
    s           : DCState instance
    p           : DCParams instance
    """
    T_f = s.T_f - 273.15
    T_b = s.T_b - 273.15
    T_e = s.T_e - 273.15

    t_min = min(np.min(T_f), np.min(T_b), np.min(T_e))
    t_max = max(np.max(T_f), np.max(T_b), np.max(T_e))

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Zonal Temperature Distribution Across the Data Center', fontsize=18, fontweight='bold')

    gs = gridspec.GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 1:3])

    # ---------------------------------------------------------
    # FRONT CHAMBER (Cold Aisle) - 7x5 Grid
    # ---------------------------------------------------------
    c1 = ax1.imshow(T_f.T, origin='lower', cmap='coolwarm', aspect='auto', vmin=t_min, vmax=t_max)
    fig.colorbar(c1, ax=ax1, label='Temperature [°C]')
    ax1.set_title('Front Chamber Temperature ($T^f$)', fontsize=14)
    
    # Overlay the exact numerical values inside the colored boxes
    for i in range(p.N_r+p.N_cu):
        for j in range(p.N_z):
            # Calculate a text color that contrasts with the background
            #text_color = 'white' if T_f[i, j] > (np.max(T_f) + np.min(T_f))/2 else 'black'
            ax1.text(i, j, f"{T_f[i, j]:.1f}", ha='center', va='center', fontweight='bold')

    # ---------------------------------------------------------
    # BACK CHAMBER (Hot Aisle) - 7x5 Grid
    # ---------------------------------------------------------
    c2 = ax2.imshow(T_b.T, origin='lower', cmap='coolwarm', aspect='auto', vmin=t_min, vmax=t_max)
    fig.colorbar(c2, ax=ax2, label='Temperature [°C]')
    ax2.set_title('Back Chamber Temperature ($T^b$)', fontsize=14)
    
    for i in range(p.N_r+p.N_cu):
        for j in range(p.N_z):
            #text_color = 'white' if T_b[i, j] > (np.max(T_b) + np.min(T_b))/2 else 'black'
            ax2.text(i, j, f"{T_b[i, j]:.1f}", ha='center', va='center', fontweight='bold')
    
    # ---------------------------------------------------------
    # SERVER EXHAUST - 5x5 Grid
    # ---------------------------------------------------------
    c3 = ax3.imshow(T_e.T, origin='lower', cmap='coolwarm', aspect='auto', vmin=t_min, vmax=t_max)
    fig.colorbar(c3, ax=ax3, label='Temperature [°C]')
    ax3.set_title('Server Exhaust Temperature ($T^e$)', fontsize=14)
    
    for i in range(p.N_r):
        for j in range(p.N_z):
            #text_color = 'white' if T_e[i, j] > (np.max(T_e) + np.min(T_e))/2 else 'black'
            ax3.text(i, j, f"{T_e[i, j]:.1f}", ha='center', va='center', fontweight='bold')
    
    # ---------------------------------------------------------
    # FORMATTING THE AXES
    # ---------------------------------------------------------
    x_labels_all = _col_labels(p)
    x_labels_racks = [lbl for lbl, irc in zip(x_labels_all, p.is_irc) if not irc]

    # Format the full-column plots (Front and Back Chambers)
    for ax in [ax1, ax2]:
        ax.set_xticks(range(p.N_r+p.N_cu))
        ax.set_xticklabels(x_labels_all, rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(p.N_z))
        ax.set_ylabel('Vertical Zones (Floor to Ceiling)', fontsize=12)

    # Format the rack-only plot (Server Exhaust) separately
    ax3.set_xticks(range(p.N_r))
    ax3.set_xticklabels(x_labels_racks, rotation=45, ha='right', fontsize=11)
    ax3.set_yticks(range(p.N_z))
    # We don't need the Y-label again since it's implied, but keeping it for consistency
    ax3.set_ylabel('Vertical Zones (Floor to Ceiling)', fontsize=12)

    plt.tight_layout()
    return fig


def plot_heatmaps(df: pd.DataFrame, m: int, n: int) -> plt.Figure:
    """
    Six-panel contour heatmaps for all performance metrics.

    Parameters
    ----------
    df   : output of run_simulation()
    m, n : grid dimensions (must match the simulation)

    Returns
    -------
    matplotlib Figure
    """
    T_cw_vals = np.sort(df["T_cw [C]"].unique())
    Q_a_vals  = np.sort(df["Q_a [m3/s]"].unique())
    Q_grid, T_grid = np.meshgrid(Q_a_vals, T_cw_vals)

    panels = [
        ("COP",                r"COP [-]",               r"Coefficient of Performance"),
        ("P_TotalCooling [W]", r"Total Cooling Power [W]", r"Total Cooling Power ($P_{ch}+P_f$)"),
        ("P_chiller [W]",      r"Chiller Power [W]",       r"Chiller Power ($P_{ch}$)"),
        ("P_fan [W]",          r"Fan Power [W]",            r"Fan Power ($P_f$)"),
        ("eta_ex",             r"Exergy Efficiency [%]",    r"Exergy Efficiency ($\eta_{ex}$)"),
        ("CPR",                r"CPR [-]",                  r"Computing Performance Ratio ($CPR$)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        "System Performance Matrix: Airflow vs Chilled Water Temperature",
        fontsize=15, fontweight="bold",
    )

    for ax, (col, cbar_label, title) in zip(axes.flat, panels):
        Z  = df[col].values.reshape(m, n)
        cf = ax.contourf(Q_grid, T_grid, Z, levels=20, cmap="coolwarm")
        cb = fig.colorbar(cf, ax=ax)
        cb.set_label(cbar_label, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"Airflow Rate $Q_a$ [m³/s]", fontsize=11)
        ax.set_ylabel(r"Chilled Water Setpoint $T_{cw}$ [°C]", fontsize=11)

    fig.tight_layout()
    return fig


def plot_chiller_comparison(p: DCParams):
    """
    Two-panel COP comparison of the Gordon-Ng (Trane CGA120, 10 ton) and
    EIR-fPLR (Trane CGAM20, 20 ton) chiller models.

    Because the two chillers have *different rated capacities* (35.2 kW vs
    68.9 kW), a fair comparison uses the dimensionless part-load ratio
        PLR = Q_ch / Q_ch,rated  (each chiller's own reference)
    to define "operating at the same load", and COP = Q_ch / P_ch as the
    performance metric (independent of machine size).

    Key result for EIR-fPLR at fixed PLR (CAPFT cancels in Q_ch/P_ch):
        COP_EIR(T_cw, PLR) = PLR × COP_ref / [EIRFT(T_cw) × EIRFPLR(PLR)]

    (a) COP vs T_cw  at PLR = [0.25, 0.50, 0.75, 1.00]
        Gordon-Ng COP rises dramatically above 10 °C (reaching ~9 at 20 °C
        for PLR = 0.5), explaining why paper [84] finds T_cw,opt = 20 °C.
        EIR-fPLR COP rises only ~10 % over its rated range [5.56, 10 °C].

    (b) COP vs PLR  at T_cw = 7.78 °C  (mid of EIR valid range)
        Both models at a common, valid temperature; shows how each responds
        to part-load.

    All panels at T_amb = p.T_amb (design-point ambient, default 25 °C).
    """
    T_amb_K = p.T_amb           # K
    T_amb_C = T_amb_K - 273.15  # °C

    # Rated capacities — each chiller's own reference
    Q_GN_ref  = 35_172.0    # W   Trane CGA120: 10 ton × 3517.2 W/ton
    Q_EIR_ref = p.Q_ch_ref  # W   Trane CGAM20: 68 930 W (from DCParams)

    # EIR-fPLR rated valid T_cw range (TRANE CGAM20 / EnergyPlus IDD)
    T_EIR_lo, T_EIR_hi = 5.56, 10.0   # °C

    # ── COP helpers  (array-safe, PLR-parametrised) ───────────────────────────

    def _COP_GN_Tcw(PLR_val, T_cw_K_arr):
        """Gordon-Ng COP vs T_cw at fixed PLR.
        Q_ch = PLR × Q_GN_ref is constant; T_cw is swept."""
        T   = np.asarray(T_cw_K_arr, dtype=float)
        Q   = float(PLR_val) * Q_GN_ref / 1e3              # kW (scalar)
        num = T_amb_K * Q + p.C1 * T * T_amb_K + p.C2 * (T_amb_K - T)
        den = T - p.C3 * Q
        with np.errstate(divide='ignore', invalid='ignore'):
            P_kW = np.where(den > 0.0, num / den - Q, np.nan)
        P_W = P_kW * 1e3
        return np.where(P_W > 1.0, float(PLR_val) * Q_GN_ref / P_W, np.nan)

    def _COP_EIR_Tcw(PLR_val, T_cw_C_arr):
        """EIR-fPLR COP vs T_cw at fixed PLR.
        Simplified form: COP = PLR × COP_ref / (EIRFT × EIRFPLR).
        CAPFT cancels because it scales both Q_ch and P_full equally."""
        T       = np.asarray(T_cw_C_arr, dtype=float)
        EIRFT   = (p.dd1 + p.dd2*T + p.dd3*T**2
                   + p.dd4*T_amb_C + p.dd5*T_amb_C**2 + p.dd6*T*T_amb_C)
        PLR_c   = float(np.clip(PLR_val, p.PLR_min, p.PLR_max))
        EIRFPLR = p.ee1 + p.ee2*PLR_c + p.ee3*PLR_c**2
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(EIRFT * EIRFPLR > 0.0,
                            float(PLR_val) * p.COP_ref / (EIRFT * EIRFPLR),
                            np.nan)

    def _COP_GN_PLR(PLR_arr, T_cw_C_fixed):
        """Gordon-Ng COP vs PLR at fixed T_cw."""
        T_K = float(T_cw_C_fixed) + 273.15
        PLR = np.asarray(PLR_arr, dtype=float)
        Q   = PLR * Q_GN_ref / 1e3                         # kW
        num = T_amb_K * Q + p.C1 * T_K * T_amb_K + p.C2 * (T_amb_K - T_K)
        den = T_K - p.C3 * Q
        with np.errstate(divide='ignore', invalid='ignore'):
            P_kW = np.where(den > 0.0, num / den - Q, np.nan)
        P_W = P_kW * 1e3
        return np.where(P_W > 1.0, PLR * Q_GN_ref / P_W, np.nan)

    def _COP_EIR_PLR(PLR_arr, T_cw_C_fixed):
        """EIR-fPLR COP vs PLR at fixed T_cw.
        COP = PLR × COP_ref / (EIRFT × EIRFPLR(PLR))."""
        T_C     = float(T_cw_C_fixed)
        PLR     = np.asarray(PLR_arr, dtype=float)
        EIRFT_v = (p.dd1 + p.dd2*T_C + p.dd3*T_C**2
                   + p.dd4*T_amb_C + p.dd5*T_amb_C**2 + p.dd6*T_C*T_amb_C)
        PLR_c   = np.clip(PLR, p.PLR_min, p.PLR_max)
        EIRFPLR = p.ee1 + p.ee2*PLR_c + p.ee3*PLR_c**2
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(EIRFT_v * EIRFPLR > 0.0,
                            PLR * p.COP_ref / (EIRFT_v * EIRFPLR),
                            np.nan)

    # ── Style ─────────────────────────────────────────────────────────────────
    GN_blue  = '#1f77b4'
    EIR_red  = '#d62728'
    PLR_list = [0.25, 0.50, 0.75, 1.00]
    lw_map   = {0.25: 1.1, 0.50: 1.7, 0.75: 2.3, 1.00: 2.9}

    # ── Shared sweeps ─────────────────────────────────────────────────────────
    T_arr   = np.linspace(4.0, 23.0, 400)   # °C — T_cw sweep for panel (a)
    T_K_arr = T_arr + 273.15
    vmask   = (T_arr >= T_EIR_lo) & (T_arr <= T_EIR_hi)
    PLR_arr = np.linspace(0.05, 1.05, 400)  # PLR sweep for panels (b)/(c)
    T_mid_C = 0.5 * (T_EIR_lo + T_EIR_hi)  # 7.78 °C — midpoint of EIR range

    from matplotlib.lines   import Line2D
    from matplotlib.patches import Patch

    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Chiller COP comparison  (adimensionalised by PLR = $Q_{ch}/Q_{ch,\\mathrm{ref}}$)\n"
        "Gordon-Ng  –  Trane CGA120  (10 ton,  $Q_\\mathrm{ref}$ = 35.2 kW)"
        r"      vs      "
        f"EIR-fPLR  –  Trane CGAM20  (20 ton,  $Q_\\mathrm{{ref}}$ = {Q_EIR_ref/1e3:.1f} kW)"
        f"      |      $T_\\mathrm{{amb}}$ = {T_amb_C:.0f} °C",
        fontsize=11, fontweight='bold',
    )

    # ─────────────────────── PANEL (a): COP vs T_cw at fixed PLR ─────────────
    ax = axes[0]
    for PLR in PLR_list:
        lw   = lw_map[PLR]
        cGN  = _COP_GN_Tcw( PLR, T_K_arr)
        cEIR = _COP_EIR_Tcw(PLR, T_arr)
        ax.plot(T_arr,          cGN,           color=GN_blue, lw=lw)
        ax.plot(T_arr[vmask],   cEIR[vmask],   color=EIR_red, lw=lw, ls='-')
        ax.plot(T_arr[~vmask],  cEIR[~vmask],  color=EIR_red, lw=lw, ls='--', alpha=0.45)

    ax.axvspan(T_EIR_lo, T_EIR_hi, alpha=0.07, color=EIR_red)
    ax.axvline(20.0, color='grey', lw=1.2, ls=':', zorder=5)
    ax.set_xlim(4, 23)
    ax.set_xlabel(r"$T_{cw}$  [°C]", fontsize=12)
    ax.set_ylabel(r"Chiller COP  [–]", fontsize=12)
    ax.set_title("(a)  COP vs supply temperature\n"
                 "(same PLR for both chillers)", fontsize=12)
    ax.grid(True, alpha=0.3)
    handles_a = [
        Line2D([0],[0], color=GN_blue, lw=2.2,
               label='Gordon-Ng  (no $T_{cw}$ limit)'),
        Line2D([0],[0], color=EIR_red, lw=2.2,
               label='EIR-fPLR  (solid = rated range)'),
        Line2D([0],[0], color=EIR_red, lw=2.2, ls='--', alpha=0.5,
               label='EIR-fPLR  (extrapolated)'),
        Patch(color=EIR_red, alpha=0.18,
              label=f'EIR rated: {T_EIR_lo}–{T_EIR_hi} °C'),
        Line2D([0],[0], color='grey', lw=1.2, ls=':',
               label=r'Paper [84]: $T_{cw,opt}=20$ °C'),
    ] + [
        Line2D([0],[0], color='k', lw=lw_map[PLR], alpha=0.75,
               label=f'PLR = {PLR:.2f}')
        for PLR in PLR_list
    ]
    ax.legend(handles=handles_a, fontsize=7.5, loc='upper left', framealpha=0.88)

    # ─────────────────── PANELS (b): COP vs PLR ──────────────────────
    ax = axes[1]
    #for ax, T_fix, title in zip(axes[1:], T_cw_panels, titles_bc):
    in_range = (T_EIR_lo <= T_mid_C <= T_EIR_hi)

    COP_GN  = _COP_GN_PLR( PLR_arr, T_mid_C)
    COP_EIR = _COP_EIR_PLR(PLR_arr, T_mid_C)

    ax.plot(PLR_arr, COP_GN,  color=GN_blue, lw=2.2, label='Gordon-Ng')
    ls_eir  = '-'  if in_range else '--'
    lbl_eir = 'EIR-fPLR' if in_range else 'EIR-fPLR  (extrapolated)'
    ax.plot(PLR_arr, COP_EIR, color=EIR_red, lw=2.2,
            ls=ls_eir, label=lbl_eir)

    # EIR valid PLR band
    ax.axvspan(p.PLR_min, p.PLR_max, alpha=0.06, color=EIR_red)
    ax.axvline(p.PLR_min, color='orange', lw=1.2, ls='--',
                label=f'EIR $PLR_{{\\min}}$ = {p.PLR_min}')
    ax.axvline(1.0, color='grey', lw=1.0, ls=':')

    if not in_range:
        ax.text(0.98, 0.97, 'EIR-fPLR\nextrapolated',
                transform=ax.transAxes, fontsize=8.5,
                color=EIR_red, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='lightyellow',
                            edgecolor=EIR_red, alpha=0.85))

    ax.set_xlabel(r"PLR  $= Q_{ch}\,/\,Q_{ch,\mathrm{ref}}$  [–]",
                    fontsize=12)
    ax.set_ylabel(r"Chiller COP  [–]", fontsize=12)
    ax.set_title(f"(b)  COP vs PLR\n ($T_{{cw}}$ = {T_mid_C:.2f} °C  –  inside EIR rated range)", fontsize=12)
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    return fig


def plot_chiller_vs_Tamb(p: DCParams, T_cw_C: float,
                         T_amb_lo: float = 0.0, T_amb_hi: float = 50.0) -> plt.Figure:
    """
    EIR-fPLR chiller (Trane CGAM20) full-load performance (PLR = 1) vs ambient
    temperature at a fixed chilled-water supply temperature T_cw, compared with
    the manufacturer datasheet (Trane CG-PRC017J-EN, Table 11, 60 Hz, SI units,
    20-ton unit).

    (a) Cooling capacity  Q_ch,max = Q_ch,ref × CAPFT(T_cw, T_amb)
    (b) COP at full load  COP = COP_ref / [EIRFT(T_cw, T_amb) × EIRFPLR(1)]

    Datasheet points exist for T_cw = 7 and 9 °C only.  For 7 < T_cw < 9 °C
    they are linearly interpolated (permitted by the datasheet, note 4);
    outside that range no datasheet points are shown (extrapolation is not
    permitted by the manufacturer).

    Solid lines = inside the rated T_amb range [p.T_amb_min, p.T_amb_max],
    dashed lines = extrapolated.

    Parameters
    ----------
    p        : DCParams instance
    T_cw_C   : chilled-water supply temperature                   [°C]
    T_amb_lo, T_amb_hi : ambient temperature sweep limits         [°C]
    """
    # ── Datasheet: Trane CGAM20, Table 11 (60 Hz, SI) ─────────────────────
    #   T_cw [°C] → (kW cooling, COP) at T_amb = 30, 35, 40, 45 °C
    T_amb_ds = np.array([30.0, 35.0, 40.0, 45.0])
    ds = {
        7.0: (np.array([73.13, 69.27, 65.05, 60.83]),
              np.array([3.46,  3.02,  2.64,  2.26])),
        9.0: (np.array([77.70, 73.48, 69.27, 64.34]),
              np.array([3.60,  3.16,  2.75,  2.37])),
    }
    T = float(T_cw_C)
    T_ds_lo, T_ds_hi = min(ds), max(ds)
    if T_ds_lo <= T <= T_ds_hi:
        w = (T - T_ds_lo) / (T_ds_hi - T_ds_lo)
        Q_ds   = (1 - w) * ds[T_ds_lo][0] + w * ds[T_ds_hi][0]
        COP_ds = (1 - w) * ds[T_ds_lo][1] + w * ds[T_ds_hi][1]
        ds_label = ("Datasheet (Table 11)" if T in ds
                    else "Datasheet (Table 11, interpolated)")
    else:
        Q_ds = COP_ds = None
        print(f"  [warning] no datasheet data for T_cw = {T:.2f} °C "
              f"(available: {T_ds_lo:g}–{T_ds_hi:g} °C) → model curves only")

    # ── EIR-fPLR model at PLR = 1 ─────────────────────────────────────────
    def _model(T_amb):
        CAPFT = (p.cc1 + p.cc2*T + p.cc3*T**2
                 + p.cc4*T_amb + p.cc5*T_amb**2 + p.cc6*T*T_amb)
        EIRFT = (p.dd1 + p.dd2*T + p.dd3*T**2
                 + p.dd4*T_amb + p.dd5*T_amb**2 + p.dd6*T*T_amb)
        EIRFPLR = p.ee1 + p.ee2 + p.ee3          # PLR = 1
        return p.Q_ch_ref * CAPFT / 1e3, p.COP_ref / (EIRFT * EIRFPLR)

    T_amb = np.linspace(T_amb_lo, T_amb_hi, 400)   # °C
    vmask = (T_amb >= p.T_amb_min) & (T_amb <= p.T_amb_max)
    Q_kW, COP = _model(T_amb)

    # ── Console comparison at the datasheet points ────────────────────────
    if Q_ds is not None:
        Q_m, COP_m = _model(T_amb_ds)
        print(f"\n  T_cw = {T:.2f} °C  |  model (EIR-fPLR, PLR=1) vs datasheet")
        print(f"  {'T_amb':>6} {'Q_model':>9} {'Q_ds':>8} {'err%':>7}"
              f" {'COP_mod':>8} {'COP_ds':>7} {'err%':>7}")
        for i in range(len(T_amb_ds)):
            print(f"  {T_amb_ds[i]:6.1f} {Q_m[i]:9.2f} {Q_ds[i]:8.2f}"
                  f" {100*(Q_m[i]/Q_ds[i]-1):7.2f}"
                  f" {COP_m[i]:8.3f} {COP_ds[i]:7.2f}"
                  f" {100*(COP_m[i]/COP_ds[i]-1):7.2f}")

    T_cw_in_range = (p.T_cw_min <= T <= p.T_cw_max)
    cw_note = "" if T_cw_in_range else "  (outside rated $T_{cw}$ range → extrapolated)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Trane CGAM20 at full load (PLR = 1): EIR-fPLR model vs datasheet  |  "
        f"$T_{{cw}}$ = {T:.2f} °C{cw_note}",
        fontsize=12, fontweight='bold',
    )

    panels = [
        (axes[0], Q_kW, Q_ds, r"Cooling capacity $Q_{ch}$  [kW]",
         "(a)  Cooling capacity vs ambient temperature"),
        (axes[1], COP,  COP_ds, r"Chiller COP  [–]",
         "(b)  COP vs ambient temperature"),
    ]
    for ax, y, y_ds, ylabel, title in panels:
        ax.plot(T_amb, np.where(vmask, y, np.nan), color='#1f77b4', lw=2.2,
                label='EIR-fPLR model')
        ax.plot(T_amb, np.where(~vmask, y, np.nan), color='#1f77b4', lw=2.2,
                ls='--', alpha=0.5, label='EIR-fPLR (extrapolated)')
        if y_ds is not None:
            ax.plot(T_amb_ds, y_ds, 'o', color='#d62728', ms=7,
                    mec='k', mew=0.6, zorder=5, label=ds_label)
        ax.axvspan(p.T_amb_min, p.T_amb_max, alpha=0.06, color='#d62728')
        ax.set_xlim(T_amb_lo, T_amb_hi)
        ax.set_xlabel(r"$T_{amb}$  [°C]", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', framealpha=0.88)

    fig.tight_layout()
    return fig


def plot_pareto_fronts(p: DCParams, LF: float = 0.5, T_cw_min: float = 5.56, T_cw_max: float = 10.0, n_weights: int   = 5, 
                       maxiter_local: int = 50, scales: tuple = (4.0, 0.65, 20.0)) -> plt.Figure:
    """
    Approximate and plot the three-objective Pareto front for
        maximise  COP,  CPR,  η_ex
    using weighted-sum scalarisation.

    For each weight triplet (w_COP, w_CPR, w_η) on a uniform grid of the
    probability simplex, one SLSQP constrained minimisation is run.
    After collecting all feasible solutions, Pareto dominance filtering
    keeps only the non-dominated ones.

    The figure has four panels:
      • top row  — 3-D scatter in (COP, CPR, η_ex) space
      • bottom row — three 2-D projections: COP–CPR, COP–η_ex, CPR–η_ex

    Parameters
    ----------
    p             : DCParams (layout, chiller model, etc.)
    LF            : mean utilisation equality constraint
    T_cw_min/max  : chilled-water setpoint bounds  [°C]
    n_weights     : simplex grid resolution.
                    Total runs = (n+1)(n+2)//2.
                    n=4 → 15 runs,  n=5 → 21 runs,  n=6 → 28 runs.
    maxiter_local : SLSQP iteration budget per weight combination.
    scales        : (COP_scale, CPR_scale, eta_scale) — rough expected
                    magnitudes used to normalise the objectives before
                    weighting.  Defaults fit typical EIR-fPLR results.
    """
    Q_a_min = 0.6  * p.N_cu
    Q_a_max = 1.33 * p.N_cu

    bounds = (
        [(T_cw_min, T_cw_max), (Q_a_min, Q_a_max)]
        + [(0.0, 1.0)] * p.N_r          # per-rack utilisation
    )

    def workload_constraint(x):
        return np.mean(x[2:]) - LF

    constraints = {'type': 'eq', 'fun': workload_constraint}
    x0_base     = np.array([T_cw_min, (Q_a_min + Q_a_max) / 2.0] + [LF] * p.N_r)

    s_COP, s_CPR, s_eta = scales

    # ── Build the weight simplex grid ────────────────────────────────────
    weights = []
    for i in range(n_weights + 1):
        for j in range(n_weights + 1 - i):
            k = n_weights - i - j
            weights.append(np.array([i, j, k], dtype=float) / n_weights)
    n_runs = len(weights)

    print(f"\nPareto front search: {n_runs} weighted-sum optimisations "
          f"(n_weights={n_weights}, maxiter={maxiter_local}) …")

    rng = np.random.default_rng(42)
    all_triples = []   # list of 11-tuples (COP, CPR, eta_ex, …) for feasible solutions

    for idx, w in enumerate(weights):
        w_COP, w_CPR, w_eta = w

        # Closure captures w and scales for this iteration
        def _obj(x, _w=w):
            triple = run_simulation(x, p, "COP", False, return_all=True)
            if triple is None:
                return 1000.0
            cop, cpr, eta = triple[0], triple[1], triple[2]
            return -(  _w[0] * cop / s_COP
                     + _w[1] * cpr / s_CPR
                     + _w[2] * eta / s_eta  )

        # Small random perturbation on u so different weight combos explore
        # different starting points and are less likely to all find the same basin
        x0 = x0_base.copy()
        x0[2:] = np.clip(
            x0[2:] + rng.uniform(-0.08, 0.08, p.N_r),
            0.0, 1.0
        )

        res = minimize(
            _obj, x0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'maxiter': maxiter_local, 'ftol': 1e-6}
        )

        # Re-evaluate at the converged point to get the clean triple
        if res.fun < 900:
            triple = run_simulation(res.x, p, "COP", False, return_all=True)
        else:
            triple = None

        if triple is not None:
            all_triples.append(triple)
            cop, cpr, eta = triple[0], triple[1], triple[2]
            print(f"  [{idx+1:3d}/{n_runs}]  "
                  f"w=({w_COP:.2f},{w_CPR:.2f},{w_eta:.2f})  →  "
                  f"COP={cop:.3f}  CPR={cpr:.4f}  η_ex={eta:.1f} %")
        else:
            print(f"  [{idx+1:3d}/{n_runs}]  "
                  f"w=({w_COP:.2f},{w_CPR:.2f},{w_eta:.2f})  →  "
                  f"no feasible solution  (f={res.fun:.1f})")

    if len(all_triples) < 2:
        raise RuntimeError(
            "Fewer than 2 feasible solutions found — "
            "check bounds, LF, and T_cw range."
        )

    values = np.array(all_triples)              # shape (N, 3)
    pareto = _pareto_mask(values)
    n_par  = int(pareto.sum())
    print(f"\n  → {n_par} / {len(values)} solutions are Pareto-non-dominated.\n")

    COP_a = values[:, 0];  CPR_a = values[:, 1];  eta_a = values[:, 2]
    COP_p = values[pareto, 0]
    CPR_p = values[pareto, 1]
    eta_p = values[pareto, 2]

    # ── Figure layout: 3-D top row + three 2-D projections bottom row ────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Three-Objective Pareto Front  —  "
        r"$\max$ COP,  $\max$ CPR,  $\max$ $\eta_{\mathrm{ex}}$",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97, top=0.90, bottom=0.09,
        wspace=0.38, hspace=0.45
    )

    # — 3-D scatter (spans full top row) ——————————————————————————————
    ax3 = fig.add_subplot(gs[0, :], projection='3d')

    ax3.scatter(COP_a, CPR_a, eta_a,
                c='steelblue', alpha=0.30, s=28,
                label=f'All feasible ({len(values)})')
    ax3.scatter(COP_p, CPR_p, eta_p,
                c='crimson', s=72, zorder=5, edgecolors='darkred', linewidths=0.5,
                label=f'Pareto front ({n_par})')

    ax3.set_xlabel('COP [–]',                  labelpad=7, fontsize=10)
    ax3.set_ylabel('CPR [–]',                  labelpad=7, fontsize=10)
    ax3.set_zlabel(r'$\eta_{\mathrm{ex}}$ [%]',labelpad=7, fontsize=10)
    ax3.set_title('3-D objective space', fontsize=10, pad=4)
    ax3.legend(fontsize=8, loc='upper left')

    # — 2-D projections ————————————————————————————————————————————
    proj_specs = [
        (gs[1, 0], COP_a, CPR_a, COP_p, CPR_p,
         'COP [–]',                  'CPR [–]',                   'COP – CPR'),
        (gs[1, 1], COP_a, eta_a, COP_p, eta_p,
         'COP [–]',                  r'$\eta_{\mathrm{ex}}$ [%]', r'COP – $\eta_{\mathrm{ex}}$'),
        (gs[1, 2], CPR_a, eta_a, CPR_p, eta_p,
         'CPR [–]',                  r'$\eta_{\mathrm{ex}}$ [%]', r'CPR – $\eta_{\mathrm{ex}}$'),
    ]

    for gsp, xa, ya, xp, yp, xl, yl, title in proj_specs:
        ax = fig.add_subplot(gsp)
        ax.scatter(xa, ya, c='steelblue', alpha=0.30, s=22, zorder=2)
        ax.scatter(xp, yp, c='crimson',   s=55,       zorder=5,
                   edgecolors='darkred', linewidths=0.5)
        ax.set_xlabel(xl,    fontsize=10)
        ax.set_ylabel(yl,    fontsize=10)
        ax.set_title(title,  fontsize=10)
        ax.grid(True, alpha=0.3)

    return fig


def plot_pareto_fronts_nsga2(p: DCParams, LF: float = 0.5, T_cw_min: float = 5.56, T_cw_max: float = 10.0, 
                             pop_size: int = 50, n_gen: int = 100, seed: int = 42) -> plt.Figure:
    """
    Approximate and plot the three-objective Pareto front using NSGA-II
    (via pymoo).  Equivalent to MATLAB's gamultiobj used in paper [84].

    Objectives  : maximise  COP,  CPR,  η_ex
    Constraint  : mean(u) = LF  — enforced via a repair operator that
                  rescales the utilisation vector after every crossover /
                  mutation so its mean stays exactly at LF.

    Parameters
    ----------
    p        : DCParams (layout, chiller model, etc.)
    LF       : mean utilisation equality constraint  [-]
    T_cw_min/max : chilled-water setpoint bounds      [°C]
    pop_size : NSGA-II population size (≥ 30 recommended)
    n_gen    : number of generations  (100 is a reasonable start;
               increase to 200–500 for a denser, more converged front)
    seed     : random seed for reproducibility

    Returns
    -------
    matplotlib Figure with the same 4-panel layout as plot_pareto_fronts:
      top row   — 3-D scatter in (COP, CPR, η_ex) space
      bottom row — three 2-D projections

    Requires
    --------
    pymoo  ≥ 0.5   (install with:  pip install pymoo)
    """

    Q_a_min  = 0.6  * p.N_cu
    Q_a_max  = 1.33 * p.N_cu
    n_var    = 2 + p.N_r
    x_lower  = np.array([T_cw_min, Q_a_min] + [0.0] * p.N_r)
    x_upper  = np.array([T_cw_max, Q_a_max] + [1.0] * p.N_r)

    # ── Problem: 3-objective, no explicit constraints (repair handles mean(u)) ─
    class _DCProblem(MooProblem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=3, xl=x_lower, xu=x_upper)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.full((len(X), 3), 1000.0)          # default = infeasible
            for i, x in enumerate(X):
                triple = run_simulation(list(x), p, "COP", False, return_all=True)
                if triple is not None:
                    cop, cpr, eta = triple[0], triple[1], triple[2]
                    F[i] = [-cop, -cpr, -eta]          # pymoo minimises → negate
            out["F"] = F

    # ── Repair: project u back onto the mean(u) = LF hyperplane ────────────
    class _MeanUtilisationRepair(MooRepair):
        def _do(self, problem, X, **kwargs):
            u = X[:, 2:].copy()                        # shape (pop, N_r)
            mean_u = u.mean(axis=1, keepdims=True)
            # 1. Scale proportionally so mean reaches LF
            scale = LF / np.where(mean_u > 1e-9, mean_u, 1e-9)
            u = np.clip(u * scale, 0.0, 1.0)
            # 2. Fine-tune: distribute the small residual evenly
            residual = u.mean(axis=1) - LF
            u -= residual[:, None] / p.N_r
            u = np.clip(u, 0.0, 1.0)
            X[:, 2:] = u
            return X

    # ── Run NSGA-II ─────────────────────────────────────────────────────────
    algorithm = NSGA2(pop_size=pop_size, repair=_MeanUtilisationRepair())

    print(f"\nNSGA-II Pareto search:  pop_size={pop_size},  n_gen={n_gen},  "
          f"n_var={n_var}  (seed={seed}) …")
    print(f"  Total evaluations ≈ {pop_size * n_gen:,}  "
          f"(each calls run_simulation once)")

    result = moo_minimize(
        _DCProblem(), algorithm,
        ('n_gen', n_gen),
        seed    = seed,
        verbose = True,
    )

    # ── Collect feasible solutions ───────────────────────────────────────────
    # result.F  = Pareto-optimal objective matrix from final generation (negated)
    # result.pop = full final population (background scatter)

    F_front  = result.F                                # (n_pareto, 3)
    ok_front = np.all(F_front < 999.0, axis=1)
    if ok_front.sum() < 2:
        raise RuntimeError(
            f"NSGA-II found only {ok_front.sum()} feasible solution(s) "
            "on the Pareto front.\n"
            "Try a larger pop_size / n_gen, or widen the T_cw / Q_a bounds."
        )
    values_par = -F_front[ok_front]                    # back to maximisation
    n_par      = len(values_par)

    # Background scatter: entire final generation
    try:
        F_pop    = result.pop.get("F")
        ok_pop   = np.all(F_pop < 999.0, axis=1)
        values_all = -F_pop[ok_pop] if ok_pop.sum() > 1 else values_par
    except Exception:
        values_all = values_par                        # graceful fallback

    n_all = len(values_all)
    print(f"\n  → Pareto front: {n_par} solutions  |  "
          f"final-gen feasible: {n_all} / {pop_size}\n")

    COP_a = values_all[:, 0];  CPR_a = values_all[:, 1];  eta_a = values_all[:, 2]
    COP_p = values_par[:, 0];  CPR_p = values_par[:, 1];  eta_p = values_par[:, 2]

    # ── Figure: same 4-panel layout as plot_pareto_fronts ───────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Three-Objective Pareto Front  —  NSGA-II  —  "
        r"$\max$ COP,  $\max$ CPR,  $\max$ $\eta_{\mathrm{ex}}$",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97, top=0.90, bottom=0.09,
        wspace=0.38, hspace=0.45
    )

    # — 3-D scatter (spans full top row) ————————————————————————————————
    ax3 = fig.add_subplot(gs[0, :], projection='3d')
    ax3.scatter(COP_a, CPR_a, eta_a,
                c='steelblue', alpha=0.30, s=28,
                label=f'Final-gen feasible ({n_all})')
    ax3.scatter(COP_p, CPR_p, eta_p,
                c='crimson', s=72, zorder=5, edgecolors='darkred', linewidths=0.5,
                label=f'Pareto front ({n_par})')
    ax3.set_xlabel('COP [–]',                   labelpad=7, fontsize=10)
    ax3.set_ylabel('CPR [–]',                   labelpad=7, fontsize=10)
    ax3.set_zlabel(r'$\eta_{\mathrm{ex}}$ [%]', labelpad=7, fontsize=10)
    ax3.set_title('3-D objective space  (NSGA-II)', fontsize=10, pad=4)
    ax3.legend(fontsize=8, loc='upper left')

    # — 2-D projections ———————————————————————————————————————————————
    proj_specs = [
        (gs[1, 0], COP_a, CPR_a, COP_p, CPR_p,
         'COP [–]',  'CPR [–]',                   'COP – CPR'),
        (gs[1, 1], COP_a, eta_a, COP_p, eta_p,
         'COP [–]',  r'$\eta_{\mathrm{ex}}$ [%]', r'COP – $\eta_{\mathrm{ex}}$'),
        (gs[1, 2], CPR_a, eta_a, CPR_p, eta_p,
         'CPR [–]',  r'$\eta_{\mathrm{ex}}$ [%]', r'CPR – $\eta_{\mathrm{ex}}$'),
    ]

    for gsp, xa, ya, xp, yp, xl_lbl, yl_lbl, title in proj_specs:
        ax = fig.add_subplot(gsp)
        ax.scatter(xa, ya, c='steelblue', alpha=0.30, s=22, zorder=2)
        ax.scatter(xp, yp, c='crimson',   s=55,       zorder=5,
                   edgecolors='darkred', linewidths=0.5)
        ax.set_xlabel(xl_lbl, fontsize=10)
        ax.set_ylabel(yl_lbl, fontsize=10)
        ax.set_title(title,   fontsize=10)
        ax.grid(True, alpha=0.3)

    return fig

def plot_pareto_fronts_multi_lf(
    p        : DCParams,
    LF_list  : list  = (0.2, 0.4, 0.6, 0.8),
    T_cw_min : float = 5.56,
    T_cw_max : float = 10.0,
    pop_size : int   = 50,
    n_gen    : int   = 100,
    seed     : int   = 42,
    save_dir : str | None = None,
) -> plt.Figure:
    """
    Compute NSGA-II Pareto fronts for each LF in LF_list and overlay them on
    a single 4-panel figure (3-D scatter + 3 × 2-D projections).

    Each load factor is assigned a distinct colour from the viridis palette;
    background scatter (full final generation) is drawn at low opacity and
    Pareto-optimal solutions at full opacity.

    If *save_dir* is given, every solution (Pareto front + feasible final-gen
    population) is re-evaluated to recover the full set of performance
    indicators and written to  <save_dir>/pareto_multi_lf.csv.  Columns:

        LF, is_pareto, T_cw_C, Q_a_m3s, u_rack_1 … u_rack_Nr,
        COP, CPR, eta_ex_pct, P_ch_W, P_f_W, P_it_W,
        T_hw_C, T_ca_C, T_ha_C, T_f_max_C, PLR

    Parameters
    ----------
    p        : DCParams (layout, chiller model, etc.)
    LF_list  : sequence of mean utilisations to sweep, e.g. [0.2, 0.4, 0.6, 0.8]
    T_cw_min/max : chilled-water setpoint bounds [°C]
    pop_size : NSGA-II population size (≥ 30 recommended)
    n_gen    : number of generations
    seed     : base random seed; incremented by 1 for each LF so runs differ
    save_dir : folder in which to write pareto_multi_lf.csv  (None = no CSV)

    Returns
    -------
    matplotlib Figure with 4-panel layout
    """
    Q_a_min = 0.6  * p.N_cu
    Q_a_max = 1.33 * p.N_cu
    n_var   = 2 + p.N_r
    x_lower = np.array([T_cw_min, Q_a_min] + [0.0] * p.N_r)
    x_upper = np.array([T_cw_max, Q_a_max] + [1.0] * p.N_r)

    LF_list = list(LF_list)
    colors  = plt.cm.viridis(np.linspace(0.10, 0.85, len(LF_list)))

    # ── Problem: physics only, LF-independent ────────────────────────────────
    class _DCProblem(MooProblem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=3, xl=x_lower, xu=x_upper)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.full((len(X), 3), 1000.0)
            for i, x in enumerate(X):
                triple = run_simulation(list(x), p, "COP", False, return_all=True)
                if triple is not None:
                    F[i] = [-triple[0], -triple[1], -triple[2]]
            out["F"] = F

    # ── Repair: captures LF via __init__ so loop closures are safe ───────────
    class _Repair(MooRepair):
        def __init__(self, lf: float):
            super().__init__()
            self._lf = lf

        def _do(self, problem, X, **kwargs):
            u        = X[:, 2:].copy()
            mean_u   = u.mean(axis=1, keepdims=True)
            scale    = self._lf / np.where(mean_u > 1e-9, mean_u, 1e-9)
            u        = np.clip(u * scale, 0.0, 1.0)
            residual = u.mean(axis=1) - self._lf
            u       -= residual[:, None] / p.N_r
            u        = np.clip(u, 0.0, 1.0)
            X[:, 2:] = u
            return X

    # ── Run NSGA-II for every LF ─────────────────────────────────────────────
    computed  = []    # list of (LF, values_par, values_all)
    csv_rows  = []    # accumulated for CSV export

    def _reeval_to_row(x_vec, LF_val: float, is_pareto: bool) -> dict:
        """Re-run physics for one solution and return a CSV row dict."""
        nan = float('nan')
        triple = run_simulation(list(x_vec), p, "COP", False, return_all=True)
        row = {
            "LF"         : LF_val,
            "is_pareto"  : is_pareto,
            "T_cw_C"     : float(x_vec[0]),
            "Q_a_m3s"    : float(x_vec[1]),
            **{f"u_rack_{i+1}": float(x_vec[2 + i]) for i in range(p.N_r)},
            "COP"        : triple[0]  if triple is not None else nan,
            "CPR"        : triple[1]  if triple is not None else nan,
            "eta_ex_pct" : triple[2]  if triple is not None else nan,
            "P_ch_W"     : triple[3]  if triple is not None else nan,
            "P_f_W"      : triple[4]  if triple is not None else nan,
            "P_it_W"     : triple[5]  if triple is not None else nan,
            "T_hw_C"     : triple[6]  if triple is not None else nan,
            "T_ca_C"     : triple[7]  if triple is not None else nan,
            "T_ha_C"     : triple[8]  if triple is not None else nan,
            "T_f_max_C"  : triple[9]  if triple is not None else nan,
            "PLR"        : triple[10] if triple is not None else nan,
        }
        return row

    for k, LF in enumerate(LF_list):
        algorithm = NSGA2(pop_size=pop_size, repair=_Repair(LF))
        print(f"\n── NSGA-II  LF = {LF:.1f}  "
              f"(pop={pop_size}, gen={n_gen}, seed={seed + k}) ──")
        print(f"   ≈ {pop_size * n_gen:,} total evaluations …")

        res = moo_minimize(
            _DCProblem(), algorithm,
            ('n_gen', n_gen),
            seed    = seed + k,
            verbose = True,
        )

        F_front  = res.F
        ok_front = np.all(F_front < 999.0, axis=1)
        if ok_front.sum() < 2:
            print(f"   WARNING: only {ok_front.sum()} feasible Pareto solution(s) "
                  f"for LF = {LF:.1f} — skipped.")
            continue
        values_par = -F_front[ok_front]
        X_par      = res.X[ok_front]            # decision vectors for Pareto front

        try:
            F_pop      = res.pop.get("F")
            X_pop      = res.pop.get("X")
            ok_pop     = np.all(F_pop < 999.0, axis=1)
            values_all = -F_pop[ok_pop] if ok_pop.sum() > 1 else values_par
            X_all      = X_pop[ok_pop]  if ok_pop.sum() > 1 else X_par
        except Exception:
            values_all = values_par
            X_all      = X_par

        print(f"   → Pareto front : {len(values_par)} solutions  |  "
              f"feasible final-gen : {len(values_all)} / {pop_size}")
        computed.append((LF, values_par, values_all))

        # ── Re-evaluate and accumulate CSV rows ───────────────────────────────
        if save_dir is not None:
            print(f"   Re-evaluating {len(X_par)} Pareto + "
                  f"{len(X_all)} population solutions for CSV …")
            for x in X_par:
                csv_rows.append(_reeval_to_row(x, LF, is_pareto=True))
            # background population (skip solutions already on the Pareto front)
            pareto_set = set(map(tuple, np.round(X_par, 8)))
            for x in X_all:
                if tuple(np.round(x, 8)) not in pareto_set:
                    csv_rows.append(_reeval_to_row(x, LF, is_pareto=False))

    if not computed:
        raise RuntimeError(
            "No feasible Pareto solutions found for any LF value. "
            "Try larger pop_size / n_gen or wider T_cw / Q_a bounds."
        )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if save_dir is not None and csv_rows:
        _ensure_dir(save_dir)
        csv_path = os.path.join(save_dir, "pareto_multi_lf.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"\n  → Pareto CSV saved : {csv_path}"
              f"  ({len(csv_rows)} rows, "
              f"{sum(r['is_pareto'] for r in csv_rows)} Pareto-front solutions)")

    # ── Figure ───────────────────────────────────────────────────────────────
    # Only Pareto-front solutions are plotted (no background population scatter).
    # Points are sorted by x-axis within each 2-D projection and connected with
    # a thin line so the front curve is immediately visible — matching the style
    # of paper [84].
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Three-Objective Pareto Front  —  NSGA-II  —  "
        r"$\max$ COP,  $\max$ CPR,  $\max$ $\eta_{\mathrm{ex}}$"
        "\nLF ∈ {" + ", ".join(f"{lf:.1f}" for lf, *_ in computed) + "}",
        fontsize=12, fontweight='bold'
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97, top=0.88, bottom=0.09,
        wspace=0.38, hspace=0.48
    )

    ax3  = fig.add_subplot(gs[0, :], projection='3d')
    ax2s = [fig.add_subplot(gs[1, j]) for j in range(3)]

    # (x-col, y-col, x-label, y-label, title)
    proj = [
        (0, 1, 'COP [–]', 'CPR [–]',                   'COP – CPR'),
        (0, 2, 'COP [–]', r'$\eta_{\mathrm{ex}}$ [%]', r'COP – $\eta_{\mathrm{ex}}$'),
        (1, 2, 'CPR [–]', r'$\eta_{\mathrm{ex}}$ [%]', r'CPR – $\eta_{\mathrm{ex}}$'),
    ]

    for (LF, values_par, _values_all), color in zip(computed, colors):
        n_par = len(values_par)
        COP_p, CPR_p, eta_p = values_par[:, 0], values_par[:, 1], values_par[:, 2]

        # 3-D — Pareto front only
        ax3.scatter(COP_p, CPR_p, eta_p,
                    color=color, alpha=1.0, s=40, zorder=5,
                    edgecolors='k', linewidths=0.3,
                    label=f"LF = {LF:.1f}  ({n_par} pts)")

        # 2-D projections — markers only (no connecting line: the front is not
        # monotone in every projection due to the third objective resolving
        # apparent domination — connecting points would create misleading zigzags)
        for ax2, (ix, iy, *_) in zip(ax2s, proj):
            ax2.scatter(values_par[:, ix], values_par[:, iy],
                        color=color, s=30, zorder=5, marker='^',
                        edgecolors='k', linewidths=0.3)

    ax3.set_xlabel('COP [–]',                   labelpad=7, fontsize=10)
    ax3.set_ylabel('CPR [–]',                   labelpad=7, fontsize=10)
    ax3.set_zlabel(r'$\eta_{\mathrm{ex}}$ [%]', labelpad=7, fontsize=10)
    ax3.set_title('3-D objective space  (NSGA-II)', fontsize=10, pad=4)
    ax3.legend(fontsize=8, loc='upper left',
               title='Load Factor', title_fontsize=8)

    for ax2, (ix, iy, xl, yl, title) in zip(ax2s, proj):
        ax2.set_xlabel(xl,   fontsize=10)
        ax2.set_ylabel(yl,   fontsize=10)
        ax2.set_title(title, fontsize=10)
        ax2.grid(True, alpha=0.3)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7b. CLIMATE SWEEP  (monthly ambient temperature × LF × city)
# ─────────────────────────────────────────────────────────────────────────────
#
#  For every (city, LF) pair, the COP optimisation of option "B" (SLSQP) or
#  "C" (basin hopping) is run once per month with p.T_amb set to that month's
#  mean ambient temperature.
#    • cold start (default): every month is an independent job starting from
#      the default x0 → all months of all cities/LFs run in parallel
#    • warm start: one chain per (city, LF), month m+1 starting from the
#      optimum of month m (faster, but the optimum depends on the path)
#    • every finished month is written to a checkpoint CSV, so an interrupted
#      sweep resumes where it stopped when it is launched again
#  When all jobs are done, the checkpoints are gathered into Excel workbooks.
#
# ─────────────────────────────────────────────────────────────────────────────

#: Monthly mean ambient temperatures, 2025  [°C]
CLIMATE_2025 = {
    "Milan"     : [ 6.4,  7.6, 12.2, 16.2, 18.8, 27.3, 25.9, 24.7, 21.3, 15.5,  9.0,  8.3],
    "Oslo"      : [-1.6, -1.2,  4.0,  9.2, 12.2, 15.4, 20.7, 16.4, 13.5,  7.9,  3.8,  2.4],
    "Marrakech" : [14.5, 15.6, 17.0, 20.7, 23.6, 28.6, 30.2, 31.7, 27.4, 24.3, 18.8, 12.7],
}
MONTHS      = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_2025   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _climate_ckpt_path(out_dir: str, city: str, p: DCParams, LF: float,
                       month_idx: int | None = None) -> str:
    """Checkpoint CSV of one (city, LF) chain, or of one month (cold start)."""
    stem = f"{city}_{_run_tag(p, LF)}"
    if month_idx is not None:
        stem += f"_m{month_idx + 1:02d}"
    return os.path.join(out_dir, "checkpoints", f"{stem}.csv")


def _climate_optimise_month(job: dict, m: int, x0s: list) -> tuple:
    """
    COP optimisation of one month (T_amb = that month's mean temperature).
    x0s is a list of starting points: one optimisation is run from each and
    the best (lowest objective) is kept.
    Appends the result row to job["ckpt"] and returns (row, x_opt).
    """
    import dataclasses
    import zlib

    p, city, LF = job["p"], job["city"], job["LF"]
    T_amb_C     = job["T_months"][m]
    T_cw_min, T_cw_max = job["T_cw_min"], job["T_cw_max"]
    Q_a_min,  Q_a_max  = job["Q_a_min"],  job["Q_a_max"]
    n_u         = p.N_r * p.N_z
    label       = f"[{city} | LF={LF}]"

    p_m         = dataclasses.replace(p, T_amb=T_amb_C + 273.15)
    bounds      = [(T_cw_min, T_cw_max), (Q_a_min, Q_a_max)] + [(0.0, 1.0)] * n_u
    constraints = {'type': 'eq', 'fun': lambda x: np.mean(x[2:]) - LF}

    t0 = datetime.datetime.now()
    best, n_iter = None, 0
    for k, x0 in enumerate(x0s):
        if job["method"] == "SLSQP":                         # = option "B"
            res = minimize(run_simulation, x0, args=(p_m, "COP", False),
                           method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': job["maxiter"]})
            cand = (res.fun, res.x, res.success, res.message)
            n_iter += res.nit
        elif job["method"] == "basinhopping":                # = option "C"
            # reproducible but different random hops for every (city, LF, month, start)
            np.random.seed(zlib.crc32(f"{city}_{LF}_{m}_{k}".encode()))
            minimizer_kwargs = {
                "method"      : "SLSQP",
                "bounds"      : bounds,
                "constraints" : constraints,
                "args"        : (p_m, "COP", False),
                "options"     : {'maxiter': min(200, 10 * len(x0))},
            }
            stepsizes = ([(T_cw_max - T_cw_min) * 0.20, (Q_a_max - Q_a_min) * 0.20]
                         + [0.15] * n_u)
            res = basinhopping(run_simulation, x0,
                               minimizer_kwargs = minimizer_kwargs,
                               niter            = job["niter"],
                               take_step        = PerVariableStep(stepsizes, bounds),
                               disp             = False)
            loc = res.lowest_optimization_result
            cand = (res.fun, res.x, loc.success, loc.message)
            n_iter += res.nit
        else:
            raise ValueError(f"method must be 'SLSQP' or 'basinhopping', got {job['method']!r}")
        if best is None or cand[0] < best[0]:
            best = cand + (k,)
    _, x_opt, success, msg, k_best = best
    if len(x0s) > 1:
        msg = f"{msg} (best of {len(x0s)} starts: #{k_best})"
    dt = (datetime.datetime.now() - t0).total_seconds()

    out = run_simulation(x_opt, p_m, "COP", plot=False,
                         return_all=True, allow_infeasible=True)
    nan = float('nan')
    (COP, CPR, eta, P_ch, P_f, P_it, T_hw, T_ca, T_ha, T_f_max, PLR,
     Q_coil, Q_ch_max, psi_ch, psi_hx, psi_f, psi_in) = \
        out[:17] if out is not None else (nan,) * 17
    psi_tot = psi_ch + psi_hx + psi_f

    h        = DAYS_2025[m] * 24.0
    P_cool   = P_ch + P_f
    u_zone   = np.asarray(x_opt[2:], dtype=float)
    u_rack   = u_zone.reshape(p.N_r, p.N_z).mean(axis=1)
    feasible = (out is not None
                and T_f_max <= p.T_server_red + 0.05
                and Q_coil <= Q_ch_max * (1 + 1e-3)          # chiller can deliver the load
                and p.PLR_min <= PLR <= p.PLR_max)

    row = {
        "city"             : city,
        "month_idx"        : m,
        "month"            : MONTHS[m],
        "hours"            : h,
        "T_amb_C"          : T_amb_C,
        "T_amb_in_chiller_range": bool(p.T_amb_min <= T_amb_C <= p.T_amb_max),
        "LF"               : LF,
        "N_packs"          : p.n_packs,
        "N_cu"             : p.N_cu,
        "N_r"              : p.N_r,
        "N_z"              : p.N_z,
        "method"           : job["method"],
        "warm_start"       : bool(job["warm_start"]),
        "T_cw_opt_C"       : float(x_opt[0]),
        "Q_a_opt_m3s"      : float(x_opt[1]),
        "Q_a_total_m3s"    : float(x_opt[1]) * p.n_packs,
        "COP"              : COP,
        "CPR"              : CPR,
        "eta_ex_pct"       : eta,
        "P_IT_kW"          : P_it   / 1e3,
        "P_ch_kW"          : P_ch   / 1e3,
        "P_f_kW"           : P_f    / 1e3,
        "P_cool_kW"        : P_cool / 1e3,
        "pPUE"             : (P_it + P_cool) / P_it if P_it > 0 else nan,
        "T_hw_C"           : T_hw,
        "T_ca_C"           : T_ca,
        "T_ha_C"           : T_ha,
        "T_f_max_C"        : T_f_max,
        "PLR"              : PLR,
        "Q_cool_load_kW"   : Q_coil   / 1e3,
        "Q_ch_max_kW"      : Q_ch_max / 1e3,
        "Q_unmet_kW"       : max(0.0, Q_coil - Q_ch_max) / 1e3,
        "Ex_in_kW"         : psi_in  / 1e3,
        "Ex_dest_chiller_kW": psi_ch / 1e3,
        "Ex_dest_HX_kW"    : psi_hx  / 1e3,
        "Ex_dest_fans_kW"  : psi_f   / 1e3,
        "Ex_dest_total_kW" : psi_tot / 1e3,
        "Ex_dest_chiller_pct": 100 * psi_ch / psi_tot if psi_tot > 0 else nan,
        "Ex_dest_HX_pct"   : 100 * psi_hx / psi_tot if psi_tot > 0 else nan,
        "Ex_dest_fans_pct" : 100 * psi_f  / psi_tot if psi_tot > 0 else nan,
        "E_IT_MWh"         : P_it   * h / 1e6,
        "E_ch_MWh"         : P_ch   * h / 1e6,
        "E_f_MWh"          : P_f    * h / 1e6,
        "E_cool_MWh"       : P_cool * h / 1e6,
        "feasible"         : bool(feasible),
        "converged"        : bool(success),
        "n_iter"           : int(n_iter),
        "optimizer_msg"    : str(msg),
        "runtime_s"        : dt,
        **{f"u_rack_{i+1}": float(u_rack[i]) for i in range(p.N_r)},
        **{f"u_z_{k}": float(u_zone[k]) for k in range(n_u)},
    }
    ckpt = job["ckpt"]
    _ensure_dir(os.path.dirname(ckpt))
    pd.DataFrame([row]).to_csv(ckpt, mode="a", index=False,
                               header=not os.path.exists(ckpt))
    print(f"{label} {MONTHS[m]} (T_amb={T_amb_C:5.1f} °C): COP={COP:.3f}  "
          f"T_cw={x_opt[0]:.2f} °C  P_cool={P_cool/1e3:.2f} kW  "
          f"{'ok' if success else 'NOT converged'}"
          f"{'' if feasible else '  [infeasible]'}  ({dt/60:.1f} min)", flush=True)
    return row, list(x_opt)


def side_loaded_u(p: DCParams, LF: float, from_centre: bool = False) -> np.ndarray:
    """
    Per-zone utilisation (N_r·N_z) with the load packed next to the IRCs:
    racks are filled to u = 1 in order of distance to the nearest IRC column,
    racks at equal distance share the remainder, so that mean(u) = LF.
    e.g. [0, 8, 0], LF = 0.2 → racks 1 and 8 at 0.8, the others at 0.
    from_centre = True fills the racks farthest from the IRCs first instead
    (e.g. LF = 0.2 → racks 4 and 5 at 0.8).
    """
    cols  = np.arange(len(p.is_irc))
    irc   = cols[p.is_irc]
    dist  = np.array([np.min(np.abs(irc - c)) for c in cols[~p.is_irc]])
    u     = np.zeros(p.N_r)
    left  = LF * p.N_r                                   # rack-units of load to place
    order = np.unique(dist)
    for d in (order[::-1] if from_centre else order):
        idx    = np.flatnonzero(dist == d)
        share  = min(1.0, left / len(idx))
        u[idx] = share
        left  -= share * len(idx)
        if left <= 1e-12:
            break
    return np.repeat(u, p.N_z)


def _climate_x0(job: dict, m: int | None = None) -> list:
    """
    Default starting point of month m (T_amb = that month's mean temperature).
      u0 = "sides"   : side-loaded u; T_cw from a 1 K scan over [T_cw_min,
                       T_cw_max], each with the smallest Q_a that keeps
                       max T_f ≈ T_server_red − 0.2 K; the best COP is kept
      u0 = "uniform" : = option "B" x0 for COP: T_cw_max, mid Q_a, uniform u
    The uniform u is a stationary point of SLSQP (COP is flat in u when
    mean(u) is fixed, and the quadratic T_f penalty has zero slope at
    max T_f = T_server_red), so the u never leave it.
    SLSQP does not move from an infeasible start either (the large T_f penalty
    makes the finite-difference gradient unusable), and it hardly moves along
    the T_f limit, where COP is flat in T_cw — hence the feasible, scanned start.
    """
    import dataclasses

    p = job["p"]
    if m is not None:
        p = dataclasses.replace(p, T_amb=job["T_months"][m] + 273.15)
    if job.get("u0", "sides") == "uniform":
        return ([job["T_cw_max"], (job["Q_a_min"] + job["Q_a_max"]) / 2]
                + [job["LF"]] * (p.N_r * p.N_z))
    return scan_x0(p, job["LF"], "COP", "sides", job["T_cw_min"], job["T_cw_max"],
                   job["Q_a_min"], job["Q_a_max"], n_Q=1)


def start_u(p: DCParams, LF: float, u0: str) -> np.ndarray:
    """Per-zone u of a starting point: "sides", "centre" or "uniform"."""
    if u0 == "uniform":
        return np.full(p.N_r * p.N_z, LF)
    return side_loaded_u(p, LF, from_centre=(u0 == "centre"))


def scan_x0(p: DCParams, LF: float, answer: str, u0: str,
            T_cw_min: float, T_cw_max: float, Q_a_min: float, Q_a_max: float,
            n_Q: int = 4) -> list:
    """
    Feasible starting point for SLSQP with the u pattern u0 (see start_u).
    T_cw is scanned in 1 K steps over [T_cw_min, T_cw_max]; for each T_cw,
    n_Q values of Q_a from the smallest one keeping max T_f ≈ T_server_red − 0.2 K
    up to Q_a_max are tried (n_Q = 1: that smallest Q_a only, which is where
    COP is best).  The feasible point (max T_f ≤ T_server_red, PLR in range)
    with the best objective *answer* is returned.
    When T_server_min applies to *answer*, the Q_a range stops at the airflow
    where min T_f ≈ T_server_min + 0.2 K, and colder points are rejected.
    Nothing here assumes which T_cw / Q_a the objective prefers: COP ends up
    at low airflow on the T_f limit, CPR at cold water and high airflow.
    """
    u        = list(start_u(p, LF, u0))
    k        = {"COP": 0, "CPR": 1, "eta_ex": 2}[answer]
    T_target = p.T_server_red - 0.2
    T_cold   = (p.T_server_min + 0.2
                if p.T_server_min is not None and answer in p.T_server_min_for else None)

    def _sim(T_cw, Q_a):
        return run_simulation([T_cw, Q_a] + u, p, answer, False,
                              return_all=True, allow_infeasible=True)

    best = None
    n_T  = int(round(T_cw_max - T_cw_min)) + 1
    for T_cw in np.linspace(T_cw_min, T_cw_max, n_T):
        def _g(Q_a):
            out = _sim(T_cw, Q_a)
            return (out[9] if out is not None else 1e3) - T_target
        if _g(Q_a_min) <= 0:
            Q_feas = Q_a_min
        elif _g(Q_a_max) >= 0:
            continue                                     # T_f limit not reachable
        else:
            Q_feas = brentq(_g, Q_a_min, Q_a_max, xtol=1e-3)
        Q_top = Q_a_max
        if T_cold is not None:
            def _h(Q_a):
                out = _sim(T_cw, Q_a)
                return (out[17] if out is not None else -1e3) - T_cold
            if _h(Q_feas) < 0:
                continue                                 # already too cold at the smallest Q_a
            if _h(Q_a_max) < 0:
                Q_top = brentq(_h, Q_feas, Q_a_max, xtol=1e-3)
        for Q_a in np.linspace(Q_feas, Q_top, n_Q):
            out = _sim(T_cw, Q_a)
            if (out is None or out[9] > p.T_server_red
                    or (T_cold is not None and out[17] < p.T_server_min)
                    or not (p.PLR_min <= out[10] <= p.PLR_max)):
                continue                                 # too hot / too cold, or chiller cannot deliver
            if best is None or out[k] > best[0]:
                best = (out[k], float(T_cw), float(Q_a))
    if best is None:                                     # nothing feasible: coldest water, max airflow
        return [T_cw_min, Q_a_max] + u
    return [best[1], best[2]] + u


#: Objective scale used by optimise_multistart.  SLSQP starts from an identity
#: Hessian and stops when f changes by < ftol (1e-6), so the gradients must be
#: O(0.1–1): typical |∂f/∂x| ≈ 0.01–2 for COP, 0.2–25 for η_ex [%] and only
#: 0.001–0.03 for CPR (which then stopped at x0 after one step).
OBJ_SCALE = {"COP": 1.0, "eta_ex": 0.1, "CPR": 100.0}


def optimise_multistart(p: DCParams, LF: float, answer: str, starts: list,
                        T_cw_min: float, T_cw_max: float,
                        Q_a_min: float, Q_a_max: float,
                        maxiter: int = 500, disp: bool = True):
    """
    SLSQP from one scanned, feasible x0 per u pattern in *starts* (scan_x0),
    on the objective scaled by OBJ_SCALE[answer]; the best result is kept.
    Returns (result, per_start) with result.fun unscaled (= run_simulation
    value) and per_start = [(u0, x0, objective, T_cw, Q_a, success), …].
    """
    scale  = OBJ_SCALE[answer]
    bounds = [(T_cw_min, T_cw_max), (Q_a_min, Q_a_max)] + [(0.0, 1.0)] * (p.N_r * p.N_z)
    cons   = {'type': 'eq', 'fun': lambda x: np.mean(x[2:]) - LF}
    best, per_start = None, []
    for u0 in starts:
        t0 = time.time()
        x0 = scan_x0(p, LF, answer, u0, T_cw_min, T_cw_max, Q_a_min, Q_a_max)
        print(f"[LF={LF}] start '{u0}': x0 T_cw = {x0[0]:.2f} °C, Q_a = {x0[1]:.3f} m³/s "
              f"({time.time() - t0:.0f} s scan) → SLSQP …", flush=True)
        res = minimize(lambda x: scale * run_simulation(x, p, answer, False), x0,
                       method='SLSQP', bounds=bounds, constraints=cons,
                       options={'disp': disp, 'maxiter': maxiter})
        res.fun = res.fun / scale
        per_start.append((u0, x0, -res.fun, res.x[0], res.x[1], res.success))
        if best is None or res.fun < best.fun:
            best = res
            best.message = f"{res.message} (best of {len(starts)} starts: {u0})"
    return best, per_start


def _climate_chain(job: dict) -> str:
    """
    Warm-start mode: the months of one (city, LF) chain in calendar order,
    month m+1 starting from the optimum of month m.
    Top-level function so it can be sent to a worker process.
    """
    warnings.filterwarnings("ignore")
    p, label = job["p"], f"[{job['city']} | LF={job['LF']}]"
    n_u      = p.N_r * p.N_z
    x_start  = None

    # ── Resume from checkpoint ────────────────────────────────────────────
    done = set()
    if os.path.exists(job["ckpt"]):
        prev = pd.read_csv(job["ckpt"])
        done = set(prev["month_idx"].astype(int))
        if len(prev):
            last = prev.sort_values("month_idx").iloc[-1]
            x_start = ([last["T_cw_opt_C"], last["Q_a_opt_m3s"]]
                       + [last[f"u_z_{k}"] for k in range(n_u)])
            print(f"{label} resuming: {len(done)} month(s) already in checkpoint", flush=True)

    for m in job["months"]:
        if m in done:
            continue
        # restart = True: also start from the default x0 and keep the best, so
        # that a chain stuck in a poor local optimum can leave it
        if x_start is None:
            x0s = [_climate_x0(job, m)]
        elif job.get("restart", True):
            x_default = _climate_x0(job, m)
            x0s = [x_start] + ([] if np.allclose(x_start, x_default) else [x_default])
        else:
            x0s = [x_start]
        _, x_start = _climate_optimise_month(job, m, x0s)
    return label


def _climate_month(job: dict) -> str:
    """
    Cold-start mode: one month, starting from the default x0.
    Top-level function so it can be sent to a worker process.
    """
    warnings.filterwarnings("ignore")
    m = job["months"][0]
    _climate_optimise_month(job, m, [_climate_x0(job, m)])
    return f"[{job['city']} | LF={job['LF']}] {MONTHS[m]}"


def _climate_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Annual totals per (city, LF) from the monthly rows."""
    g = df.groupby(["city", "LF"], sort=False)
    ann = g.agg(
        T_amb_mean_C   = ("T_amb_C",    "mean"),
        T_cw_mean_C    = ("T_cw_opt_C", "mean"),
        E_IT_MWh       = ("E_IT_MWh",   "sum"),
        E_ch_MWh       = ("E_ch_MWh",   "sum"),
        E_f_MWh        = ("E_f_MWh",    "sum"),
        E_cool_MWh     = ("E_cool_MWh", "sum"),
        eta_ex_mean_pct= ("eta_ex_pct", "mean"),
        CPR_mean       = ("CPR",        "mean"),
        n_months       = ("month",      "count"),
        n_infeasible   = ("feasible",   lambda s: int((~s.astype(bool)).sum())),
        n_not_converged= ("converged",  lambda s: int((~s.astype(bool)).sum())),
    ).reset_index()
    ann["COP_annual"]  = ann["E_IT_MWh"] / ann["E_cool_MWh"]
    ann["pPUE_annual"] = (ann["E_IT_MWh"] + ann["E_cool_MWh"]) / ann["E_IT_MWh"]
    return ann


def export_climate_excel(out_dir: str) -> list:
    """
    Gather every checkpoint CSV in <out_dir>/checkpoints into Excel files:
      • <city>.xlsx                     — one per city
            sheets: Monthly, Annual, u_distribution
      • climate_comparison_all.xlsx     — all cities together
            sheets: Monthly, Annual, plus month × (city, LF) pivots of
            COP, P_cool_kW, E_cool_MWh, T_cw_opt_C, eta_ex_pct
    Returns the list of written paths.
    """
    ck_dir = os.path.join(out_dir, "checkpoints")
    files  = [os.path.join(ck_dir, f) for f in os.listdir(ck_dir) if f.endswith(".csv")] \
             if os.path.isdir(ck_dir) else []
    if not files:
        print("  no checkpoint found → nothing to export")
        return []

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["city", "LF", "month_idx"]).reset_index(drop=True)
    u_cols    = [c for c in df.columns if c.startswith("u_z_")]
    main_cols = [c for c in df.columns if c not in u_cols]

    written = []
    for city, d in df.groupby("city", sort=False):
        path = os.path.join(out_dir, f"{city}.xlsx")
        with pd.ExcelWriter(path) as xw:
            d[main_cols].to_excel(xw, sheet_name="Monthly", index=False)
            _climate_annual(d).to_excel(xw, sheet_name="Annual", index=False)
            d[["city", "LF", "month"] + u_cols].to_excel(
                xw, sheet_name="u_distribution", index=False)
        written.append(path)

    path = os.path.join(out_dir, "climate_comparison_all.xlsx")
    with pd.ExcelWriter(path) as xw:
        df[main_cols].to_excel(xw, sheet_name="Monthly", index=False)
        _climate_annual(df).to_excel(xw, sheet_name="Annual", index=False)
        for var in ["COP", "P_cool_kW", "E_cool_MWh", "T_cw_opt_C", "eta_ex_pct"]:
            pv = df.pivot_table(index=["month_idx", "month"],
                                columns=["city", "LF"], values=var)
            pv.to_excel(xw, sheet_name=var[:31])
    written.append(path)

    for w in written:
        print(f"  → saved {w}")
    return written


def run_climate_sweep(p: DCParams, cities: list, LF_list: list,
                      T_cw_min: float, T_cw_max: float,
                      Q_a_min: float, Q_a_max: float,
                      out_dir: str, n_workers: int = 14,
                      warm_start: bool = False, method: str = "SLSQP",
                      months: list | None = None,
                      maxiter: int = 500, niter: int = 50,
                      u0: str = "sides", restart: bool = True) -> list:
    """
    Monthly COP optimisation for every city in *cities* (keys of CLIMATE_2025),
    every LF in *LF_list* and every month in *months*, then export to Excel.

    warm_start = False : every month is an independent job starting from the
                         default x0 (cold start) → all jobs run in parallel
    warm_start = True  : one job per (city, LF); month m+1 starts from the
                         optimum of month m
    method     : "SLSQP" (as option "B", maxiter) or "basinhopping"
                 (as option "C", niter hops)
    months     : month indices 0–11 or names ("Jan" …); None = all 12
    n_workers  : parallel processes; 1 runs everything in this process
    u0         : default x0 — "sides" (load next to the IRCs, best T_cw of a
                 1 K scan along max T_f ≈ T_server_red) or "uniform"
                 (T_cw_max, mid Q_a, u = LF everywhere)
    restart    : warm start only — each month also starts from the default
                 x0 and keeps the better of the two optima
    """
    _ensure_dir(out_dir)
    if months is None:
        months = list(range(12))
    months = sorted(MONTHS.index(m) if isinstance(m, str) else int(m) for m in months)

    base = dict(p=p, T_cw_min=T_cw_min, T_cw_max=T_cw_max,
                Q_a_min=Q_a_min, Q_a_max=Q_a_max, warm_start=warm_start,
                method=method, maxiter=maxiter, niter=niter, u0=u0, restart=restart)
    if warm_start:
        jobs = [dict(base, city=c, LF=LF, T_months=CLIMATE_2025[c], months=months,
                     ckpt=_climate_ckpt_path(out_dir, c, p, LF))
                for c in cities for LF in LF_list]
        worker = _climate_chain
    else:
        jobs = [dict(base, city=c, LF=LF, T_months=CLIMATE_2025[c], months=[m],
                     ckpt=_climate_ckpt_path(out_dir, c, p, LF, m))
                for c in cities for LF in LF_list for m in months]
        n_all = len(jobs)
        jobs  = [j for j in jobs if not os.path.exists(j["ckpt"])]   # resume
        if n_all > len(jobs):
            print(f"  resuming: {n_all - len(jobs)}/{n_all} month(s) already done")
        worker = _climate_month

    n_opt     = len(cities) * len(LF_list) * len(months)
    n_workers = max(1, min(n_workers, len(jobs))) if jobs else 1
    print(f"Climate sweep ({method}, {'warm' if warm_start else 'cold'} start): "
          f"{len(cities)} city(ies) × {len(LF_list)} LF × {len(months)} month(s) "
          f"= {n_opt} optimisations, {n_workers} parallel worker(s)")

    if n_workers == 1:
        for j in jobs:
            worker(j)
    else:
        # one BLAS/OpenMP thread per worker so the processes don't compete for cores
        # (inherited by the worker processes when they import numpy)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(worker, j) for j in jobs]
                for k, f in enumerate(as_completed(futs), 1):
                    print(f"{f.result()} finished  ({k}/{len(jobs)} jobs)", flush=True)
        except (BrokenProcessPool, AttributeError, OSError) as e:
            # e.g. launched from an interactive console that cannot spawn workers
            print(f"  parallel run failed ({e!r}) → continuing sequentially")
            for j in jobs:
                if warm_start or not os.path.exists(j["ckpt"]):
                    worker(j)       # finished months are skipped (checkpoint)

    return export_climate_excel(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LAYOUT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def make_layout(spec: list) -> tuple:
    """
    Build (is_irc, N_r, N_cu) from a compact layout spec.

    Each element of *spec* is either:
      0           →  one IRC column
      positive N  →  N consecutive rack columns

    Examples
    --------
    [0, 10, 0]        IRC_L | 10 racks | IRC_R          (12 columns)
    [0, 5, 0, 5, 0]   IRC_L | 5 racks | IRC_C | 5 racks | IRC_R (13 columns)
    [0, 4, 0, 3, 0]   IRC_L | 4 racks | IRC_C | 3 racks | IRC_R (10 columns)
    [0, 7, 0]         IRC_L | 7 racks | IRC_R           (9 columns)
    """
    is_irc = []
    for item in spec:
        if item == 0:
            is_irc.append(True)
        else:
            is_irc.extend([False] * int(item))
    is_irc = np.array(is_irc, dtype=bool)
    N_r  = int(np.sum(~is_irc))
    N_cu = int(np.sum(is_irc))
    return is_irc, N_r, N_cu



# ─────────────────────────────────────────────────────────────────────────────
# 9.  OPTIMIZER HELPER
# ─────────────────────────────────────────────────────────────────────────────

class PerVariableStep:
    """
    Basin-hopping step with independent step sizes per variable group.

    For the u variables a zero-mean adjustment is applied so that
    mean(u) stays approximately equal to LF before SLSQP corrects it
    exactly — this avoids wasting local-optimizer evaluations dragging
    the starting point back to the feasible hyperplane.
    """
    def __init__(self, stepsizes, bounds):
        self.stepsizes = np.asarray(stepsizes, dtype=float)
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])

    def __call__(self, x):
        delta = np.array([np.random.uniform(-s, s)
                          for s in self.stepsizes])

        # Keep sum(u) ≈ N_r × LF so the jump starts near the constraint
        # surface; SLSQP will enforce it exactly during local search
        delta[2:] -= delta[2:].mean()

        return np.clip(x + delta, self.lb, self.ub)


def _bh_job(job: dict) -> dict:
    """
    Basin-hopping (option "C") for one LF, from the scanned x0 of the u pattern
    job["u0"] (scan_x0), or from u = LF everywhere if u0 is None.
    Top-level function so it can be sent to a worker process.
    """
    warnings.filterwarnings("ignore")
    p, LF, answer = job["p"], job["LF"], job["answer"]
    T_cw_min, T_cw_max = job["T_cw_min"], job["T_cw_max"]
    Q_a_min,  Q_a_max  = job["Q_a_min"],  job["Q_a_max"]
    scale  = OBJ_SCALE[answer] if job["scaled"] else 1.0
    n_u    = p.N_r * p.N_z
    bounds = [(T_cw_min, T_cw_max), (Q_a_min, Q_a_max)] + [(0.0, 1.0)] * n_u
    cons   = {'type': 'eq', 'fun': lambda x: np.mean(x[2:]) - LF}
    if job["seed"] is not None:
        np.random.seed(job["seed"])     # PerVariableStep draws from np.random

    t0 = time.time()
    if job["u0"] is None:
        x0 = [T_cw_max, (Q_a_min + Q_a_max) / 2] + [LF] * n_u
    else:
        x0 = scan_x0(p, LF, answer, job["u0"], T_cw_min, T_cw_max, Q_a_min, Q_a_max)
    print(f"[LF={LF}] x0 T_cw = {x0[0]:.2f} °C, Q_a = {x0[1]:.3f} m³/s "
          f"({time.time() - t0:.0f} s scan) → basin-hopping, {job['niter']} hops …", flush=True)

    # Rule of thumb: allow ~10 iterations per variable so SLSQP can reach
    # the feasible region from a perturbed starting point.
    minimizer_kwargs = {
        "method"      : "SLSQP",
        "bounds"      : bounds,
        "constraints" : cons,
        "options"     : {'maxiter': min(200, 10 * len(x0))},
    }
    stepsizes = ([(T_cw_max - T_cw_min) * 0.20, (Q_a_max - Q_a_min) * 0.20]
                 + [0.15] * n_u)
    best = [np.inf]

    def report(x, f, accept):
        # also called once after the initial minimisation (k = 0)
        best[0] = min(best[0], f)
        step = "start  " if report.k == 0 else f"hop {report.k:3d}/{job['niter']}"
        report.k += 1
        print(f"[LF={LF}] {step}: {answer} = {-f / scale:.4f} "
              f"({'accepted' if accept else 'rejected'}), best = {-best[0] / scale:.4f}  "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    report.k = 0

    res = basinhopping(lambda x: scale * run_simulation(x, p, answer, False), x0,
                       minimizer_kwargs=minimizer_kwargs, niter=job["niter"],
                       take_step=PerVariableStep(stepsizes, bounds),
                       callback=report, rng=job["seed"])
    loc = res.lowest_optimization_result
    return dict(LF=LF, x=np.asarray(res.x), obj=-res.fun / scale, x0=x0,
                success=bool(loc.success), message=str(loc.message),
                minutes=(time.time() - t0) / 60)


def run_basinhopping_lfs(jobs: list, n_workers: int) -> list:
    """
    Run _bh_job for every job (one per LF), in parallel processes when
    n_workers > 1.  Returns the results sorted by LF.
    """
    n_workers = max(1, min(n_workers, len(jobs)))
    if n_workers == 1:
        return [_bh_job(j) for j in jobs]

    # one BLAS/OpenMP thread per worker so the processes don't compete for cores
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    results = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_bh_job, j) for j in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                r = f.result()
                results.append(r)
                print(f"[LF={r['LF']}] finished in {r['minutes']:.0f} min  "
                      f"({k}/{len(jobs)} LF)", flush=True)
    except (BrokenProcessPool, AttributeError, OSError) as e:
        # e.g. launched from an interactive console that cannot spawn workers
        print(f"  parallel run failed ({e!r}) → continuing sequentially")
        done = {r["LF"] for r in results}
        results += [_bh_job(j) for j in jobs if j["LF"] not in done]
    return sorted(results, key=lambda r: r["LF"])

# ─────────────────────────────────────────────────────────────────────────────
# 10.  RUN OPTIONS  (one function per value of SIMU, called from the main block)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunSetup:
    """
    What every run option needs, built in the main block from the USER
    SETTINGS.  The bounds refer to ONE pack.
    """
    p:        DCParams     # layout, physical constants and number of packs
    simu:     str          # run option letter (written to performances.csv)
    answer:   str          # objective: "COP", "eta_ex" or "CPR"
    LF:       float        # load factor of the single-LF options
    T_cw_min: float        # [°C]   bounds of the chilled-water temperature
    T_cw_max: float        # [°C]
    Q_a_min:  float        # [m³/s] bounds of the airflow of one pack
    Q_a_max:  float        # [m³/s]
    fig_dir:  str          # output folder of the objective (figures + performances.csv)


def record_performance(setup: RunSetup, LF: float, x_opt, converged, opt_msg="") -> None:
    """Re-evaluate at x_opt, log all KPIs + metadata to performances.csv."""
    p, answer = setup.p, setup.answer
    # allow_infeasible: a point just over a limit (e.g. PLR = 1.0002) is
    # still logged, with the violation noted in optimizer_msg
    triple = run_simulation(x_opt, p, answer, plot=False, return_all=True,
                            allow_infeasible=True)
    too_cold = (p.T_server_min is not None and answer in p.T_server_min_for
                and triple is not None and triple[17] < p.T_server_min - 1e-3)
    if triple is not None and (triple[9] > p.T_server_red + 1e-3 or too_cold
                               or not (p.PLR_min <= triple[10] <= p.PLR_max)):
        opt_msg = (f"{opt_msg} [infeasible: max T_f = {triple[9]:.2f} °C, "
                   f"min T_f = {triple[17]:.2f} °C, PLR = {triple[10]:.4f}]")
    nan = float('nan')
    cop       = triple[0]  if triple is not None else nan
    cpr       = triple[1]  if triple is not None else nan
    eta       = triple[2]  if triple is not None else nan
    P_ch      = triple[3]  if triple is not None else nan
    P_f       = triple[4]  if triple is not None else nan
    P_it      = triple[5]  if triple is not None else nan
    T_hw_C    = triple[6]  if triple is not None else nan
    T_ca_C    = triple[7]  if triple is not None else nan
    T_ha_C    = triple[8]  if triple is not None else nan
    T_f_max   = triple[9]  if triple is not None else nan
    PLR       = triple[10] if triple is not None else nan

    u_vals = np.asarray(x_opt[2:], dtype=float)
    row = {
        "timestamp"      : datetime.datetime.now().isoformat(timespec="seconds"),
        "simu"           : setup.simu,
        "answer"         : answer,
        "method"         : "standard",
        "LF"             : LF,
        "N_packs"        : p.n_packs,
        "N_cu"           : p.N_cu,
        "N_r"            : p.N_r,
        "N_z"            : p.N_z,
        "T_cw_opt_C"     : float(x_opt[0]),
        "Q_a_opt_m3s"    : float(x_opt[1]),
        "u_mean"         : float(np.mean(u_vals)),
        **{f"u_rack_{i+1}": float(u_vals[i]) for i in range(len(u_vals))},
        "COP"            : cop,
        "CPR"            : cpr,
        "eta_ex_pct"     : eta,
        "P_ch_W"         : P_ch,
        "P_f_W"          : P_f,
        "P_it_W"         : P_it,
        "T_hw_C"         : T_hw_C,
        "T_ca_C"         : T_ca_C,
        "T_ha_C"         : T_ha_C,
        "T_f_max_C"      : T_f_max,
        "PLR"            : PLR,
        **_perf_extra(triple),
        "converged"      : converged,
        "optimizer_msg"  : opt_msg,
    }
    save_performances(answer, row)


def _print_objective(answer: str, value: float) -> None:
    """Print the optimum of the objective *answer*, with its unit."""
    if answer == "COP":
        print(f"Maximum COP             : {value:.4f}")
    elif answer == "eta_ex":
        print(f"Maximum exergy eff.     : {value:.2f} %")
    else:
        print(f"Maximum CPR             : {value:.4f}")


def run_option_A(setup: RunSetup, x0: list) -> None:
    """Option A: forward simulation of the single operating point x0."""
    print("Running forward simulation …")
    run_simulation(x0, setup.p, setup.answer, plot=True,
                   save_dir=setup.fig_dir, run_tag=_run_tag(setup.p, setup.LF))
    record_performance(setup, setup.LF, x0, converged=True, opt_msg="forward simulation")


def run_option_B(setup: RunSetup, LF_list: list, starts: list | None) -> None:
    """
    Option B: SLSQP optimisation of setup.answer for every LF in LF_list.
    starts = list of u patterns ("sides", "centre", "uniform"): one SLSQP run
    per pattern from a scanned feasible x0 (optimise_multistart), best kept.
    starts = None: one SLSQP run from mid T_cw (T_cw_min for CPR), mid Q_a,
    u = LF everywhere, on the unscaled objective.
    """
    p, answer = setup.p, setup.answer
    summary = []
    for LF in LF_list:
        run_tag = _run_tag(p, LF)
        if starts:
            # feasible scanned x0 per u pattern, scaled objective, best kept
            # (SLSQP does not leave a uniform-u or an infeasible start, and
            # stops after one step when the gradients are tiny, as for CPR)
            print()
            result, per_start = optimise_multistart(p, LF, answer, starts,
                                                    setup.T_cw_min, setup.T_cw_max,
                                                    setup.Q_a_min, setup.Q_a_max)
            print(f"\n[LF={LF}] {answer} per start:")
            for u0, x0_k, obj, T_k, Q_k, ok in per_start:
                print(f"  {u0:8s} x0 ({x0_k[0]:5.2f} °C, {x0_k[1]:.3f})  →  {answer} = {obj:.4f}"
                      f"  T_cw = {T_k:.2f} °C  Q_a = {Q_k:.3f}  converged = {ok}")
        else:
            # COP / eta_ex: start from mid T_cw;  CPR: from cold water (low T_f)
            T_cw_x0 = (setup.T_cw_min if answer == "CPR"
                       else (setup.T_cw_max + setup.T_cw_min) / 2)
            x0      = [T_cw_x0, (setup.Q_a_min + setup.Q_a_max) / 2] + [LF] * p.N_r * p.N_z
            bounds  = ([(setup.T_cw_min, setup.T_cw_max), (setup.Q_a_min, setup.Q_a_max)]
                       + [(0.0, 1.0)] * p.N_r * p.N_z)
            print(f"[LF={LF}] Starting SLSQP optimisation from T_cw = {x0[0]:.2f} °C, "
                  f"Q_a = {x0[1]:.3f} m³/s …")
            result = minimize(
                run_simulation,
                x0,
                args=(p, answer, False),
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': lambda x, LF=LF: np.mean(x[2:]) - LF},
                options={'disp': True, 'maxiter': 500}
            )

        print(f"\n--- OPTIMISATION RESULTS  (LF = {LF}) ---")
        _print_objective(answer, -result.fun)
        print(f"Optimal T_cw            : {result.x[0]:.2f} °C")
        print(f"Optimal Q_a (per pack)  : {result.x[1]:.3f} m³/s")
        u_disp = result.x[2:]
        if len(u_disp) == p.N_r:           # per-rack mode
            for i in range(p.N_r):
                print(f"  Rack {i+1:2d}: u = {u_disp[i]:.3f}")
        else:                              # per-zone mode
            u_mat = u_disp.reshape(p.N_r, p.N_z)
            for i in range(p.N_r):
                print(f"  Rack {i+1:2d}: mean={np.mean(u_mat[i]):.3f}  "
                      + "  ".join(f"z{j}={u_mat[i,j]:.2f}" for j in range(p.N_z)))
        run_simulation(result.x, p, answer, plot=True,
                       save_dir=setup.fig_dir, run_tag=run_tag, show=False)
        record_performance(setup, LF, result.x, result.success, result.message)
        summary.append((LF, -result.fun, result.x[0], result.x[1], result.success))

    print("\n--- SUMMARY ---")
    print(f"{'LF':>5} {answer:>8} {'T_cw [°C]':>10} {'Q_a [m³/s/pack]':>16}  converged")
    for LF_k, obj, T_k, Q_k, ok in summary:
        print(f"{LF_k:5.2f} {obj:8.4f} {T_k:10.2f} {Q_k:16.3f}  {ok}")
    plt.show()


def run_option_C(setup: RunSetup, LF_list: list, niter: int, n_workers: int,
                 u0: str | None, scaled: bool, seed: int | None) -> None:
    """
    Option C: basin-hopping optimisation of setup.answer, one job per LF in
    LF_list, run in up to n_workers parallel processes (see _bh_job for
    u0 / scaled / seed).  The plots and the CSV rows are made afterwards.
    """
    p, answer = setup.p, setup.answer
    jobs = [dict(p=p, LF=LF_k, answer=answer, u0=u0, scaled=scaled,
                 T_cw_min=setup.T_cw_min, T_cw_max=setup.T_cw_max,
                 Q_a_min=setup.Q_a_min, Q_a_max=setup.Q_a_max,
                 niter=niter, seed=None if seed is None else seed + k)
            for k, LF_k in enumerate(LF_list)]
    print(f"Starting basin-hopping ({niter} hops) for LF = {LF_list}, "
          f"{max(1, min(n_workers, len(jobs)))} parallel worker(s) …")
    results = run_basinhopping_lfs(jobs, n_workers)

    for r in results:
        LF = r["LF"]
        print(f"\n--- OPTIMISATION RESULTS  (LF = {LF}, {r['minutes']:.0f} min) ---")
        _print_objective(answer, r["obj"])
        print(f"Optimal T_cw            : {r['x'][0]:.2f} °C")
        print(f"Optimal Q_a (per pack)  : {r['x'][1]:.3f} m³/s")
        run_simulation(r["x"], p, answer, plot=True,
                       save_dir=setup.fig_dir, run_tag=_run_tag(p, LF), show=False)
        record_performance(setup, LF, r["x"], r["success"], r["message"])

    print("\n--- SUMMARY ---")
    print(f"{'LF':>5} {answer:>8} {'T_cw [°C]':>10} {'Q_a [m³/s/pack]':>16}  converged")
    for r in results:
        print(f"{r['LF']:5.2f} {r['obj']:8.4f} {r['x'][0]:10.2f} {r['x'][1]:16.3f}  {r['success']}")
    plt.show()


def run_option_D(setup: RunSetup) -> None:
    """Option D: COP comparison of the Gordon-Ng and EIR-fPLR chiller models."""
    print("Generating chiller model comparison plot …")
    fig_chill = plot_chiller_comparison(setup.p)
    d_dir  = _ensure_dir(os.path.join(_BASE_OUT, "optimisation_chiller"))
    fpath  = os.path.join(d_dir, f"chiller_comparison_{_run_tag(setup.p, setup.LF)}.png")
    fig_chill.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  → saved {fpath}")
    plt.show()


def run_option_E(setup: RunSetup, n_weights: int, maxiter_local: int, scales: tuple) -> None:
    """Option E: three-objective Pareto front at setup.LF, weighted-sum method."""
    print("Computing three-objective Pareto front (weighted sum) …")
    fig_pareto = plot_pareto_fronts(
        setup.p,
        LF           = setup.LF,
        T_cw_min     = setup.T_cw_min,
        T_cw_max     = setup.T_cw_max,
        n_weights    = n_weights,
        maxiter_local= maxiter_local,
        scales       = scales,
    )
    d_dir = _ensure_dir(os.path.join(_BASE_OUT, "optimisation_pareto"))
    fpath = os.path.join(d_dir, f"pareto_weighted_sum_{_run_tag(setup.p, setup.LF)}.png")
    fig_pareto.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  → saved {fpath}")
    plt.show()


def run_option_F(setup: RunSetup, LF_list: list, pop_size: int, n_gen: int, seed: int) -> None:
    """Option F: three-objective Pareto fronts with NSGA-II, one per LF in LF_list."""
    d_dir = _ensure_dir(os.path.join(_BASE_OUT, "optimisation_pareto"))
    print(f"Computing three-objective Pareto fronts via NSGA-II "
          f"for LF ∈ {LF_list} …")
    fig_pareto = plot_pareto_fronts_multi_lf(
        setup.p,
        LF_list  = LF_list,
        T_cw_min = setup.T_cw_min,
        T_cw_max = setup.T_cw_max,
        pop_size = pop_size,
        n_gen    = n_gen,
        seed     = seed,
        save_dir = d_dir,        # writes pareto_multi_lf.csv here
    )
    fpath = os.path.join(d_dir, f"pareto_nsga2_multiLF_{_run_tag(setup.p, setup.LF)}.png")
    fig_pareto.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  → figure saved : {fpath}")
    plt.show()


def run_option_I(setup: RunSetup, T_cw_plot: float) -> None:
    """Option I: chiller COP and cooling capacity vs T_amb at T_cw = T_cw_plot."""
    print(f"Plotting chiller COP and cooling capacity vs T_amb "
          f"at T_cw = {T_cw_plot:.2f} °C …")
    fig_tamb = plot_chiller_vs_Tamb(setup.p, T_cw_plot)
    d_dir = _ensure_dir(os.path.join(_BASE_OUT, "chiller_performances"))
    fpath = os.path.join(d_dir, f"chiller_vs_Tamb_Tcw{T_cw_plot:g}.png")
    fig_tamb.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  → saved {fpath}")
    plt.show()


def run_option_J(setup: RunSetup, cities: list, LF_list: list, months: list | None,
                 warm_start: bool, method: str, niter: int, n_workers: int,
                 u0: str, restart: bool) -> None:
    """
    Option J: climate sweep — monthly COP optimisation for every city and LF,
    exported to Excel in climate_sweep/<folder> (see run_climate_sweep).
    """
    p = setup.p
    # One folder per (start mode, method); warm-start SLSQP keeps the
    # original folder name of the first sweep.
    clim_name = f"Ncu{p.N_cu}_Nr{p.N_r}_Np{p.n_packs}"
    if not (warm_start and method == "SLSQP"):
        clim_name += f"_{'warm' if warm_start else 'cold'}_{method}"
    if u0 != "uniform":     # new folder: the old checkpoints would otherwise be resumed
        clim_name += f"_x0{u0}"
    clim_dir = os.path.join(_BASE_OUT, "climate_sweep", clim_name)
    print(f"Output folder : {clim_dir}")
    run_climate_sweep(
        p,
        cities     = cities,
        LF_list    = LF_list,
        T_cw_min   = setup.T_cw_min,
        T_cw_max   = setup.T_cw_max,
        Q_a_min    = setup.Q_a_min,
        Q_a_max    = setup.Q_a_max,
        out_dir    = clim_dir,
        n_workers  = n_workers,
        warm_start = warm_start,
        method     = method,
        months     = months,
        niter      = niter,
        u0         = u0,
        restart    = restart,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11.  ENTRY POINT  —  USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════════════════
    # USER SETTINGS — everything needed to run the code is set here.
    # CODE_GUIDE.md describes every option, setting and output file.
    # ══════════════════════════════════════════════════════════════════════

    # ── 1. What to run ────────────────────────────────────────────────────
    #   "A" = forward simulation of one operating point (no optimisation)
    #   "B" = SLSQP optimisation (multi-start), one per LF in LF_LIST
    #   "C" = basin-hopping global optimisation, one per LF in LF_LIST, in parallel
    #   "D" = chiller model comparison plot (Gordon-Ng vs EIR-fPLR)
    #   "E" = three-objective Pareto front at LF, weighted-sum method
    #   "F" = three-objective Pareto fronts, NSGA-II, one per LF in F_LF_LIST
    #   "I" = chiller COP & cooling capacity vs ambient temperature (at I_T_CW)
    #   "J" = climate sweep: monthly COP optimisation for every city in
    #         CLIMATE_CITIES and LF in CLIMATE_LF → Excel in climate_sweep/
    SIMU   = "C"
    ANSWER = "COP"     # objective of A, B, C: "COP", "eta_ex" or "CPR" (J always uses COP)

    # ── 2. Data centre ────────────────────────────────────────────────────
    # LAYOUT defines ONE pack, from left to right: 0 = an IRC cooling unit,
    # a positive integer = that many consecutive racks.
    #   [0, 5, 0]        →  IRC_L | 5 racks | IRC_R   ← paper [84] original
    #   [0, 5, 0, 5, 0]  →  IRC_L | 5 racks | IRC_C | 5 racks | IRC_R
    LAYOUT  = [0, 8, 0]
    # Identical packs in the data centre, sharing ONE chiller.  Only one pack is
    # simulated; the KPIs are computed for N_PACKS packs (the shared chiller sees
    # N_PACKS × the load of one pack, which changes its PLR, COP and η_ex).
    N_PACKS = 2
    N_Z     = 5        # vertical zones per rack
    # Any other DCParams constant, e.g. {"T_amb": 30 + 273.15, "T_server_red": 25.0}
    # (names and units: see class DCParams or CODE_GUIDE.md).  {} = defaults.
    DC_OVERRIDES = {}

    # ── 3. Bounds of the decision variables (for ONE pack) ────────────────
    T_CW_MIN = 5.56    # [°C]  chilled-water supply temperature
    T_CW_MAX = 10      # [°C]  10 = rated limit of the EIR-fPLR curves (CGAM20)
    Q_A_MIN_PER_IRC = 0.6    # [m³/s]  airflow per IRC unit (pack total = × number of IRCs)
    Q_A_MAX_PER_IRC = 1.33   # [m³/s]

    # ── 4. Load factor(s) = mean server utilisation of a pack ─────────────
    LF      = 0.2                      # options A, D, E (and the run tag of D, E, F)
    LF_LIST = [0.2, 0.4, 0.6, 0.8]     # options B and C: one optimisation per LF (None = [LF])

    # ── 5. Settings of each option ────────────────────────────────────────
    # A — operating point to simulate
    A_T_CW = None      # [°C]           None = T_CW_MAX
    A_Q_A  = None      # [m³/s, pack]   None = Q_a,max / 1.8
    A_U    = "sides"   # utilisation: "sides" (load next to the IRCs), "centre" or
                       # "uniform" with mean LF, or a list of N_r (one per rack) or
                       # N_r × N_z (one per zone, rack by rack) values

    # B — SLSQP
    B_STARTS = ["sides", "centre", "uniform"]   # one SLSQP run per u pattern, best kept
                       # (works for COP, eta_ex and CPR).  Each x0 = that u pattern with
                       # the feasible T_cw / Q_a of a scan that is best for ANSWER.
                       # None = old single x0 (mid T_cw, mid Q_a, u = LF, unscaled objective)

    # C — basin hopping
    C_NITER   = 50         # hops per LF
    C_WORKERS = 4          # parallel processes, one LF each; 1 = sequential
    C_U0      = "uniform"  # x0 = scanned feasible start of this u pattern ("sides",
                           # "centre", "uniform"); None = T_cw_max, mid Q_a, u = LF
    C_SCALED  = True       # objective scaled by OBJ_SCALE (as option B); False = unscaled
    C_SEED    = None       # int for reproducible hops (LF k uses C_SEED + k); None = random

    # E — weighted-sum Pareto front
    E_N_WEIGHTS     = 5                    # simplex resolution: (n+1)(n+2)/2 SLSQP runs
    E_MAXITER_LOCAL = 50                   # SLSQP iterations per weight combination
    E_SCALES        = (4.0, 0.65, 20.0)    # typical COP, CPR, η_ex [%] used to normalise

    # F — NSGA-II Pareto fronts
    F_LF_LIST  = [0.2, 0.4, 0.6, 0.8]
    F_POP_SIZE = 150       # population size
    F_N_GEN    = 750       # generations
    F_SEED     = 42        # random seed (LF k uses F_SEED + k)

    # I — chiller plot
    I_T_CW = 9.0           # [°C] chilled-water temperature of the plot

    # J — climate sweep
    CLIMATE_CITIES     = ["Milan", "Oslo", "Marrakech"]   # keys of CLIMATE_2025
    CLIMATE_LF         = [0.2, 0.5, 0.8]
    CLIMATE_MONTHS     = None      # None = all 12, or e.g. ["Jan", "Jul"]
    CLIMATE_WARM_START = True      # True = month m+1 starts from the optimum of month m;
                                   # False = every month is an independent job
    CLIMATE_METHOD     = "SLSQP"   # "SLSQP" (as option B) or "basinhopping" (as option C)
    CLIMATE_BH_NITER   = 50        # hops (only for "basinhopping")
    CLIMATE_WORKERS    = 14        # parallel processes (14 = physical cores); 1 = sequential
    CLIMATE_U0         = "sides"   # default x0: "sides" (load next to the IRCs) or "uniform"
    CLIMATE_RESTART    = True      # warm start: each month also starts from the default x0,
                                   # the better optimum is kept

    # ══════════════════════════════════════════════════════════════════════
    # END OF USER SETTINGS — nothing below needs to be changed
    # ══════════════════════════════════════════════════════════════════════

    isirc, Nr, Ncu = make_layout(LAYOUT)
    print(f"Layout: {LAYOUT}  →  {Nr} racks, {Ncu} IRCs  (single pack)")
    if N_PACKS > 1:
        print(f"  Double-pack mode: {N_PACKS} × [{Nr} racks, {Ncu} IRCs] → "
              f"{N_PACKS*Nr} racks total, 1 shared chiller")
    print(f"  is_irc = {isirc.astype(int).tolist()}")

    p = DCParams(N_r=Nr, N_cu=Ncu, N_z=N_Z, is_irc=isirc, n_packs=N_PACKS, **DC_OVERRIDES)
    setup = RunSetup(
        p        = p,
        simu     = SIMU,
        answer   = ANSWER,
        LF       = LF,
        T_cw_min = T_CW_MIN,
        T_cw_max = T_CW_MAX,
        Q_a_min  = Q_A_MIN_PER_IRC * p.N_cu,
        Q_a_max  = Q_A_MAX_PER_IRC * p.N_cu,
        fig_dir  = _fig_dir(ANSWER),             # creates the folder if absent
    )
    print(f"Output folder : {setup.fig_dir}")
    print(f"Run tag       : {_run_tag(p, LF)}\n")

    if SIMU == "A":
        u_A = list(start_u(p, LF, A_U)) if isinstance(A_U, str) else list(A_U)
        x0_A = ([T_CW_MAX if A_T_CW is None else A_T_CW,
                 setup.Q_a_max / 1.8 if A_Q_A is None else A_Q_A] + u_A)
        run_option_A(setup, x0_A)
    elif SIMU == "B":
        run_option_B(setup, LF_LIST or [LF], B_STARTS)
    elif SIMU == "C":
        run_option_C(setup, LF_LIST or [LF], C_NITER, C_WORKERS, C_U0, C_SCALED, C_SEED)
    elif SIMU == "D":
        run_option_D(setup)
    elif SIMU == "E":
        run_option_E(setup, E_N_WEIGHTS, E_MAXITER_LOCAL, E_SCALES)
    elif SIMU == "F":
        run_option_F(setup, F_LF_LIST, F_POP_SIZE, F_N_GEN, F_SEED)
    elif SIMU == "I":
        run_option_I(setup, I_T_CW)
    elif SIMU == "J":
        run_option_J(setup, CLIMATE_CITIES, CLIMATE_LF, CLIMATE_MONTHS,
                     CLIMATE_WARM_START, CLIMATE_METHOD, CLIMATE_BH_NITER,
                     CLIMATE_WORKERS, CLIMATE_U0, CLIMATE_RESTART)
    else:
        raise ValueError(
            f"SIMU='{SIMU}' is not recognised.  Valid options:\n"
            "  'A' = forward simulation\n"
            "  'B' = SLSQP optimisation\n"
            "  'C' = basin-hopping global optimisation\n"
            "  'D' = chiller model comparison plot\n"
            "  'E' = three-objective Pareto front (weighted sum)\n"
            "  'F' = three-objective Pareto fronts (NSGA-II)\n"
            "  'I' = chiller COP & cooling capacity vs ambient temperature\n"
            "  'J' = climate sweep (monthly T_amb × LF × city) → Excel\n"
        )
