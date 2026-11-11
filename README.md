# Data_centre_model
This code simulates the physics of an air-cooled data centre, calculates its performances, and can optimise these

# Code guide: `DataCentre.py`

This file explains how the data-centre simulation and optimisation code works:
how to run it, what every setting does, and what each parameter and function is for.

**Contents**
1. What the code does
2. Quick start
3. User settings
4. Run options A–J
5. Output files
6. How the model works
7. Physical and model parameters (`DCParams`)
8. Simulation state (`DCState`)
9. Function reference
10. Companion scripts
11. Notes and known limitations

---

## 1. What the code does

The code is a steady-state model of an air-cooled data centre, based on
Gupta et al., *Applied Energy* 299 (2021) 117050.

**The physical system:**
- **One room ("pack"):** a row of rack columns and in-row cooler (IRC) columns, for example `IRC | 8 racks | IRC`.
- **Zones:** each column is split into `N_z` vertical zones, numbered from floor to ceiling.
- **Front chamber (cold aisle):** this is where the servers draw in air and where the IRCs supply cold air.
- **Back chamber (hot aisle):** this is where the servers exhaust hot air and where the IRCs extract it.
- **Chiller:** one chiller (Trane CGAM20, EnergyPlus EIR-fPLR model) cools the IRCs' water. Several identical rooms/packs can share it (`N_PACKS`).

**What one simulation computes:** given an operating point, it solves for the pressures, airflows and temperatures in every zone. It then computes:
- the chiller, fan and IT powers;
- three performance indicators (KPIs):

| KPI | Meaning | Better when |
|---|---|---|
| **COP** | P_IT / (P_chiller + P_fans), the cooling efficiency | higher |
| **CPR** | Computing Performance Ratio, which falls as the CPUs get hotter | higher |
| **η_ex** | Exergy efficiency of the cooling cycle [%] | higher |

**What the operating point is:** the decision vector of one pack,

```
x = [ T_cw,  Q_a,  u_1, u_2, …, u_n ]
```

| Entry | Meaning | Unit |
|---|---|---|
| `T_cw` | chilled-water supply temperature | °C |
| `Q_a` | total airflow supplied by the IRCs of **one pack** | m³/s |
| `u_k` | server utilisation, between 0 and 1. Either one value per rack (`n = N_r`) or one value per zone (`n = N_r × N_z`, ordered rack by rack: rack 1 zones 1…N_z, then rack 2, …) | – |

**What the optimisations do:** they maximise one KPI over `x`, with the mean
utilisation fixed to the load factor:

```
mean(u) = LF
```

The options B, C and J use one `u` per zone (40 variables for `[0, 8, 0]` with 5 zones).

---

## 2. Quick start

1. **Install the packages:** `numpy`, `pandas`, `scipy`, `matplotlib`, `pymoo` (options E and F) and `openpyxl` (the Excel output of option J).
2. **Edit the settings:** open `DataCentre.py`, scroll to the bottom (section **11. ENTRY POINT — USER SETTINGS**) and edit the block between `USER SETTINGS` and `END OF USER SETTINGS`. Nothing else in the file needs to be changed to use the code.
3. **Run it:** run the file, for example `python "DataCentre.py"`, or with Run in Spyder or VS Code.

All outputs are written next to the script (see section 5).

> **Note on parallel options (C and J):** they start worker processes. Run them
> as a script (as above), not by pasting code into an interactive console.
> If the workers cannot start, the code says so and continues sequentially.

---

## 3. User settings

All of these are in the `if __name__ == "__main__":` block at the end of the file.

### 3.1 What to run

| Setting | Values | Meaning |
|---|---|---|
| `SIMU` | `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`, `"I"`, `"J"` | Run option (section 4). |
| `ANSWER` | `"COP"`, `"eta_ex"`, `"CPR"` | Objective of options A, B and C. It also sets the output folder `optimisation_<ANSWER>`. Option J always optimises COP. |

### 3.2 Data centre

| Setting | Default | Meaning |
|---|---|---|
| `LAYOUT` | `[0, 8, 0]` | Layout of **one pack**, from left to right: `0` is an IRC column and a positive integer is that many rack columns. For example, `[0, 5, 0, 5, 0]` gives IRC \| 5 racks \| IRC \| 5 racks \| IRC. |
| `N_PACKS` | `2` | Number of identical packs that share one chiller. Only one pack is simulated (see section 6.5). |
| `N_Z` | `5` | Vertical zones per rack. |
| `DC_OVERRIDES` | `{}` | Changes any `DCParams` constant without editing the class, for example `{"T_amb": 30 + 273.15}` or `{"T_server_red": 25.0}`. Units are as in section 7, and temperatures in `DCParams` are in **K** where noted. |

### 3.3 Bounds of the decision variables (one pack)

| Setting | Default | Meaning |
|---|---|---|
| `T_CW_MIN` | `5.56` °C | Lowest chilled-water temperature. |
| `T_CW_MAX` | `10` °C | Highest chilled-water temperature. 10 °C is the rated limit of the CGAM20 curves; above it the chiller model is extrapolated. |
| `Q_A_MIN_PER_IRC` | `0.6` m³/s | Lowest airflow per IRC. The pack bound is this × the number of IRCs (1.2 m³/s for 2 IRCs). |
| `Q_A_MAX_PER_IRC` | `1.33` m³/s | Highest airflow per IRC (2.66 m³/s per pack for 2 IRCs). |

### 3.4 Load factor

| Setting | Default | Used by |
|---|---|---|
| `LF` | `0.2` | Options A and E. It also sets the run tag in the file names of D, E and F. |
| `LF_LIST` | `[0.2, 0.4, 0.6, 0.8]` | Options B and C, with one optimisation per LF. `None` = `[LF]`. |

### 3.5 Option-specific settings

| Setting | Default | Meaning |
|---|---|---|
| **Option A** | | |
| `A_T_CW` | `None` | T_cw of the simulated point [°C]. `None` = `T_CW_MAX`. |
| `A_Q_A` | `None` | Q_a of the point [m³/s per pack]. `None` = Q_a,max / 1.8. |
| `A_U` | `"sides"` | Utilisation pattern with mean `LF`: `"sides"` (load packed next to the IRCs), `"centre"` (load in the middle racks) or `"uniform"`. It can also be an explicit list of `N_r` values (one per rack) or `N_r × N_z` values (one per zone). |
| **Option B** | | |
| `B_STARTS` | `["sides", "centre", "uniform"]` | One SLSQP run per utilisation pattern, each from a feasible scanned start point; the best result is kept (see `optimise_multistart`). `None` = a single run from mid T_cw, mid Q_a and uniform u, on the unscaled objective (the old behaviour, which often stops at the start point). |
| **Option C** | | |
| `C_NITER` | `50` | Basin-hopping hops per LF. |
| `C_WORKERS` | `4` | Parallel processes, one per LF. `1` = sequential. Each worker holds its own copy of the model in memory. |
| `C_U0` | `"uniform"` | Start pattern (`"sides"`, `"centre"` or `"uniform"`) with a scanned feasible T_cw and Q_a. `None` = T_cw,max, mid Q_a, u = LF. |
| `C_SCALED` | `True` | Scale the objective with `OBJ_SCALE`, as in option B. This matters for CPR. |
| `C_SEED` | `None` | Integer for reproducible hops (LF number k uses `C_SEED + k`). `None` = random. |
| **Option E** | | |
| `E_N_WEIGHTS` | `5` | Resolution of the weight grid: (n+1)(n+2)/2 SLSQP runs (21 for n = 5). |
| `E_MAXITER_LOCAL` | `50` | SLSQP iterations per weight combination. |
| `E_SCALES` | `(4.0, 0.65, 20.0)` | Typical COP, CPR and η_ex [%] values, used to normalise the objectives before weighting. |
| **Option F** | | |
| `F_LF_LIST` | `[0.2, 0.4, 0.6, 0.8]` | One Pareto front per LF. |
| `F_POP_SIZE` | `150` | NSGA-II population. |
| `F_N_GEN` | `750` | NSGA-II generations. |
| `F_SEED` | `42` | Random seed (LF number k uses `F_SEED + k`). |
| **Option I** | | |
| `I_T_CW` | `9.0` °C | Chilled-water temperature of the chiller plot. |
| **Option J** | | |
| `CLIMATE_CITIES` | Milan, Oslo, Marrakech | Keys of `CLIMATE_2025` (monthly mean temperatures, 2025). |
| `CLIMATE_LF` | `[0.2, 0.5, 0.8]` | Load factors. |
| `CLIMATE_MONTHS` | `None` | `None` = all 12 months, or a subset such as `["Jan", "Jul"]`. |
| `CLIMATE_WARM_START` | `True` | `True` = month m+1 starts from the optimum of month m (one job per city and LF). `False` = every month is an independent job. |
| `CLIMATE_METHOD` | `"SLSQP"` | `"SLSQP"` (as in option B) or `"basinhopping"` (as in option C). |
| `CLIMATE_BH_NITER` | `50` | Hops, only used for basin hopping. |
| `CLIMATE_WORKERS` | `14` | Parallel processes. 14 = the physical cores of the machine. |
| `CLIMATE_U0` | `"sides"` | Default start: `"sides"` (scanned, feasible) or `"uniform"` (T_cw,max, mid Q_a, u = LF). |
| `CLIMATE_RESTART` | `True` | With warm start, each month also starts from the default start point and keeps the better optimum. |

---

## 4. Run options A–J

| Option | What it does | Main outputs | Typical run time |
|---|---|---|---|
| **A** | Simulates one operating point (no optimisation), prints the KPIs and plots the utilisation, pressure and temperature maps. | 3 PNGs + one row in `performances.csv` | seconds |
| **B** | **SLSQP optimisation** of `ANSWER` for every LF in `LF_LIST`, with multi-start (`B_STARTS`). Prints a table per start and a summary. | 3 PNGs per LF + one row per LF in `performances.csv` | about 20–30 min per LF (3 starts) |
| **C** | **Basin-hopping** global optimisation (`C_NITER` hops of SLSQP from random jumps), one process per LF. Each progress line starts with its LF. | Same as B | hours per LF; the LFs run in parallel |
| **D** | Plots the COP of the Gordon-Ng model (Trane CGA120) against the EIR-fPLR model (Trane CGAM20). | `optimisation_chiller/chiller_comparison_<tag>.png` | seconds |
| **E** | Three-objective Pareto front (COP, CPR, η_ex) at `LF`, by **weighted sum**: one SLSQP run per weight triplet, then keeps the non-dominated points. | `optimisation_pareto/pareto_weighted_sum_<tag>.png` | long |
| **F** | Three-objective Pareto fronts with **NSGA-II** (pymoo), one per LF in `F_LF_LIST`, overlaid in one figure. | `optimisation_pareto/pareto_nsga2_multiLF_<tag>.png` + `pareto_multi_lf.csv` | very long |
| **I** | Chiller full-load capacity and COP against ambient temperature at `I_T_CW`, compared with the datasheet. | `chiller_performances/chiller_vs_Tamb_Tcw<I_T_CW>.png` | seconds |
| **J** | **Climate sweep**: a COP optimisation for every month (T_amb = monthly mean), city and LF. It writes checkpoints as it goes, so an interrupted sweep resumes where it stopped. Exports Excel at the end. | `climate_sweep/<folder>/` (section 5.3) | many hours |

All the optimisations use the outdoor temperature `DCParams.T_amb` (25 °C by
default), except option J, which uses the monthly temperatures.

---

## 5. Output files

All paths are relative to the script's folder (`_BASE_OUT`).

### 5.1 Folders

| Folder | Written by | Content |
|---|---|---|
| `optimisation_COP/`, `optimisation_CPR/`, `optimisation_eta_ex/` | A, B, C (by `ANSWER`) | Figures + `performances.csv` |
| `optimisation_chiller/` | D | Chiller model comparison |
| `optimisation_pareto/` | E, F | Pareto figures, `pareto_multi_lf.csv` |
| `chiller_performances/` | I | Chiller vs T_amb |
| `climate_sweep/<folder>/` | J | Checkpoints, Excel workbooks |

### 5.2 Figures and `performances.csv` (options A, B, C)

**The run tag.** Each figure name contains a tag, for example
`LF2e-1_Ncu2_Nr8_Np2`. That is LF = 0.2, 2 IRCs, 8 racks and 2 packs (see `_run_tag`).

**Figures:**
- `utilisation_<tag>.png`: utilisation per rack and zone.
- `pressure_<tag>.png`: front and back chamber pressures [Pa].
- `temperature_<tag>.png`: front chamber, back chamber and server exhaust temperatures [°C].

**`performances.csv`:** each run **appends** one row per LF, so earlier runs are kept. If the columns change, the file is rewritten with the union of the columns.

| Column | Meaning |
|---|---|
| `timestamp`, `simu`, `answer`, `method` | When the row was written, run option, objective, `"standard"` |
| `LF`, `N_packs`, `N_cu`, `N_r`, `N_z` | Load factor and layout |
| `T_cw_opt_C`, `Q_a_opt_m3s` | Optimal T_cw [°C] and Q_a **per pack** [m³/s] |
| `u_mean`, `u_rack_1…` | Mean utilisation and the utilisation vector. Despite the name, with one `u` per zone these are the **zone** values (`u_rack_1` to `u_rack_5` = rack 1, zones 1–5). |
| `COP`, `CPR`, `eta_ex_pct` | KPIs for the whole data centre (all packs, shared chiller) |
| `P_ch_W`, `P_f_W`, `P_it_W` | Chiller, fan and IT power, all packs [W] |
| `T_hw_C`, `T_ca_C`, `T_ha_C` | Return water, supply air (cold air) and return air (hot air) temperatures [°C] |
| `T_f_max_C` | Hottest server inlet [°C] (limit `T_server_red` = 27 °C) |
| `PLR` | Chiller part-load ratio = load / available capacity (unclamped) |
| `Q_cool_load_W`, `Q_ch_max_W`, `Q_unmet_W` | Chiller load, available capacity, load above capacity [W] |
| `Ex_in_W`, `Ex_dest_*_W`, `Ex_dest_*_pct` | Exergy input and exergy destruction in the chiller, IRC heat exchangers and fans [W and % of the total] |
| `converged`, `optimizer_msg` | Optimiser status. A point that breaks a limit is still logged, with `[infeasible: …]` added to the message. |

### 5.3 Climate sweep (option J)

**Folder name:** `climate_sweep/Ncu<N_cu>_Nr<N_r>_Np<N_packs>[_<warm|cold>_<method>][_x0<CLIMATE_U0>]`.
The method part is left out for warm-start SLSQP.

| File | Content |
|---|---|
| `checkpoints/<city>_<tag>[_mXX].csv` | One row per finished month: the optimum, KPIs, powers, monthly energies (MWh), feasibility and utilisation per rack and zone. Delete a file to recompute it. |
| `<city>.xlsx` | Sheets `Monthly`, `Annual` (annual energy, COP, pPUE) and `u_distribution` |
| `climate_comparison_all.xlsx` | All cities: `Monthly`, `Annual`, plus month × (city, LF) tables of COP, P_cool, E_cool, T_cw and η_ex |

`climate_reader.py` turns these into figures (section 10).

---

## 6. How the model works

### 6.1 The grid

**Columns:** the pack has `N_r + N_cu` columns, from left to right, in the order
given by `LAYOUT`. `is_irc` marks the IRC columns and `is_rack` the rack columns.

**Arrays:** every quantity is an array with one row per column and one column per
zone, `[column, zone]`, with zone 0 at the floor:
- **Pressures and temperatures:** `p_f`, `T_f` (front chamber) and `p_b`, `T_b` (back chamber), for every column.
- **Server quantities:** `Q_z`, `P_z`, `T_e` and `u`, for the racks only.

### 6.2 One simulation (`run_simulation`)

1. **Start values:** the utilisation sets the server power of each zone,
   `P_z = N_s (A1 + A2·u)`.
2. **Iteration loop.** The following is repeated 5 times, because the server airflow depends on the inlet temperature and the supply air temperature depends on the return air:
   1. **Pressure network** (`pressure_network`):
      - The server airflow of each zone depends on its inlet temperature: 3 servers × 0.01415 m³/s below 25 °C, rising by 0.00142 m³/s per K up to 35 °C.
      - Mass balances at every node give the front and back pressures.
      - Air moves between neighbouring zones (resistances `R_fH`, `R_fV`, `R_bH`, `R_bV`), through the servers, and leaks from front to back through the brushes (`R_br`).
      - Each IRC supplies `Q_a / N_cu` to its front column and extracts the same from its back column.
   2. **Temperature network** (`temperature_network`):
      - Energy balances at every node, with each flow carrying the temperature of the zone it comes from.
      - Servers heat their air by `P_z / (ρ c_p Q_z)`.
      - The IRC front nodes mix the coil supply at `T_ca` with any air entering them, for example hot air leaking through the brushes.
   3. **Return air:** `T_ha` = flow-weighted temperature of the air the IRCs extract (`calc_T_ha`).
   4. **IRC heat exchanger** (`chiller_residuals`):
      - Solved for the supply air `T_ca` and the return water `T_hw`.
      - The heat exchanger's UA grows with airflow (Dittus-Boelter, exponent 0.8), and the heat flow uses the mean temperature difference.
3. **Powers and KPIs:**
   - **Chiller:** EIR-fPLR model (`calc_Pch_EIRfPLR`), section 6.3. The Gordon-Ng model (`calc_Pch`) can also be used (section 9.3).
   - **Fans:** a quadratic in the airflow per IRC (`calc_Pf`).
   - **COP:** P_IT / (P_ch + P_f).
   - **η_ex:** `calc_exergy`.
   - **CPR:** `calc_CPR`, which uses CPU temperature = f(u, T_f).
4. **Several packs:** the KPIs are recomputed for `n_packs` packs sharing one chiller (section 6.5).
5. **Penalties** are added to the objective (section 6.4).

**What `run_simulation` returns:**
- By default: **−KPI + penalty**, the value the optimisers minimise.
- If a solver fails: `1000.0`.
- With `return_all=True`: the tuple described in section 9.

### 6.3 Chiller model (EnergyPlus EIR-fPLR, Trane CGAM20)

**Capacity:** the available capacity at the current temperatures (T in °C) is

```
Q_ch,max = Q_ch_ref · CAPFT(T_cw, T_amb)
```

It's optionally capped at `Q_ch_cap`.

**Power:**

```
P_ch = (Q_ch,max · EIRFT(T_cw, T_amb) / COP_ref) · EIRFPLR(PLR)
PLR  = Q_load / Q_ch,max
```

- **The curves:** CAPFT and EIRFT are biquadratic in (T_cw, T_amb), with coefficients `cc1–cc6` and `dd1–dd6`. EIRFPLR is quadratic in PLR (`ee1–ee3`).
- **PLR range:** PLR is clipped to `[PLR_min, PLR_max] = [0.25, 1]` inside EIRFPLR. The unclamped value is kept for the penalty.
- **Accuracy:** against the CGAM20 datasheet, the error is about 2 % at full load. At part load it's accurate down to about 50 % load and over-predicts the power at 25 % load.

### 6.4 Limits (soft penalties)

The optimisers minimise `−KPI + penalty`, where

```
penalty = 1000·(T_excess)² + 1000·(PLR_excess)² + 1000·(T_deficit)²
```

| Term | Active when |
|---|---|
| `T_excess` | the hottest server inlet is above `T_server_red` (27 °C, ASHRAE A2) |
| `PLR_excess` | the PLR is outside [0.25, 1], meaning the chiller cannot deliver the load or runs too low |
| `T_deficit` | the coldest server inlet is below `T_server_min` (18 °C). It applies only to the objectives listed in `T_server_min_for` (CPR and η_ex by default). Without it, those objectives push the air as cold as possible. |

### 6.5 Several packs (`N_PACKS > 1`)

Only one pack is simulated. `_scale_kpis_for_n_packs` then rebuilds the KPIs for `n_packs` identical packs:

| Quantity | How it scales |
|---|---|
| IT power and fan power | × n |
| Chiller load | × n. The one shared chiller has a higher PLR, so its power is recomputed with EIRFPLR and is not n × the power of one pack. |
| CPR | unchanged |
| η_ex | recomputed with the total flows |

The bounds and `Q_a` always refer to **one pack**.

### 6.6 Optimisation methods

**Gradient-based local search (SLSQP, option B).** SLSQP uses finite-difference gradients. To make it work well, the code:
- **Starts from feasible points** (`scan_x0`):
  - T_cw is scanned in 1 K steps.
  - For each T_cw, Q_a values are tried from the smallest airflow that keeps the hottest inlet just below 27 °C up to Q_a,max (or the airflow where the coldest inlet reaches about 18 °C when `T_server_min` applies).
  - The feasible point with the best objective is kept.
- **Uses several utilisation patterns** (sides, centre, uniform) and keeps the best result (`optimise_multistart`). This matters because the load distribution has several local optima: concentrating the load pays, but the optimiser stays near the pattern it starts from.
- **Scales the objective** (`OBJ_SCALE`): CPR × 100 and η_ex × 0.1. Without this, SLSQP stops after one step for CPR.

**Global search (basin hopping, option C).**
- **How it searches:** random jumps (`PerVariableStep`), each followed by an SLSQP search.
- **Jump sizes:** 20 % of the T_cw and Q_a ranges, and 15 % for u. The u jumps have zero mean, so `mean(u)` stays close to LF.
- **Parallel runs:** one process per LF.

---

## 7. Physical and model parameters (`DCParams`)

`DCParams` is frozen: it cannot be changed during a run. To change a value, set it
through `DC_OVERRIDES` (section 3.2). The default values are given below.

### Geometry

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `N_s` | 3 | – | Servers per zone |
| `N_z` | 5 | – | Zones per rack (set from `N_Z`) |
| `N_r` | 10 | – | Racks per pack (set from `LAYOUT`) |
| `N_cu` | 3 | – | IRCs per pack (set from `LAYOUT`) |
| `is_irc` | – | bool array | True for IRC columns (set from `LAYOUT`). `is_rack = ~is_irc`. |
| `n_packs` | 1 | – | Packs sharing one chiller (set from `N_PACKS`) |

### Fluids

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `rho_a`, `Cp_a` | 1.18, 1006 | kg/m³, J/(kg K) | Air density, specific heat |
| `rho_w`, `Cp_w` | 997, 4180 | kg/m³, J/(kg K) | Water density, specific heat |

### Airflow resistances (pressure network)

| Name | Default | Meaning |
|---|---|---|
| `R_fH`, `R_fV` | 6.0 | Horizontal, vertical resistance in the front chamber |
| `R_bH`, `R_bV` | 31.0 | Horizontal, vertical resistance in the back chamber |
| `R_s` | 2415.3 | Inside a passive server (not used by the current network) |
| `R_br` | 3977.7 | Brushes between front and back (leakage) |

### IRC heat exchanger and water loop

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `UA_ref_per_unit` | 2070 | W/K | Nominal UA of one IRC (Rittal, 30 kW) |
| `Q_a_ref_per_unit` | 1.389 | m³/s | Nominal airflow of one IRC (5000 m³/h) |
| `Q_w` | 40 GPM = 2.52e-3 | m³/s | Chilled-water flow (fixed) |
| `Q_w_min`, `Q_w_max` | 30, 69 GPM | m³/s | Allowed evaporator flow of the CGAM20 (for information, not used) |

### Environment, fans, servers

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `T_amb` | 298.15 (25 °C) | **K** | Outdoor air temperature (chiller condenser) |
| `eta_fan` | 0.604 | – | Fan efficiency (exergy destruction in the fans) |
| `Q_cf` | 2 × 9413 CFM | m³/s | Condenser fan airflow (CGAM20) |
| `A1`, `A2` | 223.4, 154.5 | W | Server power = A1 + A2·u (HP ProLiant DL360 G5) |
| `B1`, `B2`, `B3` | 480, −3073, 6031 | – | Fan power per IRC = B1 + B2·q + B3·q² (q = airflow per IRC, m³/s) |

### Chiller

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `Q_ch_ref` | 68 930 | W | Reference capacity (CGAM20) |
| `Q_ch_cap` | None | W | Optional upper limit on the available capacity |
| `COP_ref` | 2.96 | – | Reference COP |
| `T_cw_min`, `T_cw_max` | 5.56, 10.0 | °C | Rated T_cw range. Only used in `validate_physics` and the chiller plots; the optimisation bounds come from `T_CW_MIN` and `T_CW_MAX`. |
| `T_amb_min`, `T_amb_max` | 4.44, 46.11 | °C | Rated ambient range (chiller plots, climate "in range" flag) |
| `PLR_min`, `PLR_max` | 0.25, 1.0 | – | PLR range (EIRFPLR clip + penalty) |
| `cc1`–`cc6` | – | – | CAPFT coefficients (1, T_cw, T_cw², T_amb, T_amb², T_cw·T_amb) |
| `dd1`–`dd6` | – | – | EIRFT coefficients (same terms) |
| `ee1`–`ee3` | – | – | EIRFPLR coefficients (1, PLR, PLR²) |
| `C1`–`C3` | 0, 38.14, 0.844 | – | Gordon-Ng model (only used by option D and the unused `calc_Pch`) |

### Thermal limits

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `T_server_red` | 27.0 | °C | Highest allowed server inlet temperature (penalty) |
| `T_server_min` | 18.0 | °C | Lowest allowed server inlet temperature (penalty). `None` = no limit. |
| `T_server_min_for` | ("CPR", "eta_ex") | – | Objectives the lower limit applies to. COP is left out because the limit would bind in cold months at high LF. |

### CPU temperature and CPR

| Name | Default | Meaning |
|---|---|---|
| `E1`–`E5` | 20.2, 59.1, 0.76, −8.1, −0.56 | T_CPU = E1 + E2·u + E3·T_f + E4·u² + E5·u·T_f [°C] |
| `F1`, `F2` | 0.75, −1.72e-3 | CPR = F1 + F2 · mean(T_CPU) |

---

## 8. Simulation state (`DCState`)

`DCState` holds everything computed during one simulation. `run_simulation` creates it with the right sizes. All temperatures are in **K**.

| Field | Shape | Meaning |
|---|---|---|
| `u` | racks × zones | Server utilisation |
| `p_f`, `p_b` | columns × zones | Front and back chamber pressure [Pa] |
| `T_f`, `T_b` | columns × zones | Front (server inlet) and back chamber temperature |
| `T_e` | racks × zones | Server exhaust temperature |
| `T_ha`, `T_ca` | scalar | Hot return air, cold supply air |
| `T_hw`, `T_cw` | scalar | Return water, supply water |
| `Q_z` | columns × zones | Server airflow per zone [m³/s] |
| `Q_a` | scalar | Pack airflow [m³/s] |
| `P_z` | racks × zones | IT power per zone [W] |
| `P_ch`, `P_f`, `P_cool` | scalar | Chiller, fan and total cooling power [W] |
| `COP`, `CPR`, `eta_ex`, `PLR` | scalar | KPIs (η_ex in %) |
| `Q_coil`, `Q_ch_max` | scalar | Chiller load and available capacity [W] |
| `psi_ch`, `psi_hx`, `psi_f`, `psi_in` | scalar | Exergy destruction (chiller, IRC heat exchangers, fans) and exergy input [W] |

---

## 9. Function reference

The functions are listed in the order of the file's sections. Functions whose
names start with `_` are internal helpers.

### 3. Physics

| Function | Purpose |
|---|---|
| `pressure_residuals(P_1D, p, s)` | Mass-balance error at every front and back node, for the root solver. |
| `pressure_network(p, s)` | Sets the server airflows from `T_f`, then solves the pressures. Writes `s.Q_z`, `s.p_f`, `s.p_b`. |
| `temperature_residuals(T_1D, s, p)` | Energy-balance error at every node (front, back, server exhaust), with temperatures taken from the zone each flow comes from. |
| `temperature_network(p, s)` | Solves the temperatures, starting from the previous solution when there is one. Writes `s.T_f`, `s.T_b`, `s.T_e`. |
| `calc_T_ha(s, p)` | Flow-weighted temperature of the air extracted by the IRCs. |
| `chiller_residuals(vars, s, p)` | Air-side and water-side balances of the IRC heat exchanger, solved for [T_ca, T_hw]. |
| `chiller_full_load(p, T_cw_C, T_amb_C)` | Returns (available capacity, full-load power) of the EIR-fPLR chiller [W]. |
| `calc_Pch_EIRfPLR(s, p)` | Chiller power at the current load. Sets `s.P_ch`, `s.PLR`, `s.Q_coil`, `s.Q_ch_max`. |
| `calc_Pch(s, p)` | Gordon-Ng chiller power. **Not used** by the run options (kept for reference). |
| `calc_Pf(s, p)` | Fan power of the IRCs. |
| `calc_exergy(s, p)` | Exergy destruction and η_ex [%] of one pack. |
| `calc_CPR(s, p)` | CPU temperatures and CPR. |

### 4. Validation

| Function | Purpose |
|---|---|
| `validate_physics(s, p, tol)` | Raises an error if the state is unphysical. **Not called** by the run options. |

### 5. Simulation

| Function | Purpose |
|---|---|
| `_scale_kpis_for_n_packs(s, p)` | KPIs for `n_packs` packs sharing one chiller (section 6.5). |
| `run_simulation(x, p, answer, plot, return_all=False, save_dir=None, run_tag=None, allow_infeasible=False, show=True)` | **The core function**: one full simulation of the operating point `x` (sections 1 and 6.2). Its arguments and return value are described below this table. |
| `_pareto_mask(values)` | True for the non-dominated rows of an (N × M) array, where every objective is maximised. |

**The arguments of `run_simulation`:**
- `answer`: which KPI is returned.
- `plot=True`: prints the results and draws the three maps. They are saved if `save_dir` and `run_tag` are given, and `show=False` keeps them open without blocking.
- `return_all=True`: returns a tuple instead of the objective.
- `allow_infeasible=True`: also returns the tuple for points that break a limit. Otherwise those points return `None`.

**The return value:**
- **Default:** `−KPI + penalty`, or `1000.0` if a solver fails.
- **With `return_all=True`**, a tuple with these entries (indices in brackets):

| Index | Entry | Unit |
|---|---|---|
| [0]–[2] | COP, CPR, η_ex | – , – , % |
| [3]–[5] | P_ch, P_f, P_IT, all packs | W |
| [6]–[8] | T_hw, T_ca, T_ha | °C |
| [9] | hottest server inlet | °C |
| [10] | PLR | – |
| [11] | chiller load | W |
| [12] | chiller capacity | W |
| [13]–[16] | ψ_ch, ψ_hx, ψ_f, ψ_in | W |
| [17] | coldest server inlet | °C |

### 6. Output helpers

| Function | Purpose |
|---|---|
| `_BASE_OUT` | Root output folder = the folder of the script. |
| `_OBJ_DIR` | Objective → sub-folder name. |
| `_run_tag(p, LF)` | File-name tag, for example `LF2e-1_Ncu2_Nr8_Np2`. |
| `_ensure_dir(path)` | Creates a folder if needed. |
| `_fig_dir(answer)` | Output folder of an objective (created if needed). |
| `_perf_extra(triple)` | Chiller and exergy columns of `performances.csv`, from the `return_all` tuple. |
| `save_performances(answer, data)` | Appends one row to `optimisation_<answer>/performances.csv`. |

### 7. Plotting

| Function | Purpose |
|---|---|
| `_col_labels(p)` | Axis labels of the columns (IRC Left / Rack k / IRC Right). |
| `plot_utilisation_grids(s, p)` | Utilisation map. |
| `plot_pressure_grids(s, p)` | Front and back pressure maps. |
| `plot_temperature_grids(s, p)` | Front, back and exhaust temperature maps. |
| `plot_heatmaps(df, m, n)` | Contour maps of the KPIs over a (T_cw, Q_a) grid. **Not used** by the run options. |
| `plot_chiller_comparison(p)` | Option D: COP against T_cw and against PLR, Gordon-Ng vs EIR-fPLR. |
| `plot_chiller_vs_Tamb(p, T_cw_C, T_amb_lo, T_amb_hi)` | Option I: capacity and COP against T_amb, with the datasheet points. |
| `plot_pareto_fronts(p, LF, T_cw_min, T_cw_max, n_weights, maxiter_local, scales)` | Option E: weighted-sum Pareto front and its figure. |
| `plot_pareto_fronts_nsga2(p, LF, T_cw_min, T_cw_max, pop_size, n_gen, seed)` | NSGA-II Pareto front for one LF (option F uses the multi-LF version below). |
| `plot_pareto_fronts_multi_lf(p, LF_list, T_cw_min, T_cw_max, pop_size, n_gen, seed, save_dir)` | Option F: NSGA-II fronts for several LFs, the figure, and `pareto_multi_lf.csv`. |

### 7b. Climate sweep and start points

| Name | Purpose |
|---|---|
| `CLIMATE_2025`, `MONTHS`, `DAYS_2025` | Monthly mean temperatures [°C] per city, month names, days per month. |
| `_climate_ckpt_path(out_dir, city, p, LF, month_idx)` | Path of a checkpoint CSV. |
| `_climate_optimise_month(job, m, x0s)` | COP optimisation of one month from each start point in `x0s` (best kept). Writes one checkpoint row. |
| `side_loaded_u(p, LF, from_centre)` | Utilisation with the racks next to the IRCs filled first (or the middle racks, with `from_centre=True`), so that mean(u) = LF. |
| `_climate_x0(job, m)` | Default start point of a month (`"sides"`: scanned, or `"uniform"`). |
| `start_u(p, LF, u0)` | Utilisation of a named pattern: `"sides"`, `"centre"` or `"uniform"`. |
| `scan_x0(p, LF, answer, u0, T_cw_min, T_cw_max, Q_a_min, Q_a_max, n_Q)` | Feasible start point with the best objective (section 6.6). |
| `OBJ_SCALE` | Objective scaling for SLSQP: COP 1, η_ex 0.1, CPR 100. |
| `optimise_multistart(p, LF, answer, starts, …, maxiter, disp)` | SLSQP from one scanned start per pattern. Returns (best result, table of each start). |
| `_climate_chain(job)` | Warm-start worker: the months of one (city, LF) in order. |
| `_climate_month(job)` | Cold-start worker: one month. |
| `_climate_annual(df)` | Annual totals per (city, LF). |
| `export_climate_excel(out_dir)` | Gathers the checkpoints into the Excel workbooks. |
| `run_climate_sweep(p, cities, LF_list, …)` | Option J driver: builds the jobs, runs them in parallel, exports Excel. |

### 8–9. Layout and optimiser helpers

| Name | Purpose |
|---|---|
| `make_layout(spec)` | Turns `LAYOUT` into (is_irc, N_r, N_cu). |
| `PerVariableStep(stepsizes, bounds)` | Basin-hopping jump with one step size per variable. The u jumps have zero mean. |
| `_bh_job(job)` | Option C worker: basin hopping for one LF. |
| `run_basinhopping_lfs(jobs, n_workers)` | Runs `_bh_job` for every LF in parallel processes, falling back to sequential. |

### 10. Run options

| Name | Purpose |
|---|---|
| `RunSetup` | Settings shared by every option: `p`, `simu`, `answer`, `LF`, the bounds of one pack, and `fig_dir`. |
| `record_performance(setup, LF, x_opt, converged, opt_msg)` | Re-evaluates the optimum and appends its row to `performances.csv`, with a note if it breaks a limit. |
| `_print_objective(answer, value)` | Prints the optimum with its unit. |
| `run_option_A` … `run_option_J` | One function per run option (section 4). The main block calls the one selected by `SIMU`. |

---

## 10. Companion scripts

| Script | Purpose |
|---|---|
| `performance_reader.py` | Reads `optimisation_<KPI>/performances.csv` and plots the results against LF (KPI chosen by `DEFAULT_KPI`). |
| `climate_reader.py` | Reads a `climate_sweep/<folder>` and draws the monthly, against-T_amb and annual figures and the utilisation maps. |
| `chiller_calculation.py` | Validates the chiller models against the Trane datasheets (accuracy figures, COP against T_cw for several PLR). |

---

## 11. Notes and known limitations

- **Steady state only:** there is no thermal inertia.
- **Bounds are per pack:** `Q_a` and the airflow bounds refer to one pack. Powers and KPIs are for the whole data centre.
- **Penalties, not hard constraints:** the limits are soft penalties, so an optimum may sit very slightly past a limit (for example PLR = 1.002). `performances.csv` flags such points in `optimizer_msg`.
- **η_ex and airflow:** only (1 − η_fan)·P_f counts as fan exergy destruction, while all of P_f counts as input. η_ex therefore rewards high airflow and hardly depends on the load distribution.
- **Several local optima:** the load distribution has several local optima, so the start points matter. Option B tries three patterns, and option C searches globally.
- **Validity of the chiller curves:** they are rated for T_cw 5.56–10 °C and T_amb 4.4–46 °C. Outside that range they are extrapolated.
