"""
FILD Synthetic Smoothing & Signal Reconstruction
This script processes and reconstructs Fast-Ion Loss Detector (FILD) signals 
from ASCOT simulation outputs. It utilizes K-Nearest Neighbors (KNN) and 
Inverse Distance Weighting (IDW) to smooth Monte Carlo statistical noise.
"""

from a5py import Ascot
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.tri as tri
from scipy.stats import binned_statistic

# Geometry and data loading
print("Loading geometry and data")
h5 = Ascot('odd_965.h5')

wall = h5.data.wall.active.read()
x = np.array(wall['x1x2x3'], dtype=float)
y = np.array(wall['y1y2y3'], dtype=float)
z = np.array(wall['z1z2z3'], dtype=float)
flag_array = wall["flag"].flatten()

indices = np.where(flag_array == 3)[0] 
X = x[indices]; Y = y[indices]; Z = z[indices]

AB_x, AB_y, AB_z = X[:, 1] - X[:, 0], Y[:, 1] - Y[:, 0], Z[:, 1] - Z[:, 0]
AC_x, AC_y, AC_z = X[:, 2] - X[:, 0], Y[:, 2] - Y[:, 0], Z[:, 2] - Z[:, 0]
CP_x = (AB_y * AC_z) - (AB_z * AC_y)
CP_y = (AB_z * AC_x) - (AB_x * AC_z)
CP_z = (AB_x * AC_y) - (AB_y * AC_x)
areas = 0.5 * np.sqrt(CP_x**2 + CP_y**2 + CP_z**2) 

print(f"FILD mesh loaded: {np.size(areas)} triangles.")

Cx = np.mean(X, axis=1)
Cy = np.mean(Y, axis=1)
Cz = np.mean(Z, axis=1)
puntos = np.column_stack((Cx, Cy, Cz)) 

arbol_vec = cKDTree(puntos)
r = np.sqrt(Cx**2 + Cy**2)

# Raw power extraction
ids, area_wet_unyt, power_unyt, _, _ = h5.data.active.getwall_loads(weights=True, flags=3)
itds = ids - 1 

area_wet = np.array(area_wet_unyt, dtype=float)
power = np.array(power_unyt, dtype=float)

potencia_global = np.zeros(len(flag_array))
potencia_global[itds] = power
potencia_fild_vec = potencia_global[indices] 

total_power_raw_val = np.sum(power)
power_surface = power / area_wet 
area_media = np.mean(area_wet)

print(f"Total RAW Power: {total_power_raw_val:.4e} W")

# Physics constraint: Larmor radius calculation
print("\nCalculating physical constraints: Larmor radius")

try:
    m, q, v, b = h5.data.active.getstate('mass','charge','vperp','bnorm', state='end', endcond='wall')
    rl = (m * 1.66053907e-27 * v) / (q * 1.60217663e-19 * b)
    rl_clean = rl[rl > 0.001] 
    
    if len(rl_clean) == 0:
        print("Warning: All particles are cold. Falling back to full dataset.")
        rl_clean = rl

    rlmax = float(np.max(rl_clean))
    rlmin = float(np.min(rl_clean))

    print(f'Minimum RL (Filtered >1mm): {rlmin*1000:.2f} mm')
    print(f'Maximum RL: {rlmax*1000:.2f} mm')

except Exception as e:
    print(f"Error computing RL: {e}. Falling back to default values.")
    rlmin, rlmax = 0.005, 0.040

def get_equivalent_k(radius, tree, sample_points):
    """Calculates the average number of mesh triangles within a given physical radius."""
    idx_sample = np.random.choice(len(sample_points), size=min(500, len(sample_points)), replace=False)
    indices_radio = tree.query_ball_point(sample_points[idx_sample], r=radius)
    k_med = np.mean([len(l) for l in indices_radio])
    return int(max(k_med, 1))

k_min_sweep = get_equivalent_k(rlmin, arbol_vec, puntos)
k_max_sweep = get_equivalent_k(rlmax, arbol_vec, puntos)

print(f"Neighbor search range: K=[{k_min_sweep}, {k_max_sweep}]")

# Synthetic optimization
print("\nInitializing optimization sweep")

Cx_w = np.mean(x[itds], axis=1); Cy_w = np.mean(y[itds], axis=1)
center_x = np.mean(Cx_w); center_y = np.mean(Cy_w)
sigma_sim = (rlmin + rlmax) / 2
amp_sim = 1.0e7

d2 = (Cx - center_x)**2 + (Cy - center_y)**2
truth_global = amp_sim * np.exp(-d2/(2*sigma_sim**2)) 

np.random.seed(42)
ruido_global = np.random.normal(0, amp_sim*0.15, size=len(Cx))
raw_sint_power = np.maximum((truth_global + ruido_global) * areas, 0)

temp_truth = np.zeros(len(x)); temp_truth[indices] = truth_global; truth_ref = temp_truth[itds]
temp_raw = np.zeros(len(x)); temp_raw[indices] = raw_sint_power/areas; raw_subset = temp_raw[itds]

f_raw = np.sum(truth_ref * area_wet) / (np.sum(raw_subset * area_wet) + 1e-12)
raw_subset_norm = raw_subset * f_raw
mae_raw = np.mean(np.abs(raw_subset_norm - truth_ref)) 

lista_k = range(k_min_sweep, k_max_sweep + 1)
mejoras = []
radios_eq = [] 

_, indices_all = arbol_vec.query(puntos, k=k_max_sweep+1)

for k_test in lista_k:
    inds_k = indices_all[:, :k_test]
    dists_k_exact, _ = arbol_vec.query(puntos, k=k_test)
    
    if k_test == 1: 
        r_eq_actual = np.mean(dists_k_exact)
    else: 
        r_eq_actual = np.mean(dists_k_exact[:, -1])
    radios_eq.append(r_eq_actual)

    epsilon_val = np.sqrt(area_media)
    pesos_val = 1.0 / (dists_k_exact + epsilon_val)**2
    
    if k_test == 1:
        pot_s = raw_sint_power[inds_k]; area_s = areas[inds_k]
        rec = pot_s / area_s 
    else:
        pot_s = raw_sint_power[inds_k]; area_s = areas[inds_k]
        num = np.sum(pot_s * pesos_val, axis=1)
        den = np.sum(area_s * pesos_val, axis=1)
        rec = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
    
    temp = np.zeros(len(x))
    temp[indices] = rec.flatten()  
    rec_sub = temp[itds]
    
    f = np.sum(truth_ref * area_wet) / (np.sum(rec_sub * area_wet) + 1e-12)
    rec_sub *= f
    
    mae = np.mean(np.abs(rec_sub - truth_ref))
    mejoras.append((1 - mae/mae_raw)*100)

mejoras = np.array(mejoras)
radios_eq = np.array(radios_eq)
lista_k_arr = np.array(lista_k)

idx_best = np.argmax(mejoras)
best_k = lista_k_arr[idx_best]
best_mejora = mejoras[idx_best]
best_rl_val = radios_eq[idx_best]

print(f"\nBest result: K={best_k}")
print(f"Max improvement: {best_mejora:.2f}%")
print(f"Associated Larmor radius: {best_rl_val*1000:.2f} mm")

if best_rl_val > 0.050:
    TOLERANCIA_FLAT = 0.2
    print(f"\nInfo: Optimal radius ({best_rl_val*1000:.1f}mm) > 50mm. Adjusting tolerance to {TOLERANCIA_FLAT}%")
else:
    TOLERANCIA_FLAT = 1.0
    print(f"\nInfo: Optimal radius ({best_rl_val*1000:.1f}mm) <= 50mm. Using standard tolerance {TOLERANCIA_FLAT}%")

umbral_flat = best_mejora - TOLERANCIA_FLAT
indices_flat = np.where(mejoras >= umbral_flat)[0]

if len(indices_flat) == 0: indices_flat = [idx_best]

idx_start, idx_end = indices_flat[0], indices_flat[-1]
rl_flat_min, rl_flat_max = radios_eq[idx_start], radios_eq[idx_end]
k_flat_min, k_flat_max = lista_k_arr[idx_start], lista_k_arr[idx_end]

print(f"\nStability analysis (Flat Zone, tol={TOLERANCIA_FLAT}%)")
print(f"Stable K Range: [{k_flat_min}, {k_flat_max}]")
print(f"Larmor Radius Range: [{rl_flat_min*1000:.2f} mm, {rl_flat_max*1000:.2f} mm]")

K_FINAL = best_k

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(lista_k, mejoras, 'b-o', label='Improvement (%)')
ax1.axvline(best_k, c='g', ls='--', label=f'Optimal K={best_k}')
ax1.axvspan(k_flat_min, k_flat_max, color='green', alpha=0.15, label=f'Flat Zone ({TOLERANCIA_FLAT}%)')
ax1.axhline(umbral_flat, color='orange', ls=':', label='Tolerance Threshold')
ax1.set_xlabel('K Neighbors')
ax1.set_ylabel('Improvement (%)', color='b')
ax1.grid(True, alpha=0.3); ax1.legend(loc='lower right')

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
tick_pos = np.linspace(k_min_sweep, k_max_sweep, 5)
tick_lab = np.interp(tick_pos, lista_k_arr, radios_eq)
ax2.set_xticks(tick_pos)
ax2.set_xticklabels([f"{t*1000:.1f}mm" for t in tick_lab])
ax2.set_xlabel("Equivalent Physical Radius")

plt.title("Physical Optimization of the Smoothing Radius")
plt.savefig('Optimization_K_FlatZone.png', dpi=300)
plt.show()

# Final reconstruction
print(f"\nFinal reconstruction (K={K_FINAL})")

dists_fin, inds_fin = arbol_vec.query(puntos, k=K_FINAL)
epsilon = np.sqrt(area_media)

pot_vec = potencia_fild_vec[inds_fin]
area_vec = areas[inds_fin]
dens_knn_glob = np.sum(pot_vec, axis=1) / np.sum(area_vec, axis=1)

temp = np.zeros(len(x)); temp[indices] = dens_knn_glob.flatten(); dens_knn_sub = temp[itds]
E_knn_pre = np.sum(dens_knn_sub * area_wet) 
fact_knn = total_power_raw_val / (E_knn_pre + 1e-12)
dens_knn_final = dens_knn_sub * fact_knn    

pesos_fin = 1.0 / (dists_fin + epsilon)**2 
num = np.sum(pot_vec * pesos_fin, axis=1)
den = np.sum(area_vec * pesos_fin, axis=1)
dens_idw_glob = np.divide(num, den, out=np.zeros_like(num), where=den!=0)

temp = np.zeros(len(x)); temp[indices] = dens_idw_glob.flatten(); dens_idw_sub = temp[itds]
E_idw_pre = np.sum(dens_idw_sub * area_wet) 
fact_idw = total_power_raw_val / (E_idw_pre + 1e-12)
dens_idw_final = dens_idw_sub * fact_idw    

def threshold_filter(d, a, p=0.95):
    """Retains the top p% of the signal energy, filtering out long-tail noise."""
    w = d*a; idx = np.argsort(w)[::-1]
    cut = np.searchsorted(np.cumsum(w[idx]), np.sum(w)*p)
    out = np.zeros_like(d); out[idx[:cut+1]] = d[idx[:cut+1]]
    return out

raw_alex = threshold_filter(power_surface, area_wet)
knn_alex = threshold_filter(dens_knn_final, area_wet)
idw_alex = threshold_filter(dens_idw_final, area_wet)


# Benchmark metrics
print("\nTable 1: Synthetic tournament (MAE)")
rec_knn_s = np.sum(raw_sint_power[inds_fin], axis=1) / np.sum(areas[inds_fin], axis=1)
rec_idw_s = np.sum(raw_sint_power[inds_fin]*pesos_fin, axis=1) / np.sum(areas[inds_fin]*pesos_fin, axis=1)
rec_mean = raw_sint_power / np.mean(areas)

comps = [("Raw", raw_sint_power/areas), ("Mean Area", rec_mean), ("KNN", rec_knn_s), ("IDW", rec_idw_s)]
res_list = []

for n, d in comps:
    t = np.zeros(len(x)); t[indices] = d.flatten()
    sub = t[itds] 
    sub *= np.sum(truth_ref*area_wet)/(np.sum(sub*area_wet)+1e-12)
    err = np.mean(np.abs(sub-truth_ref))
    res_list.append((n, err))

ref_err_raw = [e for n, e in res_list if n == "Raw"][0]

res_list.sort(key=lambda x:x[1])
for n, err in res_list:
    mejora = (1 - err/ref_err_raw)*100
    print(f"{n:<10} | Err: {err:.2e} | Improvement: {mejora:+.2f}%")


print("\nTable 2: Real energy balance")
E_knn_post = np.sum(dens_knn_final * area_wet)
E_idw_post = np.sum(dens_idw_final * area_wet)

lista_E = [
    ("Raw (Ref)", total_power_raw_val),
    ("KNN Pre", E_knn_pre),
    ("KNN Post", E_knn_post),
    ("IDW Pre", E_idw_pre),
    ("IDW Post", E_idw_post)
]
for n, e in lista_E:
    err = (e - total_power_raw_val) / total_power_raw_val * 100
    print(f"{n:<10} | {e:.4e} W | Err: {err:+.4f}%")


print("\nHotspot smoothing analysis")
p_raw = np.max(raw_alex)
p_idw = np.max(idw_alex)
print(f"Peak RAW: {p_raw:.2e} W/m2")
print(f"Peak IDW: {p_idw:.2e} W/m2")
print(f"Reduction: {(1-p_idw/p_raw)*100:.2f}%")


# Final visualizations
print("\nGenerating final plots...")
st = {'Raw':{'c':'r','ls':':','lw':1.5}, 'KNN':{'c':'b','ls':'--','lw':2}, 'IDW':{'c':'k','ls':'-','lw':2.5}}

theta = np.degrees(np.arctan2(Cy_w, Cx_w))
plt.figure(figsize=(10, 6))
for n, d in zip(['Raw','KNN','IDW'], [raw_alex, knn_alex, idw_alex]):
    s, e, _ = binned_statistic(theta, d, statistic='mean', bins=60)
    plt.plot((e[:-1]+e[1:])/2, s, label=n, **st[n])
plt.title("Angular Profile (95% Energy Filter)"); plt.xlabel("Degrees"); plt.ylabel("W/m2")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

rad_wetted = np.sqrt(Cx_w**2 + Cy_w**2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for n, d in zip(['Raw','KNN','IDW'], [power_surface, dens_knn_final, dens_idw_final]):
    s, e, _ = binned_statistic(rad_wetted, d, statistic='mean', bins=60)
    ax1.plot((e[:-1]+e[1:])/2, s, label=n, **st[n])
ax1.set_title("Radial UNFILTERED"); ax1.legend(); ax1.grid(True, alpha=0.3)

for n, d in zip(['Raw','KNN','IDW'], [raw_alex, knn_alex, idw_alex]):
    s, e, _ = binned_statistic(rad_wetted, d, statistic='mean', bins=60)
    ax2.plot((e[:-1]+e[1:])/2, s, label=n, **st[n])
ax2.set_title("Radial FILTERED"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.savefig('Radial_Final.png', dpi=300); plt.show()

t = np.zeros(len(x)); t[indices] = rec_idw_s.flatten(); plot_s = t[itds]
plot_s *= np.sum(truth_ref*area_wet)/(np.sum(plot_s*area_wet)+1e-12)
t = np.zeros(len(x)); t[indices] = raw_sint_power/areas; plot_r = t[itds]

triang = tri.Triangulation(Cx_w, Cy_w)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
tits = ["A) Ground Truth", "B) Raw + Noise", "C) IDW Reconstructed"]
dats = [truth_ref, plot_r, plot_s]
for i in range(3):
    ax[i].tricontourf(triang, dats[i], levels=50, cmap='jet')
    ax[i].triplot(triang, 'k-', lw=0.2, alpha=0.3); ax[i].set_title(tits[i]); ax[i].set_aspect('equal')
plt.savefig('Synthetic_2D_Final.png', dpi=300); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
im = ax[0].tripcolor(triang, raw_alex, cmap='jet', shading='flat')
ax[0].triplot(triang, 'k-', lw=0.2, alpha=0.3); ax[0].set_title("Real RAW Data"); plt.colorbar(im, ax=ax[0])
im = ax[1].tricontourf(triang, idw_alex, levels=100, cmap='jet')
ax[1].triplot(triang, 'k-', lw=0.2, alpha=0.3); ax[1].set_title(f"Real IDW Reconstruction (K={K_FINAL})"); plt.colorbar(im, ax=ax[1])
for a in ax: a.set_aspect('equal')
plt.savefig('Mapas2D_Final.png', dpi=300); plt.show()

print("Analysis complete.")
