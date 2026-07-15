import numpy as np

def optimize_smoothing(mesh_data, raw_data, k_range, arbol_vec):
    """
    Performs a sweep over K hyperparameters to find the optimal smoothing 
    based on a synthetic target function reconstruction.
    """
    Cx, Cy = mesh_data['Cx'], mesh_data['Cy']
    areas = mesh_data['areas']
    itds = raw_data['itds']
    indices = mesh_data['indices']
    
    # Synthetic Target Generation
    sigma_sim = (0.005 + 0.040) / 2 
    amp_sim = 1.0e7
    
    # Anchor the synthetic Gaussian exactly at the real hotspot
    idx_peak = np.argmax(raw_data['potencia_fild_vec'])
    center_x = Cx[idx_peak]
    center_y = Cy[idx_peak]
    
    d2 = (Cx - center_x)**2 + (Cy - center_y)**2
    truth_global = amp_sim * np.exp(-d2 / (2 * sigma_sim**2)) 
    
    # Noise Injection
    np.random.seed(42)
    ruido_global = np.random.normal(0, amp_sim * 0.15, size=len(Cx))
    raw_sint_power = np.maximum((truth_global + ruido_global) * areas, 0)
    
    temp_truth = np.zeros(len(mesh_data['x'].flatten()))
    temp_truth[indices] = truth_global
    truth_ref = temp_truth[itds]
    
    temp_raw = np.zeros(len(mesh_data['x'].flatten()))
    temp_raw[indices] = raw_sint_power / areas
    raw_subset = temp_raw[itds]
    
    f_raw = np.sum(truth_ref * raw_data['area_wet']) / (np.sum(raw_subset * raw_data['area_wet']) + 1e-12)
    raw_subset_norm = raw_subset * f_raw
    mae_raw = np.mean(np.abs(raw_subset_norm - truth_ref))
    
    k_start = min(k_range['k_min'], k_range['k_max'])
    k_end = max(k_range['k_min'], k_range['k_max'])
    lista_k = range(k_start, k_end + 1)
    
    mejoras, radios_eq = [], []
    max_query = min(k_end + 1, len(mesh_data['puntos']))
    _, indices_all = arbol_vec.query(mesh_data['puntos'], k=max_query)
    
    best_pred_map = None
    max_mejora = -np.inf

    for k_test in lista_k:
        if k_test == 1:
            inds_k = indices_all[:, 0] if indices_all.ndim > 1 else indices_all
            dists_k_exact, _ = arbol_vec.query(mesh_data['puntos'], k=k_test)
            r_eq = np.mean(dists_k_exact)
            radios_eq.append(r_eq)
            
            pot_s = raw_sint_power[inds_k]
            area_s = areas[inds_k]
            rec = pot_s / area_s 
        else:
            inds_k = indices_all[:, :k_test]
            dists_k_exact, _ = arbol_vec.query(mesh_data['puntos'], k=k_test)
            r_eq = np.mean(dists_k_exact[:, -1])
            radios_eq.append(r_eq)
            
            epsilon = np.sqrt(raw_data['area_media'])
            pesos = 1.0 / (dists_k_exact + epsilon)**2
            
            pot_s = raw_sint_power[inds_k]
            area_s = areas[inds_k]
            
            num = np.sum(pot_s * pesos, axis=1)
            den = np.sum(area_s * pesos, axis=1)
            rec = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        
        temp = np.zeros(len(mesh_data['x'].flatten()))
        temp[indices] = rec.flatten()
        rec_sub = temp[itds]
        
        f = np.sum(truth_ref * raw_data['area_wet']) / (np.sum(rec_sub * raw_data['area_wet']) + 1e-12)
        rec_sub *= f
        
        mae = np.mean(np.abs(rec_sub - truth_ref))
        mejora_pct = (1 - mae / mae_raw) * 100
        mejoras.append(mejora_pct)
        
        if mejora_pct > max_mejora:
            max_mejora = mejora_pct
            best_pred_map = np.copy(rec_sub)

    mejoras = np.array(mejoras)
    idx_best = np.argmax(mejoras)
    best_k = list(lista_k)[idx_best]
    best_mejora = mejoras[idx_best]
    
    # Stability Region Analysis
    TOLERANCE = 0.2
    umbral_flat = best_mejora - TOLERANCE
    indices_flat = np.where(mejoras >= umbral_flat)[0]
    if len(indices_flat) == 0: indices_flat = [idx_best]
    
    stats = {
        'lista_k': list(lista_k),
        'mejoras': list(mejoras),
        'radios_eq': radios_eq,
        'best_k': best_k,
        'umbral_flat': umbral_flat,
        'k_flat_min': list(lista_k)[indices_flat[0]],
        'k_flat_max': list(lista_k)[indices_flat[-1]],
        'truth_map': truth_ref,
        'raw_map': raw_subset,
        'pred_map': best_pred_map
    }
    
    return best_k, stats

def final_reconstruction(mesh_data, raw_data, arbol_vec, k_final):
    """
    Applies the optimal hyperparameter to generate the smoothed prediction.
    """
    dists, inds = arbol_vec.query(mesh_data['puntos'], k=k_final)
    epsilon = np.sqrt(raw_data['area_media'])
    
    pot_vec = raw_data['potencia_fild_vec'][inds]
    area_vec = mesh_data['areas'][inds]
    
    if k_final == 1:
        dens_idw = pot_vec / area_vec
    else:
        pesos = 1.0 / (dists + epsilon)**2 
        num = np.sum(pot_vec * pesos, axis=1)
        den = np.sum(area_vec * pesos, axis=1)
        dens_idw = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    
    temp = np.zeros(len(mesh_data['x'].flatten()))
    temp[mesh_data['indices']] = dens_idw.flatten()
    final_signal = temp[raw_data['itds']]
    
    fact = raw_data['total_power_raw_val'] / (np.sum(final_signal * raw_data['area_wet']) + 1e-12)
    return final_signal * fact
