import numpy as np
from a5py import Ascot

def calculate_larmor_bounds(filepath):
    """
    Calculates the minimum and maximum Larmor radius of the simulated 
    particles hitting the wall, serving as physical constraints for smoothing.
    """
    h5 = Ascot(filepath)
    
    try:
        # Retrieve particle state at the wall hit
        m, q, v, b = h5.data.active.getstate(
            'mass', 'charge', 'vperp', 'bnorm', 
            state='end', endcond='wall'
        )

        # Calculate Larmor radius (converting to SI units)
        rl = (m * 1.66053907e-27 * v) / (q * 1.60217663e-19 * b)

        # Filter out numerical noise/cold ions (RL < 1 mm)
        rl_clean = rl[rl > 0.001] 
        
        if len(rl_clean) == 0:
            print("Warning: All particles are cold. Falling back to full dataset.")
            rl_clean = rl

        rlmax = float(np.max(rl_clean))
        rlmin = float(np.min(rl_clean))

        print(f"Minimum RL (Filtered >1mm): {rlmin*1000:.2f} mm")
        print(f"Maximum RL: {rlmax*1000:.2f} mm")

    except Exception as e:
        print(f"Error computing Larmor radius: {e}. Falling back to default bounds.")
        rlmin, rlmax = 0.005, 0.040

    return {'rlmin': rlmin, 'rlmax': rlmax}

def get_equivalent_k_range(rl_bounds, arbol_vec, mesh_data):
    """
    Translates the physical Larmor radius bounds into a topological 
    number of neighbors (K) based on the local mesh density.
    """
    puntos = mesh_data['puntos']
    rlmin = rl_bounds['rlmin']
    rlmax = rl_bounds['rlmax']

    def get_equivalent_k(radius, tree, sample_points):
        """Calculates the average number of mesh triangles within a given physical radius."""
        idx_sample = np.random.choice(
            len(sample_points), 
            size=min(500, len(sample_points)), 
            replace=False
        )
        indices_radio = tree.query_ball_point(sample_points[idx_sample], r=radius)
        k_med = np.mean([len(l) for l in indices_radio])
        return int(max(k_med, 1))

    k_min_sweep = get_equivalent_k(rlmin, arbol_vec, puntos)
    k_max_sweep = get_equivalent_k(rlmax, arbol_vec, puntos)

    print(f"Neighbor search range evaluated: K=[{k_min_sweep}, {k_max_sweep}]")

    return {'k_min': k_min_sweep, 'k_max': k_max_sweep}
