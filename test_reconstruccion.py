import numpy as np
import pytest
from scipy.spatial import cKDTree
# Asegúrate de que reconstruction.py esté en la misma carpeta
from reconstruction import optimize_smoothing, final_reconstruction

@pytest.fixture
def dummy_data():
    """Generates mock data for testing."""
    np.random.seed(42)
    # Mocking essential attributes for reconstruction
    n_points = 10
    puntos = np.random.rand(n_points, 3)
    areas = np.random.rand(n_points) * 0.1
    
    mesh_data = {
        'x': np.random.rand(n_points, 3), 
        'y': np.random.rand(n_points, 3),
        'indices': np.arange(n_points),
        'areas': areas,
        'puntos': puntos,
        'Cx': puntos[:, 0], 'Cy': puntos[:, 1]
    }
    
    arbol_vec = cKDTree(puntos)
    
    potencia_fild_vec = np.random.rand(n_points) * 1e6
    raw_data = {
        'itds': np.arange(n_points),
        'area_wet': areas,
        'potencia_fild_vec': potencia_fild_vec,
        'total_power_raw_val': np.sum(potencia_fild_vec * areas),
        'area_media': np.mean(areas)
    }
    
    k_range = {'k_min': 2, 'k_max': 4}
    
    return mesh_data, raw_data, arbol_vec, k_range

def test_energy_conservation(dummy_data):
    """Checks if final_reconstruction conserves energy."""
    mesh_data, raw_data, arbol_vec, _ = dummy_data
    k_test = 3
    
    final_signal = final_reconstruction(mesh_data, raw_data, arbol_vec, k_test)
    
    # Energy comparison
    E_in = raw_data['total_power_raw_val']
    E_out = np.sum(final_signal * raw_data['area_wet'])
    
    assert np.isclose(E_in, E_out, rtol=1e-5), f"Energy not conserved: {E_in} vs {E_out}"

def test_optimization_runs(dummy_data):
    """Checks if optimization finishes."""
    mesh_data, raw_data, arbol_vec, k_range = dummy_data
    best_k, stats = optimize_smoothing(mesh_data, raw_data, k_range, arbol_vec)
    
    assert best_k >= k_range['k_min']
    assert best_k <= k_range['k_max']
