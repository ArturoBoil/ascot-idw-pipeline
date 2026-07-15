import numpy as np
from scipy.spatial import cKDTree
from a5py import Ascot

TARGET_FLAG = 26

def load_fild_mesh(filepath):
    """
    Loads the 3D wall geometry from the ASCOT simulation file and 
    isolates the FILD mesh using its specific flag.
    """
    h5 = Ascot(filepath)
    
    wall = h5.data.wall.active.read()
    x = np.array(wall['x1x2x3'], dtype=float)
    y = np.array(wall['y1y2y3'], dtype=float)
    z = np.array(wall['z1z2z3'], dtype=float)
    flag_array = wall["flag"].flatten()

    indices = np.where(flag_array == TARGET_FLAG)[0] 
    X = x[indices]
    Y = y[indices]
    Z = z[indices]

    AB_x = X[:, 1] - X[:, 0]
    AB_y = Y[:, 1] - Y[:, 0]
    AB_z = Z[:, 1] - Z[:, 0]
    
    AC_x = X[:, 2] - X[:, 0]
    AC_y = Y[:, 2] - Y[:, 0]
    AC_z = Z[:, 2] - Z[:, 0]
    
    CP_x = (AB_y * AC_z) - (AB_z * AC_y)
    CP_y = (AB_z * AC_x) - (AB_x * AC_z)
    CP_z = (AB_x * AC_y) - (AB_y * AC_x)
    areas = 0.5 * np.sqrt(CP_x**2 + CP_y**2 + CP_z**2) 

    Cx = np.mean(X, axis=1)
    Cy = np.mean(Y, axis=1)
    Cz = np.mean(Z, axis=1)
    puntos = np.column_stack((Cx, Cy, Cz)) 

    arbol_vec = cKDTree(puntos)

    mesh_data = {
        'x': x, 'y': y, 'z': z,
        'indices': indices,
        'areas': areas,
        'puntos': puntos,
        'Cx': Cx, 'Cy': Cy, 'Cz': Cz,
        'flag_array': flag_array
    }

    return mesh_data, arbol_vec

def extract_raw_power(filepath, mesh_data):
    """
    Extracts the unadjusted power loads mapped to the active FILD geometry.
    """
    h5 = Ascot(filepath)

    ids, area_wet_unyt, power_unyt, _, _ = h5.data.active.getwall_loads(weights=True, flags=TARGET_FLAG)
    itds = ids - 1 

    area_wet = np.array(area_wet_unyt, dtype=float)
    power = np.array(power_unyt, dtype=float)

    potencia_global = np.zeros(len(mesh_data['flag_array']))
    potencia_global[itds] = power
    potencia_fild_vec = potencia_global[mesh_data['indices']] 

    total_power_raw_val = np.sum(power)
    power_surface = power / area_wet 
    area_media = np.mean(area_wet)

    raw_data = {
        'itds': itds,
        'area_wet': area_wet,
        'power': power,
        'potencia_fild_vec': potencia_fild_vec,
        'total_power_raw_val': total_power_raw_val,
        'power_surface': power_surface,
        'area_media': area_media
    }

    return raw_data
