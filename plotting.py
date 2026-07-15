import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def print_benchmark_tables(raw_data, final_signal, stats):
    """
    Prints benchmark metrics to validate energy conservation and model accuracy.
    """
    total_power = raw_data['total_power_raw_val']
    area_wet = raw_data['area_wet']
    
    print("\n" + "="*50)
    print("ENERGY CONSERVATION METRICS")
    print("="*50)
    
    E_post = np.sum(final_signal * area_wet)
    err = (E_post - total_power) / total_power * 100
    
    print(f"{'Input (Raw)':<15} | {total_power:.4e} W | Err: +0.0000%")
    print(f"{'Predicted (IDW)':<15} | {E_post:.4e} W | Err: {err:+.4f}%")
    
    print("\n" + "="*50)
    print("HOTSPOT REDUCTION (NOISE FILTERING)")
    print("="*50)
    
    p_raw = np.max(raw_data['power_surface'])
    p_idw = np.max(final_signal)
    
    print(f"Max Density RAW: {p_raw:.2e} W/m2")
    print(f"Max Density IDW: {p_idw:.2e} W/m2")
    print(f"Attenuation:     {(1 - p_idw/p_raw) * 100:.2f}%")


def generate_plots(mesh_data, raw_data, final_signal, stats=None):
    """
    Generates Data Science oriented visualizations for model evaluation 
    and compares real physical data before and after processing.
    """
    print("\nGenerating model evaluation plots...")
    
    x_global = mesh_data['x']
    y_global = mesh_data['y']
    itds = raw_data['itds']
    
    Cx_w = np.mean(x_global[itds], axis=1)
    Cy_w = np.mean(y_global[itds], axis=1)
    triang = tri.Triangulation(Cx_w, Cy_w)
    
    # 1. OPTIMIZATION CURVE (HYPERPARAMETER TUNING)
    if stats is not None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(stats['lista_k'], stats['mejoras'], 'b-o', label='Accuracy Improvement (%)')
        ax1.axvline(stats['best_k'], c='g', ls='--', label=f"Optimal K={stats['best_k']}")
        
        ax1.axvspan(stats['k_flat_min'], stats['k_flat_max'], color='green', alpha=0.15, 
                    label='Stability Region (0.2% Tol)')
        ax1.axhline(stats['umbral_flat'], color='orange', ls=':', label='Tolerance Threshold')
        
        ax1.set_xlabel('K-Neighbors (Hyperparameter)')
        ax1.set_ylabel('Accuracy Improvement (%)', color='b')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')

        # Secondary X-Axis for physical equivalent
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        tick_pos = np.linspace(min(stats['lista_k']), max(stats['lista_k']), 5)
        tick_lab = np.interp(tick_pos, stats['lista_k'], stats['radios_eq'])
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"{t*1000:.1f}mm" for t in tick_lab])
        ax2.set_xlabel("Equivalent Spatial Feature Size")
        
        plt.title("Hyperparameter Tuning: KNN Spatial Smoothing", pad=15)
        plt.tight_layout()
        plt.savefig('Hyperparameter_Optimization.png', dpi=300)
        plt.close()

        # 2. SYNTHETIC 2D MAPS COMPARISON
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        titles = ["A) Target (Ground Truth)", "B) Noisy Input (Raw)", "C) IDW Prediction"]
        datasets = [stats['truth_map'], stats['raw_map'], stats['pred_map']]
        
        for i in range(3):
            ax[i].tricontourf(triang, datasets[i], levels=50, cmap='jet')
            ax[i].triplot(triang, 'k-', lw=0.1, alpha=0.2)
            ax[i].set_title(titles[i])
            ax[i].set_aspect('equal')
            
        plt.tight_layout()
        plt.savefig('Synthetic_2D_Evaluation.png', dpi=300)
        plt.close()

    # 3. REAL PHYSICAL DATA COMPARISON (RAW VS IDW)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    raw_surface = raw_data['power_surface']
    
    im0 = ax[0].tripcolor(triang, raw_surface, cmap='jet', shading='flat')
    ax[0].triplot(triang, 'k-', lw=0.1, alpha=0.2)
    ax[0].set_title("A) Real Noisy Input (Raw Physical Data)")
    ax[0].set_aspect('equal')
    plt.colorbar(im0, ax=ax[0], label="Power Density (W/m2)")
    
    im1 = ax[1].tricontourf(triang, final_signal, levels=100, cmap='jet')
    ax[1].triplot(triang, 'k-', lw=0.1, alpha=0.2)
    ax[1].set_title("B) Smoothed Output (IDW Prediction)")
    ax[1].set_aspect('equal')
    plt.colorbar(im1, ax=ax[1], label="Power Density (W/m2)")
    
    plt.suptitle("Real Data Processing: Noise Reduction on Active Simulation", fontsize=14)
    plt.tight_layout()
    plt.savefig('Real_Data_Evaluation.png', dpi=300)
    plt.close()
