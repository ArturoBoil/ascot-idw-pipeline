import argparse
import sys

# Import custom modules for the processing pipeline
from geometry import load_fild_mesh, extract_raw_power
from physics import calculate_larmor_bounds, get_equivalent_k_range
from reconstruction import optimize_smoothing, final_reconstruction
from plotting import generate_plots, print_benchmark_tables

def main():
    # Configure command-line arguments
    parser = argparse.ArgumentParser(
        description="FILD Synthetic Smoothing and Signal Reconstruction for ASCOT outputs."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the ASCOT HDF5 simulation file"
    )
    args = parser.parse_args()

    print(f"Initializing processing for: {args.input}")

    # Sequential execution flow
    try:
        # 1. Geometry and RAW Data Extraction
        print("Step 1: Extracting geometry and raw power data")
        mesh_data, arbol_vec = load_fild_mesh(args.input)
        raw_data = extract_raw_power(args.input, mesh_data)

        # 2. Physics Constraints
        print("Step 2: Calculating physical constraints (Larmor radius)")
        rl_bounds = calculate_larmor_bounds(args.input)
        k_range = get_equivalent_k_range(rl_bounds, arbol_vec, mesh_data)

        # 3. Synthetic Optimization and Signal Reconstruction
        print("Step 3: Optimizing smoothing parameters and reconstructing signal")
        best_k, stats = optimize_smoothing(mesh_data, raw_data, k_range, arbol_vec)
        final_signals = final_reconstruction(mesh_data, raw_data, arbol_vec, best_k)

        # 4. Results and Visualizations
        print("Step 4: Generating benchmark tables and visualizations")
        print_benchmark_tables(raw_data, final_signals, stats)
        generate_plots(mesh_data, raw_data, final_signals, stats)

        print("Pipeline executed successfully.")

    except Exception as e:
        print(f"Error: Pipeline execution failed. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
