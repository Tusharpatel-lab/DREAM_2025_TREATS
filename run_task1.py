#!/usr/bin/env python3

import argparse
import os
import sys
import gc
import warnings
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import sparse
from scipy.sparse import csr_matrix
import xgboost as xgb
from collections import Counter

warnings.filterwarnings('ignore')

def main(): 
    # ---- CLI options ----------------------------------------------------------
    parser = argparse.ArgumentParser(description='Memory-efficient ML pipeline')
    parser.add_argument('--input', type=str, default='/input',
                    help='Input directory [default=/input]')
    parser.add_argument('--output', type=str, default='/output',
                    help='Output directory [default=/output]')
    args = parser.parse_args()

    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")

    # ---- Verify input --------------------------------------------------------
    input_file = os.path.join(args.input, "data.h5ad")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # ---- Read data ------------------------------------------------------------
    print("Reading .h5ad in backed (on-disk) mode ...")
    adata = ad.read_h5ad(input_file, backed='r')

    # ---- Get gene order early and create mapping ----------------------------
    with open("/usr/local/bin/gene_order.pkl", "rb") as f:
        gene_order = pickle.load(f)

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    print(f"Cells: {n_cells}, Genes: {n_genes}")

    # Create gene mapping for reordering chunks
    print("Creating gene order mapping ...")
    adata_gene_names = adata.var_names.tolist()
    gene_indices = [adata_gene_names.index(gene) for gene in gene_order if gene in adata_gene_names]
    print(f"Found {len(gene_indices)} genes from gene_order in the dataset")

    # Cells ID
    cell_names = adata.obs_names.tolist()

    # ---- Process metadata (lightweight) --------------------------------------
    # Convert metadata to pandas DataFrame
    matdata = adata.obs.copy()

    # Sanitize column names
    if any(char in col for col in matdata.columns for char in ['/','\\', ' ', '.']):
        print("Sanitizing metadata column names ...")
        matdata.columns = [col.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('.', '_')
                        for col in matdata.columns]

    # Align metadata
    matdata.index = cell_names
    if not all(matdata.index == cell_names):
        matdata = matdata.loc[cell_names]

    # ---- Extract Donor IDs ---------------------------------------------------
    print("Extracting Donor IDs ...")
    donor_ids = matdata['Donor_ID'].values

    # ---- Load models and levels ----------------------------------------------
    print("Loading models ...")
    def load_model(model_path):
        """Load XGBoost model"""
        model = xgb.Booster()
        model.load_model(model_path)
        return model

    def load_pickle(file_path):
        """Load pickle file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # Load models
    adnc_model = load_model("/usr/local/bin/Submission_model1_ADNC.json") 
    braak_model = load_model("/usr/local/bin/Submission_model1_Braak.json")
    cerad_model = load_model("/usr/local/bin/Submission_model1_CERAD.json")
    thal_model = load_model("/usr/local/bin/Submission_model1_Thal.json")
    
    # Load levels
    levels_used_adnc = load_pickle("/usr/local/bin/Submission_model1_ADNC_level.pkl")
    levels_used_braak = load_pickle("/usr/local/bin/Submission_model1_Braak_level.pkl")
    levels_used_cerad = load_pickle("/usr/local/bin/Submission_model1_CERAD_level.pkl")
    levels_used_thal = load_pickle("/usr/local/bin/Submission_model1_Thal_level.pkl")

    # ---- Clean up ------------------------------------------------------------
    try:
        gc.collect()
    except:
        pass

    # ===== FUNCTION FOR MAKING PREDICTIONS =====

    def predict_and_process(model, dmatrix, levels_used, prediction_name, label_mapping):
        print(f"Predicting {prediction_name} ...")
        pred_probs = model.predict(dmatrix)
        
        if len(pred_probs.shape) == 1:
            pred_probs = pred_probs.reshape(-1, len(levels_used))
        
        pred_labels = np.argmax(pred_probs, axis=1)
        predicted_original = [levels_used[label] for label in pred_labels]
        predicted_mapped = [label_mapping.get(str(pred), str(pred)) for pred in predicted_original]

        return predicted_mapped

    # Define the mappings
    adnc_mapping = {
        "1": "Not AD", "2": "Low", "3": "Intermediate", "4": "High", 
        1: "Not AD", 2: "Low", 3: "Intermediate", 4: "High"
    }
    
    braak_mapping = {
        "1": "Braak 0", "2": "Braak II", "3": "Braak III",
        "4": "Braak IV", "5": "Braak V", "6": "Braak VI",
        1: "Braak 0", 2: "Braak II", 3: "Braak III",
        4: "Braak IV", 5: "Braak V", 6: "Braak VI"
    }
    
    cerad_mapping = {
        "1": "Absent", "2": "Sparse", "3": "Moderate", "4": "Frequent",
        1: "Absent", 2: "Sparse", 3: "Moderate", 4: "Frequent"
    }
    
    thal_mapping = {
        "1": "Thal 0", "2": "Thal 1", "3": "Thal 2",
        "4": "Thal 3", "5": "Thal 4", "6": "Thal 5",
        1: "Thal 0", 2: "Thal 1", 3: "Thal 2",
        4: "Thal 3", 5: "Thal 4", 6: "Thal 5"
    }

    # ---- Process data in chunks and make predictions ------------------------
    print("Processing data in chunks of 50000 cells ...")

    # Initialize lists to store results
    all_pred_ADNC = []
    all_pred_THAL = []
    all_pred_BRAAK = []
    all_pred_CERAD = []
    all_donor_ids = []

    chunk_size = 50000

    n_chunks = (n_cells + chunk_size - 1) // chunk_size  # Ceiling division
    print(f"Will process {n_chunks} chunks")

    for i in range(n_chunks):

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_cells)

        print(f"Processing chunk {i+1}/{n_chunks}: cells {start_idx} to {end_idx-1} ...")
        
        # Extract chunk using slicing (this should work with backed mode)
        chunk = adata.X[start_idx:end_idx, :]

        # Sort genes in this chunk according to gene_order
        #print(f"  Sorting genes in chunk ...")

        chunk_csr = chunk.tocsr()    
        
        # Reorder genes in the chunk
        chunk_csr_sorted = chunk_csr[:, gene_indices]

        dmatrix_chunk = xgb.DMatrix(chunk_csr_sorted)
        
        # Make predictions for this chunk
        predicted_adnc_mapped = predict_and_process(adnc_model, dmatrix_chunk, levels_used_adnc, "ADNC", adnc_mapping)
        predicted_thal_mapped = predict_and_process(thal_model, dmatrix_chunk, levels_used_thal, "Thal", thal_mapping)
        predicted_braak_mapped = predict_and_process(braak_model, dmatrix_chunk, levels_used_braak, "Braak", braak_mapping)
        predicted_cerad_mapped = predict_and_process(cerad_model, dmatrix_chunk, levels_used_cerad, "CERAD", cerad_mapping)
        
        # Get corresponding donor IDs for this chunk
        donor_ids_chunk = donor_ids[start_idx:end_idx]
        
        # Store results
        all_pred_ADNC.extend(predicted_adnc_mapped)
        all_pred_THAL.extend(predicted_thal_mapped)
        all_pred_BRAAK.extend(predicted_braak_mapped)
        all_pred_CERAD.extend(predicted_cerad_mapped)
        all_donor_ids.extend(donor_ids_chunk)
        
        # Update start index for next chunk
        start_idx = end_idx
        
        # Clean up memory
        del chunk_csr, dmatrix_chunk, predicted_adnc_mapped, predicted_thal_mapped, predicted_braak_mapped, predicted_cerad_mapped, donor_ids_chunk
        gc.collect()

    # Convert lists to numpy arrays
    all_pred_ADNC = np.array(all_pred_ADNC)
    all_pred_THAL = np.array(all_pred_THAL)
    all_pred_BRAAK = np.array(all_pred_BRAAK)
    all_pred_CERAD = np.array(all_pred_CERAD)
    all_donor_ids = np.array(all_donor_ids)

    print(f"Processed {len(all_pred_ADNC)} total predictions")

    # ---- Build predictions dataframe ----------------------------------------
    print("Assembling predictions dataframe ...")
    predictions = pd.DataFrame({
        'Donor_ID': all_donor_ids,
        'predicted_ADNC': all_pred_ADNC,
        'predicted_THAL': all_pred_THAL,
        'predicted_BRAAK': all_pred_BRAAK, 
        'predicted_CERAD': all_pred_CERAD
    })

    # Get mode function
    def get_mode(series):
        """Get the most common value in a series"""
        counts = Counter(series)
        return counts.most_common(1)[0][0]

    # Aggregate by Donor_ID (get mode for each prediction type)
    print("Aggregating predictions by donor ...")
    predictions_by_donor = predictions.groupby('Donor_ID', observed=False).agg({
        'predicted_ADNC': get_mode,
        'predicted_THAL': get_mode,
        'predicted_BRAAK': get_mode,
        'predicted_CERAD': get_mode
    }).reset_index()

    # Rename columns to match R output
    predictions_by_donor.columns = ['Donor ID', 'predicted ADNC', 'predicted Thal', 'predicted Braak', 'predicted CERAD']

    # ---- Write output --------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "predictions.csv")
    print(f"Writing predictions to: {output_file}")
    predictions_by_donor.to_csv(output_file, index=False)
    
    print(f"Processing completed successfully! Number of predictions: {len(predictions_by_donor)}")
    
    # Final garbage collection
    gc.collect()

if __name__ == "__main__":
    main()