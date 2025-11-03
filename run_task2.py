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
    with open("/usr/local/bin/gene_order_sub.pkl", "rb") as f:
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

    # Load models
    class_model_6e10 = load_model("/usr/local/bin/class_model_6e10_sub.json")  
    reg_model_6e10 = load_model("/usr/local/bin/reg_model_6e10_sub.json")
    class_model_AT8 = load_model("/usr/local/bin/class_modelAT8_sub.json")
    reg_model_AT8 = load_model("/usr/local/bin/reg_model_AT8_sub.json")

    # ---- Clean up ------------------------------------------------------------
    try:
        gc.collect()
    except:
        pass

    # ===== FUNCTION FOR MAKING PREDICTIONS =====
    def predict_two_stage(class_model, reg_model, X_test, class_threshold=0.5):
        class_probs = class_model.predict(X_test)
        class_pred = (class_probs > class_threshold).astype(int)
        reg_pred = reg_model.predict(X_test)
        reg_pred_exp = np.exp(reg_pred)
        final_pred = np.where(class_pred == 0, 0, reg_pred_exp)
        return final_pred

    # ---- Process data in chunks and make predictions ------------------------
    print("Processing data in chunks of 50000 cells ...")

    # Initialize lists to store results
    all_pred_6e10 = []
    all_pred_AT8 = []
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
        pred_6e10_chunk = predict_two_stage(class_model_6e10, reg_model_6e10, dmatrix_chunk, class_threshold=0.5)
        pred_AT8_chunk = predict_two_stage(class_model_AT8, reg_model_AT8, dmatrix_chunk, class_threshold=0.5)
        
        # Get corresponding donor IDs for this chunk
        donor_ids_chunk = donor_ids[start_idx:end_idx]
        
        # Store results
        all_pred_6e10.extend(pred_6e10_chunk)
        all_pred_AT8.extend(pred_AT8_chunk)
        all_donor_ids.extend(donor_ids_chunk)
        
        # Update start index for next chunk
        start_idx = end_idx
        
        # Clean up memory
        del chunk_csr, dmatrix_chunk, pred_6e10_chunk, pred_AT8_chunk, donor_ids_chunk
        gc.collect()

    # Convert lists to numpy arrays
    all_pred_6e10 = np.array(all_pred_6e10)
    all_pred_AT8 = np.array(all_pred_AT8)
    all_donor_ids = np.array(all_donor_ids)

    print(f"Processed {len(all_pred_6e10)} total predictions")

    # ---- Build predictions dataframe ----------------------------------------
    print("Assembling predictions dataframe ...")
    predictions = pd.DataFrame({
        'Donor_ID': all_donor_ids,
        'predicted_6e10': all_pred_6e10,
        'predicted_AT8': all_pred_AT8,
        'predicted_GFAP': all_pred_6e10,  # Same as 6e10
        'predicted_NeuN': all_pred_AT8    # Same as AT8
    })

    # Aggregate by Donor_ID (get mode for each prediction type)
    print("Aggregating predictions by donor ...")
    predictions_by_donor = predictions.groupby('Donor_ID', observed=False).agg({
        'predicted_6e10': 'median',
        'predicted_AT8': 'median',
        'predicted_GFAP': 'median',
        'predicted_NeuN': 'median'
    }).reset_index()

    # Rename columns to match R output
    predictions_by_donor.columns = ['Donor ID', 'predicted 6e10', 'predicted AT8', 'predicted GFAP', 'predicted NeuN']

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