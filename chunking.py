import scanpy as sc
import numpy as np
import re

# Load dataset in backed mode
adata = sc.read_h5ad("./SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad", backed="r")
# Clean obs column names globally
adata.obs.columns = [re.sub(r"[\/]", "_", col) for col in adata.obs.columns]

chunk_size = 50000   # adjust to your RAM
n_cells = adata.n_obs
n_chunks = int(np.ceil(n_cells / chunk_size))
print(f"Splitting into {n_chunks} chunks of ~{chunk_size} cells each")

for i in range(n_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, n_cells)

    print(f"Processing chunk {i+1}/{n_chunks}: cells {start}–{end}")

    adata_chunk = adata[start:end, :].to_memory()
    adata_chunk.write_h5ad(f"./SEAAD_A9_chunk_{i+1}.h5ad")
