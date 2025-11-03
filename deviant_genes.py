import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import anndata2ri
import argparse
# Import R packages
scry = importr('scry')

def converter():
    """Local conversion context for pandas + AnnData <-> R."""
    return ro.conversion.localconverter(ro.default_converter + anndata2ri.converter)

def elbow_chord(xs: np.ndarray, ys: np.ndarray):
    """
    Find elbow index using the 'maximum distance to the line between endpoints' (chord) method.
    xs: 1D array of x values (e.g., 1..n)
    ys: 1D array of y values sorted in descending order
    Returns: zero-based elbow index
    """
    # endpoints
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]

    # distances from points to the line through (x0,y0)-(x1,y1)
    num = np.abs((y1 - y0) * xs - (x1 - x0) * ys + x1*y0 - y1*x0)
    den = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    d = num / (den + 1e-12)

    return int(np.argmax(d))

p = argparse.ArgumentParser(description="Pipeline for hd5ad")
p.add_argument("--files", nargs="+", help="Paths to .h5ad files")
p.add_argument("--subset", type=str, default="Supertype", help="Subset by provided metadat column.")
p.add_argument("--outdir", default=".", help="Output directory (default: current)")
args = p.parse_args()

print(f"Running mode: {args.subset}")
print(f"Output directory: {args.outdir}")

for file in args.files:
    # h5ad = sc.read_h5ad(file)
    print(f"Reading {file}")
    h5ad = sc.read_h5ad(file)

    metadata = h5ad.obs

    if args.subset in metadata.columns:
        print("Column 'Supertype' is in metadata.")
        print("Subsetting by 'Supertype'.")

        subclass = metadata["Supertype"]
        subclasses = subclass.unique().astype(str)

        print(f"Found {len(subclasses)} subclasses.")

        results = {}
        for x in subclasses:
            print(f"Calculating deviance for {x}")
            sub = h5ad[metadata["Supertype"] == x].X.T
            with converter():
                binom_deviance = scry.devianceFeatureSelection(sub)
            results[x] = binom_deviance

        s = pd.DataFrame(results, index=h5ad.var_names)
        print("\nDeviances calculated.\n")

        med_col = s.median(axis=1)
        sum_col = s.sum(axis=1)

        med_col_ordered = med_col.sort_values(ascending=False)
        sum_col_ordered = sum_col.sort_values(ascending=False)

        ranks_dense = s.rank(ascending=False, method="dense", na_option="bottom").astype("Int64")

        out_file_median = os.path.join(
            args.outdir,
            f"{os.path.basename(file)}_median_order_deviance.csv.gz"
        )
        s.loc[med_col_ordered.index].to_csv(out_file_median, index=True, compression='gzip')

        print(f"Saved: {out_file_median}")

        out_file_sum = os.path.join(
            args.outdir,
            f"{os.path.basename(file)}_sum_order_deviance.csv.gz"
        )
        s.loc[sum_col_ordered.index].to_csv(out_file_sum, index=True, compression='gzip')

        print(f"Saved: {out_file_sum}")

        out_file_rank = os.path.join(
            args.outdir,
            f"{os.path.basename(file)}_rank_deviance.csv.gz"
        )
        ranks_dense.to_csv(out_file_rank, index=True, compression='gzip')

        print(f"Saved: {out_file_rank}")
