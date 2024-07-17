import scanpy as sc
import pandas as pd

sample = pd.read_csv('df_resampled.csv')
adata = sc.read('PFC427_train_clean_anno.h5ad')

# get the intersection of the indices
common_indices = adata.obs.merge(sample, how='inner', on='obsnames').obsnames

# check common indices
if common_indices.empty:
    print("No common indices between adata.obs['obsnames'] and sample.")
else:
    print(f"Found {len(common_indices)} common indices between adata.obs['obsnames'] and sample.")
    adata = adata[adata.obs['obsnames'].isin(common_indices), :]

# check NaN values in .obs
if adata.obs.isna().any().any():
    print("NaN values found in adata.obs.")
    adata.obs.dropna(inplace=True)
    adata = adata[adata.obs.index, :]

adata.write('PFC427_train_clean_anno_sample.h5ad')
