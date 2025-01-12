# %%
import numpy as np
import pandas as pd
import scanpy as sc

adata = sc.read('PFC427_train_clean.h5ad')
correct_metadata = pd.read_csv('correct_metadata.tsv', sep='\t', index_col=0)

# %%
adata.obs['subject_cell_id'] = adata.obs['Patient_ID'].astype(str) + '_' + adata.obs['bc'].astype(str)
adata.obs.index = adata.obs.index.rename('cellName')
adata.obs

# %%
correct_metadata = correct_metadata.rename(columns={'Batch':'batch', 'percent_ribo':'pct_ribo', 'percent_mito':'pct_mito', 'Individual':'Patient_ID', 'Dataset':'Patient_Batch_ID'})
correct_metadata.drop(columns=['n_counts', 'n_genes', 'pct_ribo', 'pct_mito'], inplace=True)
correct_metadata

# %%
adata = adata[adata.obs.index.isin(correct_metadata.index), :]
cols_to_use = correct_metadata.columns.difference(adata.obs.columns)
adata.obs = adata.obs.join(correct_metadata[cols_to_use])
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs

# %%
total_obs = len(set(adata.obs.index))
total_correct = len(set(correct_metadata.index))

common = adata.obs.index.isin(correct_metadata.index).sum()

missing_obs = common - total_obs
missing_correct = common - total_correct

missing_obs_percentage = (missing_obs / total_obs) * 100
missing_correct_percentage = (missing_correct / total_correct) * 100

print(f"Total in obs_metadata: {total_obs}")
print(f"Total in correct_metadata: {total_correct}")
print(f"Common: {common}")
print(f"Missing in obs_metadata: {missing_obs} ({missing_obs_percentage:.2f}%)")
print(f"Missing in correct_metadata: {missing_correct} ({missing_correct_percentage:.2f}%)")


adata.write('PFC427_train_clean_anno.h5ad')

print(adata)
print()
print(adata.obs.head())
print()
print(adata.X)

sc.pl.umap(adata, color='Major_Cell_Type', title='Cell Type UMAP',
           frameon=False, legend_fontweight='normal', legend_fontsize=15, save='_cell_type_UMAP.pdf')
