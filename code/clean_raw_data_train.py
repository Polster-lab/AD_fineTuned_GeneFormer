# %%
# https://github.com/mathyslab7/ROSMAP_snRNAseq_PFC/tree/main/Code/Scanpy_processing_data
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
import scrublet as scr
import warnings
warnings.filterwarnings('ignore')

adata = sc.read('PFC427_train.h5ad')
adata.raw = adata.copy()

# convert adata.raw.X to a CSR matrix if it's not already
if not isinstance(adata.raw.X, scipy.sparse.csr_matrix):
    print('Converting adata.raw.X to a CSR matrix')
    adata.raw = anndata.AnnData(X=scipy.sparse.csr_matrix(adata.raw.X), var=adata.raw.var, obs=adata.obs)

# %%
sc.settings.set_figure_params(dpi=500)
unique_batches = adata.obs['batch'].unique()
scrublet_results = {}

for batch in unique_batches:
    batch_adata = adata[adata.obs['batch'] == batch].copy()
    try:
        scrub_result = sc.external.pp.scrublet(batch_adata, expected_doublet_rate=0.045, n_prin_comps=20, copy=True)
        scrublet_results[batch] = scrub_result
        print(f"Processed batch {batch}\n")
        del batch_adata
        del scrub_result
    except Exception as e:
        print(f"Error processing batch {batch}:", e)

# initialize the columns if they don't exist
if 'doublet_score' not in adata.obs.columns:
    adata.obs['doublet_score'] = pd.NA
if 'predicted_doublet' not in adata.obs.columns:
    adata.obs['predicted_doublet'] = pd.NA

for batch, result_adata in scrublet_results.items():
    # extract the indices (cell identifiers) present in this batch's result
    # update the original adata.obs for these indices
    indices = result_adata.obs.index
    adata.obs.loc[indices, 'doublet_score'] = result_adata.obs['doublet_score']
    adata.obs.loc[indices, 'predicted_doublet'] = result_adata.obs['predicted_doublet']

adata.obs['predicted_doublet'] = adata.obs['predicted_doublet'].astype('category')

# %%
adata.obs['n_counts'] = adata.X.sum(axis=1)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key='batch')
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['n_counts'])
sc.pp.scale(adata, max_value=10)
print('Cleaned low quality cells.')

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=100, n_pcs=20)
sc.tl.leiden(adata, resolution=0.04, n_iterations=10)
sc.tl.umap(adata)  # Compute UMAP embedding
sc.pl.umap(adata, color=['leiden'], palette=sc.pl.palettes.default_20, save='_PFC_normalised.pdf')

adata = sc.AnnData(X=adata.raw[adata.obs.index, :].X, obs=adata.obs, var=adata.raw.var, obsm=adata.obsm, uns=adata.uns, obsp=adata.obsp)
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata.raw = adata.copy()

# %%
sc.pl.umap(adata, color=['predicted_doublet'], palette=sc.pl.palettes.default_20, save='_PFC427_doublets.pdf')
adata = adata[adata.obs["predicted_doublet"] == False, :]

# %%
adata = adata[adata.obs['predicted_doublet'].isin([False])]
adata = adata[adata.obs['leiden'].isin(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','30'])]

# %%
sc.pl.umap(adata, color='batch', palette='viridis', save='_PFC427_batch.pdf')

# %%
cols_to_remove = ['doublet_score', 'predicted_doublet']
adata.obs.drop(columns=cols_to_remove, inplace=True)
# drop .X values that no longer have any cells associated with them
adata = sc.AnnData(X=adata.raw[adata.obs.index, :].X, obs=adata.obs, var=adata.raw.var, obsm=adata.obsm, uns=adata.uns, obsp=adata.obsp)
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata.obs

# %%
adata.write('PFC427_train_clean.h5ad')

# %%
individual_metadata = pd.read_csv('individual_metadata_deidentified.tsv', sep='\t')
batch_metadata = pd.read_csv('batch_mapping_deidentified.tsv', sep='\t')

# subject where 'Pathologic_diagnosis_of_AD' is 'yes'
subjects_AD = individual_metadata[individual_metadata['Pathologic_diagnosis_of_AD'] == 'yes'].subject
subjects_AD_metadata = batch_metadata[batch_metadata.subject.isin(subjects_AD)].dataset
subjects_AD_dataset = adata.obs.index.map(lambda x: x.rsplit('_', 1)[0] + '-' + x.split('-')[-1])

# %%
print('For the AD population:\n')
total = adata.obs.loc[adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].count()
percentiles = adata.obs.loc[adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].value_counts(bins=6, sort=False)
print('Counts per bin (and population percentage):')
for bin, count in percentiles.items():
    print(f'{bin}: {count} ({(count / total) * 100:.2f}%)')
print()

print(adata.obs.loc[adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].describe())
print()

threshold = 0.1
high_mito_pct = (adata.obs.loc[adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'] > threshold).mean() * 100
print(f'{high_mito_pct:.2f}% of cells in the AD population have a high mitochondrial gene count (10% mean threshold).')

with open('AD_pct_mito.txt', 'w') as f:
    f.write('For the AD population:\n')
    f.write('Counts per bin (and population percentage):')
    for bin, count in percentiles.items():
        f.write(f'{bin}: {count} ({(count / total) * 100:.2f}%)\n')
    f.write('\n\n')

    f.write(str(adata.obs.loc[adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].describe()))
    f.write('\n\n')

    f.write(f'{high_mito_pct:.2f}% of cells in the AD population have a high mitochondrial gene count (10% mean threshold).\n')

# %%
print('For the NCI population:\n')
total = adata.obs.loc[~adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].count()
percentiles = adata.obs.loc[~adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].value_counts(bins=6, sort=False)
print('Counts per bin (and population percentage):')
for bin, count in percentiles.items():
    print(f'{bin}: {count} ({(count / total) * 100:.2f}%)')
print()

print(adata.obs.loc[~adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].describe())
print()

threshold = 0.1
high_mito_pct = (adata.obs.loc[~adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'] > threshold).mean() * 100
print(f'{high_mito_pct:.2f}% of cells in the NCI population have a high mitochondrial gene count (10% mean threshold).')

with open('NCI_pct_mito.txt', 'w') as f:
    f.write('For the NCI population:\n')
    f.write('Counts per bin (and population percentage):')
    for bin, count in percentiles.items():
        f.write(f'{bin}: {count} ({(count / total) * 100:.2f}%)\n')
    f.write('\n\n')

    f.write(str(adata.obs.loc[~adata.obs['Patient_Batch_ID'].isin(subjects_AD_metadata), 'pct_mito'].describe()))
    f.write('\n\n')

    f.write(f'{high_mito_pct:.2f}% of cells in the NCI population have a high mitochondrial gene count (10% mean threshold).\n')


