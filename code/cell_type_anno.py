# %%
# https://github.com/mathyslab7/ROSMAP_snRNAseq_PFC/tree/main/Code/Scanpy_processing_data
import copy
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
import scrublet as scr
import json
import decoupler as dc
import warnings
warnings.filterwarnings('ignore')

adata = sc.read('PFC427_train_clean.h5ad')

# %%
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata.raw = adata.copy()

# %%
# prepare data and plot UMAP
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['log_norm'] = adata.X.copy()
sc.tl.pca(adata, svd_solver='arpack')
dc.swap_layer(adata, 'log_norm', X_layer_key=None, inplace=True)
sc.pp.neighbors(adata, n_neighbors=100, n_pcs=20)
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden', title='RNA UMAP',
           frameon=False, legend_fontweight='normal', legend_fontsize=15, save='_RNA_UMAP.pdf')


# %%
# query Omnipath and get PanglaoDB
markers = dc.get_resource('PanglaoDB')

markers['human'] = markers['human'].astype(bool)
markers['canonical_marker'] = markers['canonical_marker'].astype(bool)
markers['human_sensitivity'] = markers['human_sensitivity'].astype(float)

markers = markers[markers['human'] & markers['canonical_marker'] & (markers['human_sensitivity'] > 0.5)]
markers = markers[~markers.duplicated(['cell_type', 'genesymbol'])]

cell_types = pd.read_csv('Cell_types.csv')
unique_cell_types = pd.concat([cell_types['cell_type_high_resolution'], cell_types['major_cell_type']]).unique()
regex = '|'.join(unique_cell_types)
markers = markers[markers['cell_type'].str.contains(regex, case=False, na=False) | markers['genesymbol'].str.contains(regex, case=False, na=False)]
markers

# %%
dc.run_ora(
    mat=adata,
    net=markers,
    source='cell_type',
    target='genesymbol',
    min_n=0,
    verbose=True,
    use_raw=False
)

adata.obsm['ora_estimate']

# %%
acts = dc.get_acts(adata, obsm_key='ora_estimate')

# We need to remove inf and set them to the maximum value observed for pvals=0
acts_v = acts.X.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
acts.X[~np.isfinite(acts.X)] = max_e

acts

# %%
sc.pl.umap(acts, color=['Astrocytes', 'leiden'], cmap='RdBu_r', save='_Astrocytes.pdf')
sc.pl.violin(acts, keys=['Astrocytes'], groupby='leiden', save='_Astrocytes.pdf')

# %%
anno = dc.rank_sources_groups(acts, groupby='leiden', reference='rest', method='t-test_overestim_var')
anno

# %%
n_ctypes = 3
ctypes_dict = anno.groupby('group').head(n_ctypes).groupby('group')['names'].apply(lambda x: list(x)).to_dict()
print(ctypes_dict)

with open('ctypes_dict.json', 'w') as f:
    json.dump(ctypes_dict, f)

# %%
sc.pl.matrixplot(acts, ctypes_dict, 'leiden', dendrogram=True, standard_scale='var',
                 colorbar_title='Z-scaled scores', cmap='RdBu_r', save='_Astrocytes.pdf')

# %%
sc.pl.violin(acts, keys=['Astrocytes'], groupby='leiden')

# %%
annotation_dict = anno.groupby('group').head(1).set_index('group')['names'].to_dict()
annotation_dict

# %%
# Add cell type column based on annotation
adata.obs['cell_type'] = [annotation_dict[clust] for clust in adata.obs['leiden']]

# Visualize
sc.pl.umap(adata, color='cell_type', save='_cell_type.pdf')

# %%
# To find marker genes for each cluster
sc.tl.rank_genes_groups(adata, groupby='leiden', method='t-test')

marker_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(1).to_numpy().flatten()
sc.pl.dotplot(adata, marker_genes, groupby='leiden', save='rank_genes_groups.pdf')

sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', swap_axes=True, save='rank_genes_groups.pdf')

# %%
adata = sc.AnnData(X=adata.raw.X, var=adata.raw.var, obs=adata.obs)
print(adata.X)
adata.write('PFC427_train_clean_anno.h5ad')


