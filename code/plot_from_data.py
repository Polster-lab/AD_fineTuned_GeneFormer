import numpy as np
import pandas as pd
import scanpy as sc

adata = sc.read('PFC427_train_clean_anno.h5ad')
adata.raw = adata.copy()

# preprocess the data for UMAP
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=100, n_pcs=20)
sc.tl.leiden(adata)
sc.tl.umap(adata)
# visualize
sc.pl.umap(adata, color='Major_Cell_Type', title='Cell Type UMAP',
           frameon=False, legend_fontweight='normal', legend_fontsize=15, save='_cell_type_UMAP.pdf')

# visualize the expression of genes
# sc.pl.stacked_violin(adata, var_names=adata.var_names, groupby='Cell_Type')
