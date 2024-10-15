# %%
import scanpy as sc
import anndata as ad

# %%
train_adata = sc.read('PFC427_train_clean_anno_sample.h5ad')
test_adata = sc.read('PFC427_test_clean_anno_sample.h5ad')

# %%
train_adata.obs['dataset'] = 'train'
test_adata.obs['dataset'] = 'test'

train_adata.obs.head(2)

# %%
# concatenate the AnnData objects, keeping everything crucial
merged_adata = ad.concat(
    [train_adata, test_adata],   
    axis=0,                      
    join="outer",                
    merge="same",               
    uns_merge="same",            
    index_unique=None
)

# %%
# duplicate columns necessary for later use
merged_adata.obs['Major Cell Type'] = merged_adata.obs['Major_Cell_Type']
merged_adata.obs['AD diagnosis'] = merged_adata.obs['AD_diagnosis'].astype(str).str.lower()

merged_adata.obs

# %%
merged_adata.write('PFC427_clean_anno_sample.h5ad')
