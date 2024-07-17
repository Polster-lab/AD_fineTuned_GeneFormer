import pandas as pd
import anndata


adata = anndata.read_h5ad('PFC427_raw_data.h5ad')

adata_obs_df = pd.DataFrame(adata.obs)
adata_var_df = pd.DataFrame(adata.var)

ensembl_ID_df = pd.read_csv('ensembl_ID_bioservices.csv')
ensembl_ID_df.set_index('external_gene_name', inplace=True)

# add the Ensembl IDs as a new column in adata.var
# drop the rows that do not have an Ensembl ID
# subset the adata object to include only the remaining variables
adata.var['ensembl_id'] = ensembl_ID_df.loc[adata.var.index, 'ensembl_gene_id']
adata.var.dropna(subset=['ensembl_id'], inplace=True)
adata = adata[:, adata.var.index]

adata.write('PFC427_raw_data_with_ensembl.h5ad')

print(adata)
