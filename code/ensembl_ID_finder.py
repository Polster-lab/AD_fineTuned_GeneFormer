from bioservices import BioMart
import pandas as pd
import anndata

adata = anndata.read_h5ad('PFC427.h5ad')

adata_obs_df = pd.DataFrame(adata.obs)
adata_var_df = pd.DataFrame(adata.var)

bm = BioMart()
bm.registry()
bm.database = 'ensembl'
bm.dataset = 'hsapiens_gene_ensembl'

# get a list of gene names from the index of the var DataFrame
gene_names = adata.var.index.tolist()

xml_query = """
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
    <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
        <Filter name = "external_gene_name" value = "{}"/>
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "external_gene_name" />
    </Dataset>
</Query>
""".format(",".join(gene_names))

response = bm.query(xml_query)

# the response is a string with each line corresponding to a gene. we can split it into lines and then split each line into columns to create a DataFrame.
lines = response.split("\n")
data = [line.split("\t") for line in lines if line]
result_df = pd.DataFrame(data, columns=["ensembl_gene_id", "external_gene_name"])
# remove duplicates from result_df
result_df = result_df.drop_duplicates('external_gene_name')

# left join to include all gene names from adata.var
merged_df = pd.merge(adata.var, result_df, how='left', left_on=adata.var.index, right_on='external_gene_name')

print("Merged DataFrame:")
print(merged_df)

merged_df.to_csv('ensembl_ID_bioservices.csv', index=False)