from bioservices import BioMart
import pandas as pd
import os


filename = 'PFC427_test_clean'
df = pd.read_csv(filename+'.csv')
df.rename(columns={'Unnamed: 0': 'gene_name'}, inplace=True)
df

# initialize BioMart service
mart = BioMart()

# check available datasets (for confirmation)
datasets = mart.get_datasets("ENSEMBL_MART_ENSEMBL")
print("Available datasets:\n", datasets)

# start query for human genes
mart.new_query()
mart.add_dataset_to_xml("hsapiens_gene_ensembl")

# add attributes and filters
mart.add_attribute_to_xml("hgnc_symbol")
mart.add_attribute_to_xml("ensembl_gene_id")

gene_names = df['gene_name'].tolist()
gene_names = ",".join(gene_names)
mart.add_filter_to_xml("hgnc_symbol", gene_names)

# generate XML query and print (to check if it's being generated correctly)
xml_query = mart.get_xml()
print("Generated XML query:\n", xml_query)

try:
    response = mart.query(xml_query)
    print("Response received:\n", response)
except Exception as e:
    print(f"Error in querying BioMart: {e}")

# split response into lines, then split each line by tab
data = [line.split("\t") for line in response.strip().split("\n")]
response_df = pd.DataFrame(data, columns=['gene_name', 'ensembl_gene_id'])

# drop duplicates, merge with left join
response_df = response_df.drop_duplicates(subset='gene_name')
df_ensembl = pd.merge(df, response_df, on='gene_name', how='left')
df_ensembl

# save to csv
df_ensembl.to_csv(filename+'_ensembl.csv', index=False)