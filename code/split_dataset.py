# %%
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.model_selection import train_test_split


# %%
adata = sc.read('PFC427_raw_data.h5ad')
print(f"Number of rows: {adata.shape[0]}")
individual_metadata = pd.read_csv('individual_metadata_deidentified.tsv', sep='\t')
batch_metadata = pd.read_csv('batch_mapping_deidentified.tsv', sep='\t')

# %%
# locate AD patients
subjects_AD = individual_metadata[individual_metadata['Pathologic_diagnosis_of_AD'] == 'yes'].subject
subjects_AD_metadata = batch_metadata[batch_metadata.subject.isin(subjects_AD)].dataset
subjects_AD_dataset = adata.obs.index.map(lambda x: str(x).rsplit('_', 1)[0] + '-' + str(x).split('-')[-1])

# %%
adata.obs['Patient_Batch_ID'] = subjects_AD_dataset
# create dictionary that maps batches to patient IDs
id_dict = dict(zip(batch_metadata['dataset'], batch_metadata['subject']))
# map the 'Patient ID' from subjects_AD
adata.obs['Patient_ID'] = adata.obs['Patient_Batch_ID'].map(id_dict)
# map patient codes to batches, and batches to cell rows
adata.obs['AD_diagnosis'] = subjects_AD_dataset.isin(subjects_AD_metadata)

# %%
individual_metadata.set_index('subject', inplace=True)
adata.obs = adata.obs.join(individual_metadata, on='Patient_ID')

# %%
# identify rows with any NaN values in .obs
nan_rows = adata.obs.isna().any(axis=1)
adata = adata[~nan_rows]
print(f"Number of rows after dropping NaN values: {adata.shape[0]}")
print(adata.obs.head())

# %%
# split the dataset into training and testing sets by patient ID
# ensures that patient data is not split between training and testing sets
patient_ids = adata.obs['Patient_ID'].unique()
train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

# %%
adata[adata.obs['Patient_ID'].isin(train_patient_ids)].write('PFC427_raw_train.h5ad')

print('Saved.')