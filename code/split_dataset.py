# %%
# https://github.com/mathyslab7/ROSMAP_snRNAseq_PFC/tree/main/Code/Scanpy_processing_data
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
import scrublet as scr
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

adata = sc.read('PFC427_raw_data_with_ensembl.h5ad')

# %%
individual_metadata = pd.read_csv('individual_metadata_deidentified.tsv', sep='\t')
batch_metadata = pd.read_csv('batch_mapping_deidentified.tsv', sep='\t')

# locate AD patients
subjects_AD = individual_metadata[individual_metadata['Pathologic_diagnosis_of_AD'] == 'yes'].subject
subjects_AD_metadata = batch_metadata[batch_metadata.subject.isin(subjects_AD)].dataset
subjects_AD_dataset = adata.obs.index.map(lambda x: x.rsplit('_', 1)[0] + '-' + x.split('-')[-1])

adata.obs['Patient_Batch_ID'] = subjects_AD_dataset
# create dictionary that maps batches to patient IDs
id_dict = dict(zip(batch_metadata['dataset'], batch_metadata['subject']))
# map the 'Patient ID' from subjects_AD
adata.obs['Patient_ID'] = adata.obs['Patient_Batch_ID'].map(id_dict)
# map patient codes to batches, and batches to cell rows
adata.obs['AD_diagnosis'] = subjects_AD_dataset.isin(subjects_AD_metadata)

individual_metadata.set_index('subject', inplace=True)
adata.obs = adata.obs.join(individual_metadata, on='Patient_ID')

# %%
# split the dataset into training and testing sets by patient ID
# ensures that patient data is not split between training and testing sets
patient_ids = adata.obs['Patient_ID'].unique()

train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

train_set = adata[adata.obs['Patient_ID'].isin(train_patient_ids)]
test_set = adata[adata.obs['Patient_ID'].isin(test_patient_ids)]

train_set.write('PFC427_train.h5ad')
test_set.write('PFC427_test.h5ad')

