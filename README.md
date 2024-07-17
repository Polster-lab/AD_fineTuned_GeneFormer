### Temporary README file

#### Order of operations
ensembl_ID_finder.py - Find EnsemblIDs, necessary for Geneformer

convert_gene_name.py - Map gene names in .var of the h5ad dataset to ensemblIDs

split_dataset.py - Split .h5ad dataset to training and testing sets before preprocessing. Split is based on Patient ID, so that there is no patient data leakage between sets

clean_raw_data_train/test.py - Adjusted cleaning pipeline from ROSMAP repository files

map_metadata.py - Maps extra metadata to cleaned .h5ad files. Since the extra metadata only applies to ROSMAPs cleaned data, our result can be at best the same as ROSMAPs (realistically less data will remain)

undersampling.py - Takes a stratified .obs sample from the training dataset, as it is too large otherwise. 

sample_data.py - Samples from the training dataset based on the file created by undersampling.py

tokenise_data.py - Tokenise data using Geneformer API

hyperparameter_tuning.py - Fine-tune Geneformer. Uses a custom_geneformer.py file that alters plotting and training functions slightly
