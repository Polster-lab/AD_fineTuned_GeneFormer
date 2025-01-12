# Gene Regulatory Networks Inference using Bidirectional Encoder Representations from Transformers

This project applies **Geneformer**—a BERT-like transformer model pre-trained on millions of single-cell transcriptomes—to **single-cell RNA sequencing data** from the ROSMAP study. Our primary goals are:

- **Cell Type Classification**: Fine-tune Geneformer to distinguish major cell types (e.g., neurons, microglia).
- **Alzheimer’s Disease (AD) Classification**: Attempt to classify AD vs. non-AD patients based on scRNA-seq of the prefrontal cortex.

While the project references the complete data and pipeline described in [Mathys et al., *Cell*, 2023](#rosmap-paper) (the ROSMAP paper) and [Theodoris et al., *Nature*, 2023](#geneformer-paper) (the Geneformer paper), **we are only providing partial code** here. The code focuses on the *practical steps* used to prepare and tokenize data before fine-tuning Geneformer.

# Order of Operations

Below is the sequence of scripts used. Note that each script may depend on prior outputs and is built for a specific stage of the pipeline:

1. **`ensembl_ID_finder.py`**  
   - Finds **Ensembl IDs** (required by Geneformer).

2. **`convert_gene_name.py`**  
   - Maps gene names in the `.var` of the `.h5ad` dataset to Ensembl IDs.

3. **`split_dataset.py`**  
   - Splits the `.h5ad` file into **training** and **testing** sets prior to preprocessing.  
   - Splitting is done *by patient ID* to avoid data leakage.

4. **`clean_raw_data_train.py`** and **`clean_raw_data_test.py`**  
   - Applies an adapted **cleaning pipeline** based on the methodology described in the ROSMAP repository papers.  
   - Involves quality control, doublet detection, normalization, etc.

5. **`map_metadata.py`**  
   - Joins extra metadata (e.g., cell-type labels, sample info) into the processed `.h5ad` files.  
   - Limited to the metadata included in the ROSMAP study.

6. **`undersampling.py`**  
   - Takes a stratified sample of cells from the training dataset (to handle large dataset size).

7. **`sample_data.py`**  
   - Creates a final sample from the training dataset based on undersampling results.

8. **`tokenise_data.py`**  
   - Transforms the final sampled `.h5ad` into Geneformer’s **rank-based encoding** format (tokenizes each cell by top expressed genes).

9. **`hyperparameter_tuning.py`**  
   - Fine-tunes Geneformer on the tokenized data using custom modifications in `custom_geneformer.py`.  
   - Explores multiple hyperparameters (learning rate, warmup steps, etc.) for optimal classification performance.

# Key Points

- **Major Cell Types**: Geneformer performs well in major brain cell types.  
- **AD vs. Non-AD Patients**: The model struggles to generalize for AD classification, possibly due to the complexity of AD biology or the strict 2,048-gene limit in the current pipeline.  
- **In Silico Perturbations**: Geneformer’s capability to “perturb” or “knock out” genes computationally can highlight genes relevant to certain cell-type signatures. Some discovered genes are not widely recognized in AD literature, suggesting either novel leads or noise.
- **Future Work**: Newer versions of Geneformer, with double the amount of attention layers and genes selected, are very promising, showcasing better metrics in AD classification. 

# Attribution

This repository **does not contain** the complete dataset or all analysis details from the references below. We **credit** both the ROSMAP group and the Geneformer authors for their foundational work:

<a name="geneformer-paper"></a>
> **Geneformer Paper**  
> Theodoris, C.V., Xiao, L., Chopra, A., Chaffin, M.D., Al Sayed, Z.R., Hill, M.C., Mantineo, H., Brydon, E.M., Zeng, Z., Liu, X.S., Ellinor, P.T.  
> “Transfer learning enables predictions in network biology.” *Nature*, 2023.  
> [Link](https://www.nature.com/articles/s41586-023-06139-9)

<a name="rosmap-paper"></a>
> **ROSMAP Paper**  
> Mathys, H., Abdelhady, G., Jiang, X., Ng, A.P., Ghanbari, K., Kunisky, A.K., Mantero, J., Galani, K., Lohia, V.N., Fortier, G.E., et al.  
> “Single-cell atlas reveals correlates of high cognitive function, dementia, and resilience to Alzheimer’s disease pathology.” *Cell*, 2023.  
> [Link](https://doi.org/10.1016/j.cell.2023.08.039)

Please refer to these publications for full details on the underlying data and the original Geneformer framework.

---

*Note: This repository only provides **partial code** used for archiving and guided replication purposes. Additional data files (e.g., `.h5ad` datasets) and original scripts referenced by the original Geneformer and ROSMAP publications are not included here.*  

