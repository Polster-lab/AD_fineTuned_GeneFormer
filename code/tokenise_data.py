from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer({'bc': 'bc', 'obsnames': 'obsnames', 'batch': 'batch', 'n_counts': 'n_counts', 'pct_ribo': 'pct_ribo', 'pct_mito': 'pct_mito', 
                             'Patient_Batch_ID': 'Patient_Batch_ID', 'Patient_ID': 'Patient_ID', 'AD_diagnosis': 'AD_diagnosis', 'msex': 'msex', 'age_death': 'age_death', 
                             'pmi': 'pmi', 'race': 'race', 'subject_cell_id': 'subject_cell_id', 
                             'Cell_Type': 'Cell_Type', 'Major_Cell_Type': 'Major_Cell_Type'}, nproc=64)
tk.tokenize_data(".", 
                 "/path/to/data/", 
                 "PFC427_train_clean_anno_sample", 
                 file_format="h5ad")
