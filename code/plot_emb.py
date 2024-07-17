from geneformer import EmbExtractor
import pandas as pd
import matplotlib.pyplot as plt

# initiate EmbExtractor
embex = EmbExtractor(nproc=32, emb_label=['input_ids', 'AD_diagnosis', 'msex', 'age_death', 'batch', 'Major_Cell_Type'],
                     labels_to_plot=['Major_Cell_Type'],
                     token_dictionary_file='/path/to/token_dictionary.pkl',
                     max_ncells=500000)

embs = pd.read_csv('PFC427_test_emb.csv', index_col=0)

# plot UMAP of cell embeddings
# note: scanpy umap necessarily saves figs to figures directory
embex.plot_embs(embs=embs, 
                plot_style="umap",
                output_directory=".",  
                output_prefix="emb_plot")

# plot heatmap of cell embeddings
embex.plot_embs(embs=embs, 
                plot_style="heatmap",
                output_directory=".",  
                output_prefix="emb_plot")
plt.savefig('heatmap.png')
