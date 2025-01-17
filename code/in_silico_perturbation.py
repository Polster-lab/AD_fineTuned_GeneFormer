# %%
from geneformer import InSilicoPerturber
from geneformer import InSilicoPerturberStats
from geneformer import EmbExtractor

import os
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

# %%
# each perturbation should go to each own folder, to avoid any overwrites
# perturbations with alt states are named with '_mult' in this project
output_prefix = "opc_to_exc"
output_dir = f"path/to/{output_prefix}/"
os.makedirs(output_dir, exist_ok=True)

# %%
# first obtain start, goal, and alt embedding positions
# this function was changed to be separate from perturb_data
# to avoid repeating calcuations when parallelizing perturb_data
cell_states_to_model={"state_key": "Major Cell Type", 
                      "start_state": 'Opc', 
                      "goal_state": 'Exc', 
                      "alt_states": []}

filter_data_dict = {"Major Cell Type": ['Exc','Opc']}

embex = EmbExtractor(model_type="CellClassifier",
                     num_classes=7,
                     emb_layer=0,
                     max_ncells=15000,
                     filter_data=filter_data_dict,
                     summary_stat="exact_mean",
                     forward_batch_size=200,
                     nproc=10)

state_embs_dict = embex.get_state_embs(cell_states_to_model,
                                       "/12tb_dsk1/dimitri/Geneformer_apptainer/ROSMAP/finetune_train/finetune_800k/240812_geneformer_cellClassifier_classifier_train/ksplit1",
                                       "/12tb_dsk1/dimitri/Geneformer_apptainer/ROSMAP/finetune_train/datasets/classifier_train_labeled_test.dataset",
                                       output_dir,
                                       output_prefix)

# %%
isp = InSilicoPerturber(perturb_type="delete",
                        perturb_rank_shift=None,
                        genes_to_perturb="all",
                        combos=0,
                        anchor_gene=None,
                        model_type="CellClassifier",
                        num_classes=7,
                        emb_mode="cell",
                        cell_emb_style="mean_pool",
                        cell_states_to_model=cell_states_to_model,
                        state_embs_dict=state_embs_dict,
                        filter_data=filter_data_dict,
                        max_ncells=1500,
                        emb_layer=0,
                        forward_batch_size=200,
                        nproc=12)

# outputs intermediate files from in silico perturbation
isp.perturb_data("path/to/Geneformer",
                 "path/to/tokenised_dataset",
                 output_dir,
                 output_prefix)

# %%
ispstats = InSilicoPerturberStats(mode="goal_state_shift",
                                  genes_perturbed="all",
                                  combos=0,
                                  anchor_gene=None,
                                  cell_states_to_model=cell_states_to_model)


# extracts data from intermediate files and processes stats to output in final .csv
ispstats.get_stats(output_dir,
                   None,
                   output_dir,
                   output_prefix)

# %%
df = pd.read_csv(output_dir + output_prefix + ".csv", index_col=0)
df

# %% [markdown]
# ### Matrixplot by top significant genes

# %%
adata = sc.read_h5ad("/12tb_dsk1/dimitri/Geneformer_apptainer/ROSMAP/PFC427_clean_anno_sample_tmp.h5ad")
adata.obs

# %%
adata.obs['original_dataset'].value_counts()

# %%
inh_neurons = pd.read_csv('path/to/inh_to_exc_mult/inh_to_exc_mult.csv', index_col=0)
microglia = pd.read_csv('path/to/mic_to_exc_mult/mic_to_exc_mult.csv', index_col=0)
oligodendrocytes = pd.read_csv('path/to/oli_to_exc_mult/oli_to_exc_mult.csv', index_col=0)
vascular = pd.read_csv('path/to/vas_to_exc_mult/vas_to_exc_mult.csv', index_col=0)
astrocytes = pd.read_csv('path/to/ast_to_exc_mult/ast_to_exc_mult.csv', index_col=0)
opc = pd.read_csv('path/to/opc_to_exc_mult/opc_to_exc_mult.csv', index_col=0)

# Add a column specifying the original dataset (major cell type, first three letters)
inh_neurons['Major_Cell_Type'] = 'Inh'
microglia['Major_Cell_Type'] = 'Mic'
oligodendrocytes['Major_Cell_Type'] = 'Oli'
vascular['Major_Cell_Type'] = 'Vas'
astrocytes['Major_Cell_Type'] = 'Ast'
opc['Major_Cell_Type'] = 'OPC'

# Merge the datasets on common columns
merged_data = pd.concat([inh_neurons, microglia, oligodendrocytes, vascular, astrocytes, opc], join='inner')

# Reorder columns to have 'Major_Cell_Type' at the front
cols = ['Major_Cell_Type'] + [col for col in merged_data.columns if col != 'Major_Cell_Type']
merged_data = merged_data[cols]

#merged_data.sort_values('N_Detections', ascending=False, inplace=True)
merged_data

# %%
merged_data[merged_data['Sig'] == 1].to_csv('path/to/most_perturb_genes.csv')

# %%
def get_top_genes_with_cell_types(df, top_n=5):
    # Step 1: Get top_n genes per cell type
    top_genes_per_cell = df.groupby('Major_Cell_Type').apply(
        lambda x: x.loc[x['Shift_to_goal_end'].abs().nlargest(top_n).index]
    ).reset_index(drop=True)
    
    # Step 2: Create a mapping from gene to the cell types where it is in top_n
    gene_celltype_map = top_genes_per_cell.groupby('Gene')['Major_Cell_Type'].apply(list).to_dict()
    
    # Step 3: Get unique top genes as before, ensuring uniqueness across cell types
    selected_genes = set()
    result_df = pd.DataFrame()
    for _, group in df.groupby('Major_Cell_Type'):
        # Exclude genes that have already been selected
        group = group[~group['Gene'].isin(selected_genes)]
        # Get the top_n genes by absolute Shift_to_goal_end
        top_genes = group.loc[group['Shift_to_goal_end'].abs().nlargest(top_n).index]
        # Add the selected genes to the set
        selected_genes.update(top_genes['Gene'])
        # Append the top genes to the result dataframe
        result_df = pd.concat([result_df, top_genes])
    
    # Step 4: Add the column with all cell types where the gene appears in top_n
    result_df['Cell_Types_with_Top_Gene'] = result_df['Gene'].map(gene_celltype_map)
    
    return result_df.reset_index(drop=True)

# Apply the function to your data
top_genes = get_top_genes_with_cell_types(merged_data[merged_data['Sig'] == 1], top_n=7)
top_genes

# %%
adata_subset = adata[adata.obs['original_dataset'] == 'test']
adata_subset = adata_subset[:, adata_subset.var_names.isin(top_genes['Gene_name'])]

# step 1: Calculate the average expression of each gene for each Major Cell Type
average_expression = adata_subset.to_df().groupby(adata_subset.obs['Major Cell Type']).mean()

# step 2: Sort genes based on which Major Cell Type has the highest expression
# get the index of the maximum value (cell type with highest expression) for each gene
sorted_genes = average_expression.idxmax().sort_values().index.tolist()

# step 3: Generate the matrix plot with sorted genes
sc.pl.matrixplot(
    adata_subset,
    var_names=sorted_genes,
    groupby='Major Cell Type', 
    title='Expression of significant genes through in silico perturbation',
    standard_scale='var',  # standardise the gene expression scale
    figsize=(20, 6),
    show=False
)

fig = plt.gcf()
for ax in fig.axes:
    ax.title.set_size(22)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=16)
plt.subplots_adjust(right=1.4, left=0.1, top=0.9, bottom=0.2)
plt.show()

# %% [markdown]
# ### Repeat, no alt states

# %%
inh_neurons = pd.read_csv('path/to/inh_to_exc/inh_to_exc.csv', index_col=0)
microglia = pd.read_csv('path/to/mic_to_exc/mic_to_exc.csv', index_col=0)
oligodendrocytes = pd.read_csv('path/to/oli_to_exc/oli_to_exc.csv', index_col=0)
vascular = pd.read_csv('path/to/vas_to_exc/vas_to_exc.csv', index_col=0)
astrocytes = pd.read_csv('path/to/ast_to_exc/ast_to_exc.csv', index_col=0)
opc = pd.read_csv('path/to/opc_to_exc/opc_to_exc.csv', index_col=0)

# Add a column specifying the original dataset (major cell type, first three letters)
inh_neurons['Major_Cell_Type'] = 'Inh'
microglia['Major_Cell_Type'] = 'Mic'
oligodendrocytes['Major_Cell_Type'] = 'Oli'
vascular['Major_Cell_Type'] = 'Vas'
astrocytes['Major_Cell_Type'] = 'Ast'
opc['Major_Cell_Type'] = 'OPC'

# Merge the datasets on common columns
merged_data = pd.concat([inh_neurons, microglia, oligodendrocytes, vascular, astrocytes, opc], join='inner')

# Reorder columns to have 'Major_Cell_Type' at the front
cols = ['Major_Cell_Type'] + [col for col in merged_data.columns if col != 'Major_Cell_Type']
merged_data = merged_data[cols]

#merged_data.sort_values('N_Detections', ascending=False, inplace=True)
merged_data

# %%
def get_top_genes_with_cell_types(df, top_n=5):
    # Step 1: Get top_n genes per cell type
    top_genes_per_cell = df.groupby('Major_Cell_Type').apply(
        lambda x: x.loc[x['Shift_to_goal_end'].abs().nlargest(top_n).index]
    ).reset_index(drop=True)
    
    # Step 2: Create a mapping from gene to the cell types where it is in top_n
    gene_celltype_map = top_genes_per_cell.groupby('Gene')['Major_Cell_Type'].apply(list).to_dict()
    
    # Step 3: Get unique top genes as before, ensuring uniqueness across cell types
    selected_genes = set()
    result_df = pd.DataFrame()
    for _, group in df.groupby('Major_Cell_Type'):
        # Exclude genes that have already been selected
        group = group[~group['Gene'].isin(selected_genes)]
        # Get the top_n genes by absolute Shift_to_goal_end
        top_genes = group.loc[group['Shift_to_goal_end'].abs().nlargest(top_n).index]
        # Add the selected genes to the set
        selected_genes.update(top_genes['Gene'])
        # Append the top genes to the result dataframe
        result_df = pd.concat([result_df, top_genes])
    
    # Step 4: Add the column with all cell types where the gene appears in top_n
    result_df['Cell_Types_with_Top_Gene'] = result_df['Gene'].map(gene_celltype_map)
    
    return result_df.reset_index(drop=True)

# Apply the function to your data
top_genes = get_top_genes_with_cell_types(merged_data[merged_data['Sig'] == 1], top_n=7)
top_genes

# %%
adata_subset = adata[adata.obs['original_dataset'] == 'test']
adata_subset = adata_subset[:, adata_subset.var_names.isin(top_genes['Gene_name'])]

# step 1: Calculate the average expression of each gene for each Major Cell Type
average_expression = adata_subset.to_df().groupby(adata_subset.obs['Major Cell Type']).mean()

# step 2: Sort genes based on which Major Cell Type has the highest expression
# get the index of the maximum value (cell type with highest expression) for each gene
sorted_genes = average_expression.idxmax().sort_values().index.tolist()

# step 3: Generate the matrix plot with sorted genes
sc.pl.matrixplot(
    adata_subset,
    var_names=sorted_genes,
    groupby='Major Cell Type', 
    title='Expression of significant genes through in silico perturbation',
    standard_scale='var',  # standardise the gene expression scale
    figsize=(20, 6),
    show=False
)

# Adjust font sizes
fig = plt.gcf()
for ax in fig.axes:
    ax.title.set_size(22)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=16)

plt.subplots_adjust(right=1.4, left=0.1, top=0.9, bottom=0.2)
plt.show()

# %%



