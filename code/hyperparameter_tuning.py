# %%
import datetime
import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_geneformer import CustomClassifier

adata = pd.read_csv("PFC427_train_clean_anno_sample.csv")   

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

tmpdir = os.environ.get('TMPDIR')
output_prefix = "cm_classifier_train"
output_dir = os.path.join(tmpdir, f"finetune_train/{datestamp}")
os.makedirs(output_dir, exist_ok=True)

# %%
#filter_data_dict={"Major_Cell_Type":["Ast", "Inh", "Exc", "Mic", "Oli", "Opc", "Vas"]}
training_args = {
    "num_train_epochs": 0.5,
    "learning_rate": 0.000001,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 700,
    "weight_decay":0.01,
    "per_device_train_batch_size": 2,
    "seed": 42,
}
cc = CustomClassifier(classifier="cell",
                cell_state_dict = {"state_key": "Major_Cell_Type", "states": "all"},
                #filter_data=filter_data_dict,
                training_args=training_args,
                max_ncells=100000,
                freeze_layers = 4,
                num_crossval_splits = 5,
                forward_batch_size=100,
                nproc=64)

# %%
# Get unique IDs
unique_ids = adata['obsnames'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.1, random_state=42)
train_ids, eval_ids = train_test_split(train_ids, test_size=0.11, random_state=42)

train_test_id_split_dict = {"attr_key": "obsnames",
                            "train": train_ids.tolist()+eval_ids.tolist(),
                            "test": test_ids.tolist()}

# Example input_data_file: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
cc.prepare_data(input_data_file="PFC427_train_clean_anno_sample.dataset",
                output_directory=output_dir,
                output_prefix=output_prefix,
                split_id_dict=train_test_id_split_dict)

# %%
train_valid_id_split_dict = {"attr_key": "obsnames",
                            "train": train_ids,
                            "eval": eval_ids}

# 6 layer Geneformer: https://huggingface.co/ctheodoris/Geneformer/blob/main/model.safetensors
all_metrics = cc.validate(model_directory="/path/to/Geneformer",
                          prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
                          id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
                          output_directory=output_dir,
                          output_prefix=output_prefix,
                          split_id_dict=train_valid_id_split_dict,
                          n_hyperopt_trials=10)
                          # to optimize hyperparameters, set n_hyperopt_trials=100 (or alternative desired # of trials)

# # %% [markdown]
# # ### Evaluate saved model

# # %%
# cc = CustomClassifier(classifier="cell",
#                 cell_state_dict = {"state_key": "Pathologic_diagnosis_of_AD", "states": "all"},
#                 forward_batch_size=100,
#                 nproc=64)

# # %%
# all_metrics_test = cc.evaluate_saved_model(
#         model_directory=f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/",
#         id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
#         test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
#         output_directory=output_dir,
#         output_prefix=output_prefix,
#     )

# # %%
# cc.custom_plot_conf_mat(
#         conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
#         output_directory=output_dir,
#         output_prefix=output_prefix,
# )
# plt.savefig(f"{output_dir}/{output_prefix}_confusion_matrix.png")

# # %%



