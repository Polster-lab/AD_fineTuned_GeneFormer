# %%
import datetime
import os
from contextlib import redirect_stdout
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_geneformer import CustomClassifier

adata = pd.read_csv("PFC427_train_clean_anno_sample.csv")   

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "classifier_ft"
output_dir = f"path/to/output"
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
# we split the data based on the 'dataset' column, which signifies the original dataset the sample was taken from (train or test)
unique_ids = adata[adata['dataset'] == 'train']['obsnames'].unique()
train_ids, eval_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
test_ids = adata[adata['dataset'] == 'test']['obsnames'].unique()

train_test_id_split_dict = {"attr_key": "obsnames",
                            "train": train_ids.tolist()+eval_ids.tolist(),
                            "test": eval_ids.tolist()}

# Example input_data_file: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
cc.prepare_data(input_data_file="PFC427_clean_anno_sample.dataset",
                output_directory=output_dir,
                output_prefix=output_prefix,
                split_id_dict=train_test_id_split_dict)

train_valid_id_split_dict = {"attr_key": "obsnames",
                            "train": train_ids,
                            "eval": eval_ids}

# %%
output_file = os.path.join(output_dir, "output_1k.txt")

with open(output_file, "w") as f:
    with redirect_stdout(f):

        # 6 layer Geneformer: https://huggingface.co/ctheodoris/Geneformer/blob/main/model.safetensors
        all_metrics = cc.validate(model_directory="path/to/Geneformer",
                                prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
                                id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
                                output_directory=output_dir,
                                output_prefix=output_prefix,
                                split_id_dict=train_valid_id_split_dict,
                                n_hyperopt_trials=15)
                                # to optimize hyperparameters, set n_hyperopt_trials=100 (or alternative desired # of trials)

# %%
# Evaluate the pretrained model on the prepared dataset
all_metrics = cc.evaluate_saved_model(
        model_directory=f"path/to/checkpoint",
        id_class_dict_file=f"path/to/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"path/to/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

# Plot confusion matrix
cc.plot_conf_mat(
        conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
        output_directory=output_dir,
        output_prefix=output_prefix,
)
plt.show()
