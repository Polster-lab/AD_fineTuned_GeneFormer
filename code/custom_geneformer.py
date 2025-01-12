## Functions taken from https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/classifier.py (older version than current)
## Edited for custom implementation of logging, hyperparameter optimisation, early stopping and plotting

import datetime
import logging
import os
import pickle
import subprocess
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm, trange
from sklearn.model_selection import StratifiedKFold
from sklearn import logger, preprocessing
from sklearn.metrics import (
    ConfusionMatrixDisplay
)
import seaborn as sns

from transformers import Trainer, EarlyStoppingCallback, TrainerCallback
from transformers.training_args import TrainingArguments

from geneformer import Classifier, DataCollatorForCellClassification, DataCollatorForGeneClassification
from geneformer import perturber_utils as pu
from geneformer import classifier_utils as cu
from geneformer import evaluation_utils as eu
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE

sns.set()
logger = logging.getLogger(__name__)


class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_file):
        print("SaveMetricsCallback")
        self.output_file = output_file
        # Initialize the header of the output file
        with open(self.output_file, "w") as f:
            f.write("Epoch,Training Loss,Validation Loss,Accuracy,Macro F1\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        print("on_log")
        if state.is_world_process_zero:
            with open(self.output_file, "a") as f:
                epoch = state.epoch
                training_loss = logs.get('loss', float('nan'))
                eval_loss = logs.get('eval_loss', float('nan'))
                accuracy = logs.get('eval_accuracy', float('nan'))
                macro_f1 = logs.get('eval_macro_f1', float('nan'))
                f.write(f"{epoch},{training_loss},{eval_loss},{accuracy},{macro_f1}\n")


# plot confusion matrix
def custom_plot_confusion_matrix(
    conf_mat_df, title, output_dir, output_prefix, custom_class_order
):
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.grid(False)
    #fig.set_size_inches(13, 13)
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {"axes.grid": False})
    if custom_class_order is not None:
        conf_mat_df = conf_mat_df.reindex(
            index=custom_class_order, columns=custom_class_order
        )
    display_labels = eu.generate_display_labels(conf_mat_df)
    conf_mat = preprocessing.normalize(conf_mat_df.to_numpy(), norm="l1")
    display = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=display_labels
    )
    display.plot(cmap="Blues", values_format=".3f", ax=ax)
    plt.title(title)
    plt.show()

    output_file = (Path(output_dir) / f"{output_prefix}_conf_mat").with_suffix(".pdf")
    display.figure_.savefig(output_file, bbox_inches="tight")


class CustomClassifier(Classifier):
    def __init__(self, *args, **kwargs):
        super(CustomClassifier, self).__init__(*args, **kwargs)


    def train_classifier(
        self,
        model_directory,
        num_classes,
        train_data,
        eval_data,
        output_directory,
        predict=False,
        early_stopping_patience=2,  # Add a parameter for early stopping patience
        early_stopping_threshold=0.05,  # Add a parameter for early stopping threshold
        metric_for_best_model="eval_loss"  # Add a parameter for the metric to monitor
    ):
        """
        Fine-tune model for cell state or gene classification.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        num_classes : int
            | Number of classes for classifier
        train_data : Dataset
            | Loaded training .dataset input
            | For cell classifier, labels in column "label".
            | For gene classifier, labels in column "labels".
        eval_data : None, Dataset
            | (Optional) Loaded evaluation .dataset input
            | For cell classifier, labels in column "label".
            | For gene classifier, labels in column "labels".
        output_directory : Path
            | Path to directory where fine-tuned model will be saved
        predict : bool
            | Whether or not to save eval predictions from trainer
        """

        ##### Validate and prepare data #####
        train_data, eval_data = cu.validate_and_clean_cols(
            train_data, eval_data, self.classifier
        )

        if (self.no_eval is True) and (eval_data is not None):
            logger.warning(
                "no_eval set to True; model will be trained without evaluation."
            )
            eval_data = None

        if (self.classifier == "gene") and (predict is True):
            logger.warning(
                "Predictions during training not currently available for gene classifiers; setting predict to False."
            )
            predict = False

        # ensure not overwriting previously saved model
        saved_model_test = os.path.join(output_directory, "pytorch_model.bin")
        if os.path.isfile(saved_model_test) is True:
            logger.error("Model already saved to this designated output directory.")
            raise
        # make output directory
        subprocess.call(f"mkdir {output_directory}", shell=True)

        ##### Load model and training args #####
        model = pu.load_model(self.model_type, num_classes, model_directory, "train")

        def_training_args, def_freeze_layers = cu.get_default_train_args(
            model, self.classifier, train_data, output_directory
        )

        if self.training_args is not None:
            def_training_args.update(self.training_args)
        logging_steps = round(
            len(train_data) / def_training_args["per_device_train_batch_size"] / 10
        )
        def_training_args["logging_steps"] = logging_steps
        def_training_args["output_dir"] = output_directory
        if eval_data is None:
            def_training_args["evaluation_strategy"] = "no"
            def_training_args["load_best_model_at_end"] = False
        else:
            def_training_args["evaluation_strategy"] = "epoch"
            def_training_args["load_best_model_at_end"] = True
            def_training_args["metric_for_best_model"] = metric_for_best_model  # Set the metric for early stopping

        training_args_init = TrainingArguments(
            **def_training_args
        )

        if self.freeze_layers is not None:
            def_freeze_layers = self.freeze_layers

        if def_freeze_layers > 0:
            modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        ##### Fine-tune the model #####
        # define the data collator
        if self.classifier == "cell":
            data_collator = DataCollatorForCellClassification()
        elif self.classifier == "gene":
            data_collator = DataCollatorForGeneClassification()

        # Integrate the SaveMetricsCallback into your trainer
        output_file = os.path.join(output_directory, "training_progress.csv")
        save_metrics_callback = SaveMetricsCallback(output_file)

        # create the trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=cu.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold), save_metrics_callback]
        )

        # train the classifier
        trainer.train()
        trainer.save_model(output_directory)
        if predict is True:
            # make eval predictions and save predictions and metrics
            predictions = trainer.predict(eval_data)
            prediction_output_path = f"{output_directory}/predictions.pkl"
            with open(prediction_output_path, "wb") as f:
                pickle.dump(predictions, f)
            trainer.save_metrics("eval", predictions.metrics)
        return trainer


    def hyperopt_classifier(
        self,
        model_directory,
        num_classes,
        train_data,
        eval_data,
        output_directory,
        n_trials=100,
    ):
        """
        Fine-tune model for cell state or gene classification.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        num_classes : int
            | Number of classes for classifier
        train_data : Dataset
            | Loaded training .dataset input
            | For cell classifier, labels in column "label".
            | For gene classifier, labels in column "labels".
        eval_data : None, Dataset
            | (Optional) Loaded evaluation .dataset input
            | For cell classifier, labels in column "label".
            | For gene classifier, labels in column "labels".
        output_directory : Path
            | Path to directory where fine-tuned model will be saved
        n_trials : int
            | Number of trials to run for hyperparameter optimization
        """

        # initiate runtime environment for raytune
        import ray
        from ray import tune
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune.stopper import ExperimentPlateauStopper, TrialPlateauStopper


        ray.shutdown()  # engage new ray session
        ray.init()

        ##### Validate and prepare data #####
        train_data, eval_data = cu.validate_and_clean_cols(
            train_data, eval_data, self.classifier
        )

        if (self.no_eval is True) and (eval_data is not None):
            logger.warning(
                "no_eval set to True; hyperparameter optimization requires eval, proceeding with eval"
            )

        # ensure not overwriting previously saved model
        saved_model_test = os.path.join(output_directory, "pytorch_model.bin")
        if os.path.isfile(saved_model_test) is True:
            logger.error("Model already saved to this designated output directory.")
            raise
        # make output directory
        subprocess.call(f"mkdir {output_directory}", shell=True)

        ##### Load model and training args #####
        model = pu.load_model(self.model_type, num_classes, model_directory, "train")
        def_training_args, def_freeze_layers = cu.get_default_train_args(
            model, self.classifier, train_data, output_directory
        )
        del model

        if self.training_args is not None:
            def_training_args.update(self.training_args)
        logging_steps = round(
            len(train_data) / def_training_args["per_device_train_batch_size"] / 10
        )
        def_training_args["logging_steps"] = logging_steps
        def_training_args["output_dir"] = output_directory
        if eval_data is None:
            def_training_args["evaluation_strategy"] = "no"
            def_training_args["load_best_model_at_end"] = False
        def_training_args.update(
            {"save_strategy": "epoch", "save_total_limit": 1}
        )  # only save last model for each run
        training_args_init = TrainingArguments(**def_training_args)

        ##### Fine-tune the model #####
        # define the data collator
        if self.classifier == "cell":
            data_collator = DataCollatorForCellClassification()
        elif self.classifier == "gene":
            data_collator = DataCollatorForGeneClassification()

        # define function to initiate model
        def model_init():
            model = pu.load_model(
                self.model_type, num_classes, model_directory, "train"
            )

            if self.freeze_layers is not None:
                def_freeze_layers = self.freeze_layers

            if def_freeze_layers > 0:
                modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            model = model.to("cuda:0")
            return model

        # create the trainer
        trainer = Trainer(
            model_init=model_init,
            args=training_args_init,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=cu.compute_metrics,
        )

        # specify raytune hyperparameter search space
        if self.ray_config is None:
            logger.warning(
                "No ray_config provided. Proceeding with default, but ranges may need adjustment depending on model."
            )
            def_ray_config = {
                "num_train_epochs": 50,
                "learning_rate": tune.loguniform(1e-6, 1e-3),
                "weight_decay": tune.uniform(0.0, 0.3),
                "lr_scheduler_type": tune.choice(["linear", "cosine", "polynomial"]),
                "warmup_steps": tune.uniform(100, 2000),
                "seed": tune.uniform(0, 100),
                "per_device_train_batch_size": tune.choice(
                    [def_training_args["per_device_train_batch_size"]]
                ),
            }

        hyperopt_search = HyperOptSearch(metric="eval_loss", mode="min")

        # stopper = ExperimentPlateauStopper(
        #     metric="eval_loss",
        #     mode="min",
        #     patience=2,
        # )

        stopper = TrialPlateauStopper(
            metric="eval_loss",
            mode="min",
            grace_period=10,  # minimum number of epochs before stopping
            std=0.005  # minimum improvement to consider a significant change
        )

        # optimize hyperparameters
        trainer.hyperparameter_search(
            direction="minimize",
            backend="ray",
            resources_per_trial={"cpu": int(self.nproc / self.ngpu), "gpu": 1},
            hp_space=lambda _: def_ray_config
            if self.ray_config is None
            else self.ray_config,
            search_alg=hyperopt_search,
            n_trials=n_trials,  # number of trials
            progress_reporter=tune.CLIReporter(
                max_report_frequency=600,
                sort_by_metric=True,
                max_progress_rows=n_trials,
                mode="min",
                metric="eval_loss",
                metric_columns=["loss", "eval_loss", "eval_accuracy", "eval_macro_f1"],
            ),
            local_dir=output_directory,
            stop=stopper,
        )

        return trainer


    def validate(
        self,
        model_directory,
        prepared_input_data_file,
        id_class_dict_file,
        output_directory,
        output_prefix,
        split_id_dict=None,
        attr_to_split=None,
        attr_to_balance=None,
        max_trials=100,
        pval_threshold=0.1,
        save_eval_output=True,
        predict_eval=True,
        predict_trainer=False,
        n_hyperopt_trials=0,
    ):
        """
        (Cross-)validate cell state or gene classifier.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        prepared_input_data_file : Path
            | Path to directory containing _labeled.dataset previously prepared by Classifier.prepare_data
        id_class_dict_file : Path
            | Path to _id_class_dict.pkl previously prepared by Classifier.prepare_data
            | (dictionary of format: numerical IDs: class_labels)
        output_directory : Path
            | Path to directory where model and eval data will be saved
        output_prefix : str
            | Prefix for output files
        split_id_dict : None, dict
            | Dictionary of IDs for train and eval splits
            | Three-item dictionary with keys: attr_key, train, eval
            | attr_key: key specifying name of column in .dataset that contains the IDs for the data splits
            | train: list of IDs in the attr_key column to include in the train split
            | eval: list of IDs in the attr_key column to include in the eval split
            | For example: {"attr_key": "individual",
            |               "train": ["patient1", "patient2", "patient3", "patient4"],
            |               "eval": ["patient5", "patient6"]}
            | Note: only available for CellClassifiers with 1-fold split (self.classifier="cell"; self.num_crossval_splits=1)
        attr_to_split : None, str
            | Key for attribute on which to split data while balancing potential confounders
            | e.g. "patient_id" for splitting by patient while balancing other characteristics
            | Note: only available for CellClassifiers with 1-fold split (self.classifier="cell"; self.num_crossval_splits=1)
        attr_to_balance : None, list
            | List of attribute keys on which to balance data while splitting on attr_to_split
            | e.g. ["age", "sex"] for balancing these characteristics while splitting by patient
        max_trials : None, int
            | Maximum number of trials of random splitting to try to achieve balanced other attribute
            | If no split is found without significant (p < pval_threshold) differences in other attributes, will select best
        pval_threshold : None, float
            | P-value threshold to use for attribute balancing across splits
            | E.g. if set to 0.1, will accept trial if p >= 0.1 for all attributes in attr_to_balance
        save_eval_output : bool
            | Whether to save cross-fold eval output
            | Saves as pickle file of dictionary of eval metrics
        predict_eval : bool
            | Whether or not to save eval predictions
            | Saves as a pickle file of self.evaluate predictions
        predict_trainer : bool
            | Whether or not to save eval predictions from trainer
            | Saves as a pickle file of trainer predictions
        n_hyperopt_trials : int
            | Number of trials to run for hyperparameter optimization
            | If 0, will not optimize hyperparameters
        """

        if self.num_crossval_splits == 0:
            logger.error("num_crossval_splits must be 1 or 5 to validate.")
            raise

        # ensure number of genes in each class is > 5 if validating model
        if self.classifier == "gene":
            insuff_classes = [k for k, v in self.gene_class_dict.items() if len(v) < 5]
            if (self.num_crossval_splits > 0) and (len(insuff_classes) > 0):
                logger.error(
                    f"Insufficient # of members in class(es) {insuff_classes} to (cross-)validate."
                )
                raise

        ##### Load data and prepare output directory #####
        # load numerical id to class dictionary (id:class)
        with open(id_class_dict_file, "rb") as f:
            id_class_dict = pickle.load(f)
        class_id_dict = {v: k for k, v in id_class_dict.items()}

        # load previously filtered and prepared data
        data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
        data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

        # define output directory path
        current_date = datetime.datetime.now()
        datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
        if output_directory[-1:] != "/":  # add slash for dir if not present
            output_directory = output_directory + "/"
        output_dir = f"{output_directory}{datestamp}_geneformer_{self.classifier}Classifier_{output_prefix}/"
        subprocess.call(f"mkdir {output_dir}", shell=True)

        # get number of classes for classifier
        num_classes = cu.get_num_classes(id_class_dict)

        ##### (Cross-)validate the model #####
        results = []
        all_conf_mat = np.zeros((num_classes, num_classes))
        iteration_num = 1
        if self.classifier == "cell":
            for i in trange(self.num_crossval_splits):
                print(
                    f"****** Validation split: {iteration_num}/{self.num_crossval_splits} ******\n"
                )
                ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
                if self.num_crossval_splits == 1:
                    # single 1-eval_size:eval_size split
                    if split_id_dict is not None:
                        data_dict = dict()
                        data_dict["train"] = pu.filter_by_dict(
                            data,
                            {split_id_dict["attr_key"]: split_id_dict["train"]},
                            self.nproc,
                        )
                        data_dict["test"] = pu.filter_by_dict(
                            data,
                            {split_id_dict["attr_key"]: split_id_dict["eval"]},
                            self.nproc,
                        )
                    elif attr_to_split is not None:
                        data_dict, balance_df = cu.balance_attr_splits(
                            data,
                            attr_to_split,
                            attr_to_balance,
                            self.eval_size,
                            max_trials,
                            pval_threshold,
                            self.cell_state_dict["state_key"],
                            self.nproc,
                        )

                        balance_df.to_csv(
                            f"{output_dir}/{output_prefix}_train_valid_balance_df.csv"
                        )
                    else:
                        data_dict = data.train_test_split(
                            test_size=self.eval_size,
                            stratify_by_column=self.stratify_splits_col,
                            seed=42,
                        )
                    train_data = data_dict["train"]
                    eval_data = data_dict["test"]
                else:
                    # 5-fold cross-validate
                    num_cells = len(data)
                    fifth_cells = int(num_cells * 0.2)
                    num_eval = int(min((self.eval_size * num_cells), fifth_cells))
                    start = int(i * fifth_cells)
                    end = start + num_eval
                    # ensure end is within bounds
                    if end > num_cells:
                        end = num_cells
                    eval_indices = [j for j in range(start, end)]
                    train_indices = [j for j in range(num_cells) if j not in eval_indices]
                    eval_data = data.select(eval_indices)
                    train_data = data.select(train_indices)
                if n_hyperopt_trials == 0:
                    trainer = self.train_classifier(
                        model_directory,
                        num_classes,
                        train_data,
                        eval_data,
                        ksplit_output_dir,
                        predict_trainer,
                    )
                else:
                    trainer = self.hyperopt_classifier(
                        model_directory,
                        num_classes,
                        train_data,
                        eval_data,
                        ksplit_output_dir,
                        n_trials=n_hyperopt_trials,
                    )
                    if iteration_num == self.num_crossval_splits:
                        return
                    else:
                        iteration_num = iteration_num + 1
                        continue

                result = self.evaluate_model(
                    trainer.model,
                    num_classes,
                    id_class_dict,
                    eval_data,
                    predict_eval,
                    ksplit_output_dir,
                    output_prefix,
                )
                results += [result]
                all_conf_mat = all_conf_mat + result["conf_mat"]
                iteration_num = iteration_num + 1

        elif self.classifier == "gene":
            # set up (cross-)validation splits
            targets = pu.flatten_list(self.gene_class_dict.values())
            labels = pu.flatten_list(
                [
                    [class_id_dict[label]] * len(targets)
                    for label, targets in self.gene_class_dict.items()
                ]
            )
            assert len(targets) == len(labels)
            n_splits = int(1 / self.eval_size)
            skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
            # (Cross-)validate
            for train_index, eval_index in tqdm(skf.split(targets, labels)):
                print(
                    f"****** Validation split: {iteration_num}/{self.num_crossval_splits} ******\n"
                )
                ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
                # filter data for examples containing classes for this split
                # subsample to max_ncells and relabel data in column "labels"
                train_data, eval_data = cu.prep_gene_classifier_split(
                    data,
                    targets,
                    labels,
                    train_index,
                    eval_index,
                    self.max_ncells,
                    iteration_num,
                    self.nproc,
                )

                if n_hyperopt_trials == 0:
                    trainer = self.train_classifier(
                        model_directory,
                        num_classes,
                        train_data,
                        eval_data,
                        ksplit_output_dir,
                        predict_trainer,
                    )
                else:
                    trainer = self.hyperopt_classifier(
                        model_directory,
                        num_classes,
                        train_data,
                        eval_data,
                        ksplit_output_dir,
                        n_trials=n_hyperopt_trials,
                    )
                    if iteration_num == self.num_crossval_splits:
                        return
                    else:
                        iteration_num = iteration_num + 1
                        continue
                result = self.evaluate_model(
                    trainer.model,
                    num_classes,
                    id_class_dict,
                    eval_data,
                    predict_eval,
                    ksplit_output_dir,
                    output_prefix,
                )
                results += [result]
                all_conf_mat = all_conf_mat + result["conf_mat"]
                # break after 1 or 5 splits, each with train/eval proportions dictated by eval_size
                if iteration_num == self.num_crossval_splits:
                    break
                iteration_num = iteration_num + 1

        all_conf_mat_df = pd.DataFrame(
            all_conf_mat, columns=id_class_dict.values(), index=id_class_dict.values()
        )
        all_metrics = {
            "conf_matrix": all_conf_mat_df,
            "macro_f1": [result["macro_f1"] for result in results],
            "acc": [result["acc"] for result in results],
        }
        all_roc_metrics = None  # roc metrics not reported for multiclass
        if num_classes == 2:
            mean_fpr = np.linspace(0, 1, 100)
            all_tpr = [result["roc_metrics"]["interp_tpr"] for result in results]
            all_roc_auc = [result["roc_metrics"]["auc"] for result in results]
            all_tpr_wt = [result["roc_metrics"]["tpr_wt"] for result in results]
            mean_tpr, roc_auc, roc_auc_sd = eu.get_cross_valid_roc_metrics(
                all_tpr, all_roc_auc, all_tpr_wt
            )
            all_roc_metrics = {
                "mean_tpr": mean_tpr,
                "mean_fpr": mean_fpr,
                "all_roc_auc": all_roc_auc,
                "roc_auc": roc_auc,
                "roc_auc_sd": roc_auc_sd,
            }
        all_metrics["all_roc_metrics"] = all_roc_metrics
        if save_eval_output is True:
            eval_metrics_output_path = (
                Path(output_dir) / f"{output_prefix}_eval_metrics_dict"
            ).with_suffix(".pkl")
            with open(eval_metrics_output_path, "wb") as f:
                pickle.dump(all_metrics, f)

        return all_metrics
    

    def plot_conf_mat(
        self,
        conf_mat_dict,
        output_directory,
        output_prefix,
        custom_class_order=None,
    ):
        """
        Plot confusion matrix results of evaluating the fine-tuned model.

        **Parameters**

        conf_mat_dict : dict
            | Dictionary of model_name : confusion_matrix_DataFrame
            | (all_metrics["conf_matrix"] from self.validate)
        output_directory : Path
            | Path to directory where plots will be saved
        output_prefix : str
            | Prefix for output file
        custom_class_order : None, list
            | List of classes in custom order for plots.
            | Same order will be used for all models.
        """

        for model_name in conf_mat_dict.keys():
            custom_plot_confusion_matrix(
                conf_mat_dict[model_name],
                model_name,
                output_directory,
                output_prefix,
                custom_class_order,
            )
