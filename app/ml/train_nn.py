import logging
import os
import pathlib
from math import sqrt
from typing import List, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
import yaml
from keras.layers import concatenate
from matplotlib import pyplot as plt
from pandas.plotting import table
from sklearn.metrics import (
    f1_score,
    precision_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from tensorflow import keras
from wandb.integration.keras import WandbModelCheckpoint
from sklearn.utils import resample
from scipy.stats import norm

from wandb.keras import WandbCallback

from app.data_handling.get_training_data import AutopilotDataset

from app.ml.custom_SAM import SharpnessAwareMinimizationPatched as sam
from app.ml.train import calculate_mean
from app.ml.train_helper import (
    return_train_validation_cv_split,
    get_dataset,
    inverse_normalize_data,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def metric_state():
    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc", curve="ROC"),
        keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
    ]

    return METRICS


# from generate_deep_data import AutopilotDataset
with (pathlib.Path(__file__).parent.parent / "config" / "constants.yaml").open(
    "r"
) as stream:
    code_dict = yaml.safe_load(stream)


def make_model(config: dict):
    if config["norm_type"] == "LayerNorm":
        norm_cls = tf.keras.layers.LayerNormalization
    else:
        norm_cls = tf.keras.layers.BatchNormalization

    # defining two sets of inputs
    df_input = tf.keras.Input(shape=(60, 7))
    ti_input = tf.keras.Input(shape=16)
    station_input = tf.keras.Input(shape=2)

    # put sliding windows trough lstm
    masked_input = tf.keras.layers.Masking(mask_value=1024)(df_input)
    net = tf.keras.layers.LSTM(config["in_lstm"], return_sequences=True)(masked_input)
    net = tf.keras.layers.LSTM(config["out_lstm"])(net)
    net = norm_cls()(net)
    net = tf.keras.layers.Dropout(0.1)(net)

    station_net = tf.keras.layers.Embedding(
        config["input_dim_station_embedding"] + 1, 2, input_length=1
    )(station_input[:, 1])
    station_net = tf.keras.layers.Reshape([2])(station_net)

    clinic_net = tf.keras.layers.Embedding(21 + 1, 2, input_length=1)(
        station_input[:, 0]
    )
    clinic_net = tf.keras.layers.Reshape([2])(clinic_net)

    if config["clinic_filter"] != "None":
        net = concatenate([net, ti_input, station_net])
    else:
        net = concatenate([net, ti_input, clinic_net, station_net])

    net = tf.keras.layers.Dense(config["dense"])(net)
    net = norm_cls()(net)
    net = tf.keras.layers.ReLU()(net)

    dense1 = tf.keras.layers.Dense(1, activation="sigmoid", name="label")(net)

    model = tf.keras.Model(inputs=[df_input, ti_input, station_input], outputs=dense1)

    if config["use_sam"]:
        model = sam(model=model)

    assert config["loss_function"] in {"Adam", "AdamW"}
    config["weight_decay_score"] = (
        0 if config["loss_function"] == "Adam" else config["learning_rate"] * 0.1
    )
    model.compile(
        # Keep factor 10 smaller than learning rate
        optimizer=tf.keras.optimizers.experimental.AdamW(
            weight_decay=config["weight_decay_score"],
            learning_rate=config["learning_rate"],
        ),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=config["apply_weight_balancing"]
        ),
        metrics=metric_state(),
    )
    return model


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, config):
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 25:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = current_lr * 0.5
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.config["weight_decay_score"] = new_lr
            logging.info(f"lr update now: {new_lr}")
        if epoch == 45:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = current_lr * 0.1
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.config["weight_decay_score"] = new_lr
            logging.info(f"lr update now: {new_lr}")
        if epoch == 65:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = current_lr * 0.1
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.config["weight_decay_score"] = new_lr
            logging.info(f"lr update now: {new_lr}")


def manage_callbacks() -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    # Create checkpoint callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_prc",
        verbose=1,
        patience=10,
        mode="max",
        restore_best_weights=True,
    )

    wandbcallback1 = WandbCallback(save_model=False, mode="max", monitor="val_prc")

    wannbCB2 = WandbModelCheckpoint(
        monitor="val_prc",
        mode="max",
        filepath=f"{wandb.run.dir}/model-best.tf",
        save_best_only=True,
        # save_freq=100
    )
    # wannbCB3 = WandbModelCheckpoint(filepath=f"{wandb.run.dir}/model-latest.tf")

    # reducelronplateu = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_prc", factor=0.5, patience=10, mode="max", cooldown=2
    # )

    # modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    #     monitor="val_prc", mode="max", filepath="../.tmp", save_best_only=True
    # )

    callbacks += [wandbcallback1, wannbCB2, early_stopping]
    return callbacks


@tf.function
def check_valid(x):
    print()
    if x["clinic"] == "heart_thorax":
        tf.print(x["clinic"])
        print("fail")
    return x


def view_samples(validation_dataset: tf.data.Dataset) -> None:
    for (sq_batch, mt_batch, station), label_batch in validation_dataset.take(5):
        for sq, mt, st, label in zip(sq_batch, mt_batch, station, label_batch):
            # assert st[-1].numpy() <= 17, f"{st}"
            if label == 1:
                # print(f"Time dependend: {sq}")
                # print(f"Sequential: {mt}")
                print(f"Station: {st}")
                # print(f"Label: {label}")


def generate_eval_samples(validation_dataset: tf.data.Dataset, model) -> None:
    def log_sample(
        df_mt: pd.DataFrame,
        df_sq: pd.DataFrame,
        df_label: pd.DataFrame,
        prediction: pd.DataFrame,
        category_name: str,
    ):
        fig, axes = plt.subplots(
            1,
            4,
            sharex=False,
            figsize=(20, 5),
            gridspec_kw={"width_ratios": [10, 1, 1, 1]},
        )
        plt.subplots_adjust(wspace=0.05, hspace=0)

        fig.patch.set_visible(False)
        axes[0].axis("off")
        axes[1].axis("off")
        axes[2].axis("off")
        axes[3].axis("off")

        tab_mt = table(
            axes[0],
            df_mt,
            loc="center",
            cellLoc="center",
            colWidths=[0.1 for x in df_mt.columns],
        )

        tab_mt.auto_set_column_width(col=list(range(len(df_mt.columns))))
        table(axes[1], df_sq, loc="center", cellLoc="center")
        table(axes[2], df_label, loc="center", cellLoc="center")
        table(axes[3], prediction, loc="center", cellLoc="center")

        axes[0].set_title("Time Dependent Input")
        axes[1].set_title("Sequential Input")
        axes[2].set_title("True Label")
        axes[3].set_title("Prediction")

        wandb.log({category_name: [wandb.Image(fig, caption="df")]})
        plt.close()

    for (mt_batch, sq_batch), label_batch in validation_dataset.take(3):
        true_predictions_counts = 0
        true_label_counts = 0

        label_day1 = label_batch["dense_day1"].numpy()
        label_day2 = label_batch["dense_day2"].numpy()
        label_day3 = label_batch["dense_day3"].numpy()

        label_list = []
        for i, value in enumerate(label_day1):
            label_list.append([value, label_day2[i], label_day3[i]])

        for mt, sq, label in zip(mt_batch, sq_batch, label_list):
            prediction = model.predict_on_batch(
                (mt[tf.newaxis, ...], (sq[tf.newaxis, ...]))
            )
            prediction = [np.round(v[0][0], 2) for v in prediction.values()]
            prediction = pd.DataFrame(prediction, columns=["Prediciton"])

            df_mt = pd.DataFrame(mt).reset_index(drop=True)
            df_mt.columns = code_dict["MT_FEATURES"].values()
            scaler_dict = pd.read_pickle("ml/scalers.pkl")
            df_mt.zTHRA = (
                inverse_normalize_data(df_mt.zTHRA, scaler_dict["zthra_scaler"])
                .numpy()
                .astype(int)
            )
            # df_mt.TK = (
            #     inverse_normalize_data(df_mt.TK, scaler_dict["tkz_scaler"])
            #     .numpy()
            #     .astype(int)
            # )

            df_sq = pd.DataFrame(sq).reset_index(drop=True)
            df_sq.columns = ["SQ_Features"]
            df_sq.index = code_dict["SQ_Features"].values()
            df_sq.loc["age"] = (
                inverse_normalize_data(df_sq.loc["age"], scaler_dict["age_scaler"])
                .numpy()
                .astype(int)
            )

            # logging.info(df_sq.loc["age"])
            df_label = pd.DataFrame(label, columns=["Label"]).reset_index(drop=True)

            if (prediction > 0.5).any().any() and true_predictions_counts <= 25:
                log_sample(df_mt, df_sq, df_label, prediction, "True Predictions")
                true_predictions_counts += 1
                continue

            if (df_label).values.sum() > 0 and true_label_counts <= 25:
                log_sample(df_mt, df_sq, df_label, prediction, "True Labels")
                true_label_counts += 1

            if true_predictions_counts > 50 and true_label_counts > 50:
                break
    return


def train(config):
    ### -> check sweep_rf.yaml
    print(f"batch_size: {config['batch_size']}")
    print(f"dense: {config['dense']}")
    print(f"in_lstm: {config['in_lstm']}")
    print(f"learning_rate: {config['learning_rate']}")
    print(f"loss_function: {config['loss_function']}")
    print(f"norm_type: {config['norm_type']}")
    print(f"out_lstm: {config['out_lstm']}")
    print(f"sam: {config['use_sam']}")
    print(f"apply_weight_balancing: {config['apply_weight_balancing']}")
    print(f"ds name: {config['dataset_name']}")
    ####

    run = wandb.init(
        project=config["wb_project_name"],
        notes="init",
        tags=["fixed_embed"],
        config=config,
        mode="online",  # {'disabled', 'online', 'dryrun', 'offline', 'run'}
        # allow_val_change=True,
        settings=wandb.Settings(code_dir="."),
        # name=f'ep:{config["epochs"]}_bt:{config["batch_size"]}_lr:{config["learning_rate"]}'
    )
    wandb.run.log_code(".")

    train_splits, val_splits = return_train_validation_cv_split(cv_idx=1, cv_folds=5)

    # No CV
    model = make_model(config)
    train_dataset = get_dataset(train_splits, config)  # 85504
    val_dataset = get_dataset(val_splits, config)  # 23000

    ###
    model.fit(
        train_dataset,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        callbacks=[manage_callbacks, CustomCallback(config)],
        # callbacks=[manage_callbacks(train_dataset)],
        validation_data=val_dataset,
        # class_weight=class_weight
    )
    # generate_eval_samples(val_dataset, model)
    ###

    model.summary()
    run.finish()


def evaluate_nn_model(model, dataset):
    metrics = metric_state()
    evaluation = model.evaluate(
        dataset, return_dict=True, callbacks=[wandb.keras.WandbCallback()], verbose=0
    )
    evaluation_metrics = {metric.name: evaluation[metric.name] for metric in metrics}
    wandb.log(evaluation_metrics)
    return evaluation_metrics


def train_cv(config):
    # Cross validation
    for cv_id in range(1, 6):
        train_splits, val_splits = return_train_validation_cv_split(
            cv_idx=cv_id, cv_folds=5
        )

        with wandb.init(
            project=config["wb_project_name"],
            notes="",
            tags=["final_model"],
            config=config,
            mode="online",  # {'disabled', 'online', 'dryrun', 'offline', 'run'}
            settings=wandb.Settings(code_dir="app/."),
            name=f"cv:{cv_id}",
        ) as run:
            model = make_model(config)

            wandb.run.log_code(".")

            train_dataset = get_dataset(train_splits, config, True, True, True)
            val_dataset = get_dataset(val_splits, config, True, True, True)

            model.fit(
                train_dataset,
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                callbacks=[manage_callbacks(), CustomCallback(config)],
                validation_data=val_dataset,
            )

            test_dataset = get_dataset("test", config, True, True, True)
            evaluation_metrics = evaluate_nn_model(model, test_dataset)
            print(evaluation_metrics)
            wandb.log({"test": evaluation_metrics})

        run.finish()


class EnsembleDL:
    def __init__(
        self, config, wandb_log=False, data_percentage=None, rebuild_model=False
    ):
        self.config = config
        ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
        ds.download_and_prepare()
        train_splits, val_splits = return_train_validation_cv_split(
            cv_idx=1, cv_folds=5
        )
        self.train_dataset = get_dataset(train_splits, config)
        self.val_dataset = get_dataset(val_splits, config)
        self.test_dataset = get_dataset("test", config)
        self.wandb_log = wandb_log

        if data_percentage:
            self.reduce_dataset(data_percentage)

        if (
            not Path(
                "app/config/model_"
                + self.config["clinic_filter"]
                + "/ensemble_model.h5"
            ).exists()
            or rebuild_model
        ):
            logging.info("Reloading Ensemble Model")
            self.model = None
            self.load_ensemble_models(5, 0)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.loss = tf.keras.losses.BinaryCrossentropy()
            self.metrics = []
            self.threshold = 0.5
        else:
            self.model = tf.keras.models.load_model(
                "app/config/model_"
                + self.config["clinic_filter"]
                + "/ensemble_model.h5"
            )

        if wandb_log:
            wandb.init(
                project=config["wb_project_name"],
                notes="ensemble",
                tags=["ensemble"],
                config=config,
                mode="online",  # {'disabled', 'online', 'dryrun', 'offline', 'run'}
                settings=wandb.Settings(code_dir="."),
            )

    def reduce_dataset(self, data_percentage):
        train_count = len(list(self.train_dataset))
        val_count = len(list(self.val_dataset))
        test_count = len(list(self.test_dataset))
        self.train_dataset = self.train_dataset.take(int(train_count * data_percentage))
        self.val_dataset = self.val_dataset.take(int(val_count * data_percentage))
        self.test_dataset = self.test_dataset.take(int(test_count * data_percentage))

    def load_ensemble_models(
        self, n_splits: int, ref_model: int = 0
    ) -> List[tf.keras.Model]:
        models = []
        for index in range(1, n_splits + 1):
            model_dir = Path(
                "app/config/model_" + self.config["clinic_filter"] + f"/cv_{index}"
            )
            model = tf.keras.models.load_model(model_dir)
            model._name = "model_" + str(index)  # Set a unique name for the model
            models.append(model)
        self.build_ensemble(loaded_models=models, ref_model=ref_model)
        return models

    def build_ensemble(
        self, loaded_models: List[tf.keras.Model], ref_model: int = 0
    ) -> None:
        logging.info("Fusing the models together into an ensemble.")
        models_output = [
            model(loaded_models[ref_model].inputs, training=False)
            for model in loaded_models
        ]
        self.optimizer = loaded_models[ref_model].optimizer
        self.metrics = loaded_models[ref_model].metrics
        self.loss = loaded_models[ref_model].loss
        logging.info(
            f"Setting the optimizer, the metrics and the loss to the ones of the model at position {ref_model} and compiling."
        )

        print(self.metrics)
        print(self.loss)
        print(self.optimizer)
        self.model = tf.keras.Model(
            inputs=loaded_models[ref_model].inputs,
            outputs=tf.keras.layers.average(models_output),
        )

        self.model.compile(
            optimizer=self.optimizer,
            metrics=metric_state(),
            loss=self.loss,
            run_eagerly=True,
        )
        logging.info(self.model.summary())
        self.model.save(
            "app/config/model_" + self.config["clinic_filter"] + "/ensemble_model.h5"
        )

    def evaluate_model(
        self,
        dataset: tf.data,
        y_pred,
        y_true,
        data_type: str,
        threshold=0.5,
        iteration=None,
    ) -> tuple[Union[float, Any], Union[float, Any], Union[float, Any], Any, Any, Any]:
        print(f"threshold {threshold}")

        print(f"y_pred {y_pred}")
        print(f"y_true {y_true}")
        if dataset is None and (y_pred is None or y_true is None):
            logging.error(
                "You must provide a dataset and the true labels to evaluate the model."
            )
            exit(0)

        if y_pred is not None and y_true is not None:
            logging.info("Evaluating the model on the provided labels.")
            y_pred_class = (y_pred >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
            print(f"tn {tn}, fp {fp}, fn {fn}, tp {tp}")

            auc_pr = average_precision_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred_class)
            f1 = f1_score(y_true, y_pred_class)
            precision = precision_score(y_true, y_pred_class)

        elif threshold == 0.5:
            metrics = self.model.evaluate(dataset)
            logging.info(f"Metrics: {metrics}")
            tp = metrics[1]
            fp = metrics[2]
            tn = metrics[3]
            fn = metrics[4]
            precision = metrics[5]
            recall = metrics[6]
            auc = metrics[7]
            auc_pr = metrics[8]
            f1 = recall * precision * 2 / (recall + precision)
            x = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = ((tp * tn) - (fp * fn)) / x
        else:
            y_true = []
            y_pred = []
            y_proba = []

            for batch_features, batch_labels in dataset.take(-1):
                # Predict the labels for the batch features
                y_proba_batch = self.model.predict(
                    batch_features
                )  # temporary variable for batch predictions
                y_pred_batch = [1 if prob >= threshold else 0 for prob in y_proba_batch]
                # Append true and predicted labels for the batch to lists
                y_true.extend(batch_labels.numpy())
                y_pred.extend(y_pred_batch)  # corrected line
                y_proba.append(
                    y_proba_batch
                )  # append the batch predictions to the list
            # Convert lists to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_proba = np.concatenate(y_proba)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            auc_pr = average_precision_score(y_true, y_proba)
            mcc = matthews_corrcoef(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
        speci = tn / (tn + fp)
        sensi = tp / (tp + fn)

        logging.info("AUC PR: " + str(auc_pr))
        logging.info("MCC: " + str(mcc))
        logging.info("F1 score: " + str(f1))
        logging.info("Precision: " + str(precision))
        logging.info("Specificity: " + str(speci))
        logging.info("Sensitivity: " + str(sensi))

        if self.wandb_log and not iteration:
            metrics_table = wandb.Table(columns=["Metric", "Value"])
            metrics_table.add_data("AUC PR", auc_pr)
            metrics_table.add_data("MCC", mcc)
            metrics_table.add_data("F1 score", f1)
            metrics_table.add_data("Precision", precision)
            metrics_table.add_data("Specificity", speci)
            metrics_table.add_data("Sensitivity", sensi)
            wandb.log({data_type + "_metrics_" + str(threshold): metrics_table})

        return auc_pr, mcc, f1, precision, speci, sensi

    def get_optimal_threshold(self):
        y_true = []
        y_scores = []

        # Iterate over each batch of the dataset
        for batch_features, batch_labels in self.val_dataset:
            # Predict the scores for the batch features
            batch_scores = self.model.predict(batch_features)
            # Append true labels and scores for the batch to lists
            y_true.extend(batch_labels.numpy())
            y_scores.extend(batch_scores)

        # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        precision, recall, threshold = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        return threshold[np.argmax(f1_scores)]

    def evaluate_bootstrap_dl(
        self,
        opti_threshold=0.5,
        n_iterations=1000,
        confidence_level=0.95,
    ):
        score_dicts = {
            metric: [] for metric in ["auc", "mcc", "f1", "ps", "speci", "sensi", "rs"]
        }

        # Predict the entire dataset
        y_pred = self.model.predict(self.test_dataset)
        labels = np.concatenate([y for x, y in self.test_dataset], axis=0)

        for i in range(n_iterations):
            resampled_indices = resample(np.arange(y_pred.shape[0]))
            y_pred_resampled = y_pred[resampled_indices]
            labels_resampled = labels[resampled_indices]

            # Now you can pass this dataset_resampled to your evaluate_model method
            auc_pr, mcc, f1, precision, speci, sensi = self.evaluate_model(
                None,
                y_pred_resampled,
                labels_resampled,
                data_type="test",
                threshold=opti_threshold,
                iteration=i,
            )

            score_dicts["auc"].append(auc_pr)
            score_dicts["mcc"].append(mcc)
            score_dicts["f1"].append(f1)
            score_dicts["ps"].append(precision)
            score_dicts["speci"].append(speci)
            score_dicts["sensi"].append(sensi)
            score_dicts["rs"].append(0)

        print(score_dicts)

        means, stds = calculate_mean(score_dicts)

        score_dicts.pop("rs")
        ci_lower = []
        ci_upper = []
        for mean, std in zip(means, stds):
            ci_low, ci_upp = norm.interval(
                confidence_level, loc=mean, scale=std / sqrt(n_iterations)
            )
            ci_lower.append(ci_low)
            ci_upper.append(ci_upp)

        if self.wandb_log:
            column_names = ["Metric", "Mean", "Confidence Interval"]
            table = wandb.Table(columns=column_names)
            wandb.log(
                {
                    "iteration": n_iterations,
                    "confidence_level": confidence_level,
                    "threshold": opti_threshold,
                }
            )

            metrics = [
                "AUC-PR",
                "MCC",
                "F1-Score",
                "Precision",
                "Specificity",
                "Sensitivity",
            ]
            for metric, mean, ci_low, ci_upp in zip(metrics, means, ci_lower, ci_upper):
                table.add_data(
                    metric, mean, f"[{np.round(ci_low, 2)}, {np.round(ci_upp, 2)}]"
                )

            table_title = (
                f"Bootstrap Results for test on {self.config['clinic_filter']} using {opti_threshold}"
                if opti_threshold
                else f"Bootstrap Results for test on {self.config['clinic_filter']}"
            )
            wandb.log({table_title: table})

        return means, stds, ci_lower, ci_upper

    def evaluate_bootstrap_dl_none(
        self,
        opti_threshold=0.5,
        n_iterations=1000,
        confidence_level=0.95,
    ):
        score_lists = {
            metric: [] for metric in ["auc", "mcc", "f1", "ps", "speci", "sensi", "rs"]
        }

        # Convert to NumPy array or pandas DataFrame
        sq_features = []
        mt_features = []
        station_features = []
        labels = []

        for (sq_batch, mt_batch, station), label_batch in self.test_dataset.take(-1):
            sq_features.append(sq_batch.numpy())
            mt_features.append(mt_batch.numpy())
            station_features.append(station.numpy())
            labels.append(label_batch.numpy())

        sq_features = np.concatenate(sq_features, axis=0)
        mt_features = np.concatenate(mt_features, axis=0)
        station_features = np.concatenate(station_features, axis=0)
        labels = np.concatenate(labels, axis=0)

        sq_features_flat = sq_features.reshape(sq_features.shape[0], -1)
        mt_features_flat = mt_features.reshape(mt_features.shape[0], -1)
        station_features_flat = station_features.reshape(station_features.shape[0], -1)

        for i in range(n_iterations):
            indices = np.arange(sq_features_flat.shape[0])
            resampled_indices = resample(indices)

            sq_features_resampled = sq_features_flat[resampled_indices]
            mt_features_resampled = mt_features_flat[resampled_indices]
            station_features_resampled = station_features_flat[resampled_indices]
            labels_resampled = labels[resampled_indices]

            sq_features_resampled = sq_features_resampled.reshape(-1, 60, 7)
            mt_features_resampled = mt_features_resampled.reshape(
                -1, mt_features.shape[1]
            )
            station_features_resampled = station_features_resampled.reshape(
                -1, station_features.shape[1]
            )

            # Convert your resampled data to a tf.data.Dataset
            dataset_resampled = tf.data.Dataset.from_tensor_slices(
                (
                    (
                        sq_features_resampled,
                        mt_features_resampled,
                        station_features_resampled,
                    ),
                    labels_resampled,
                )
            )

            dataset_resampled = dataset_resampled.batch(2048)

            # Now you can pass this dataset_resampled to your evaluate_model method
            auc_pr, mcc, f1, precision, speci, sensi = self.evaluate_model(
                dataset_resampled,
                None,
                None,
                data_type="test",
                threshold=opti_threshold,
                iteration=i,
            )

            score_lists["auc"].append(auc_pr)
            score_lists["mcc"].append(mcc)
            score_lists["f1"].append(f1)
            score_lists["ps"].append(precision)
            score_lists["speci"].append(speci)
            score_lists["sensi"].append(sensi)
            score_lists["rs"].append(0)

        means, stds = calculate_mean(score_lists)

        score_lists.pop("rs")
        ci_lower = []
        ci_upper = []
        for mean, std in zip(means, stds):
            ci_low, ci_upp = norm.interval(
                confidence_level, loc=mean, scale=std / sqrt(n_iterations)
            )
            ci_lower.append(ci_low)
            ci_upper.append(ci_upp)

        if self.wandb_log:
            column_names = ["Metric", "Mean", "Confidence Interval"]
            table = wandb.Table(columns=column_names)
            wandb.log(
                {
                    "iteration": n_iterations,
                    "confidence_level": confidence_level,
                    "threshold": opti_threshold,
                }
            )

            metrics = [
                "AUC-PR",
                "MCC",
                "F1-Score",
                "Precision",
                "Specificity",
                "Sensitivity",
            ]
            for metric, mean, ci_low, ci_upp in zip(metrics, means, ci_lower, ci_upper):
                table.add_data(
                    metric, mean, f"[{np.round(ci_low, 2)}, {np.round(ci_upp, 2)}]"
                )

            table_title = (
                f"Bootstrap Results for test on {self.config['clinic_filter']} using {opti_threshold}"
                if opti_threshold
                else f"Bootstrap Results for test on {self.config['clinic_filter']}"
            )
            wandb.log({table_title: table})

        return means, stds, ci_lower, ci_upper


def main(config) -> None:
    if config["eval_ensemble_bs"]:
        # evaluation of ensemble model on validation set and test set using bootstrap
        # 1. scores on validation set
        # 2. scores on validation set using optimal threshold
        # 3. scores on test set using bootstrap
        # 4. scores on test set using bootstrap and optimal threshold
        ############################
        model = EnsembleDL(config, wandb_log=True, rebuild_model=False)
        # # 1. scores on validation set
        logging.info("Evaluating ensemble model on validation set")
        model.evaluate_model(model.val_dataset, None, None, data_type="val")

        # 2. scores on validation set using optimal threshold
        calc_threshold = {"none": 0.4647, "hematooncology": 0.3845}
        if not calc_threshold.get(config["clinic_filter"]):
            optimal_threshold = model.get_optimal_threshold()
            logging.info(f"Optimal threshold: {optimal_threshold}")
            calc_threshold[config["clinic_filter"]] = np.round(optimal_threshold, 4)

        logging.info(
            f"Evaluating ensemble model on validation set using optimal threshold of {calc_threshold[config['clinic_filter']]}"
        )
        model.evaluate_model(
            model.val_dataset,
            None,
            None,
            data_type="val",
            threshold=calc_threshold[config["clinic_filter"]],
        )

        # 3. scores on test set using bootstrap
        logging.info("Evaluating ensemble model on test set using bootstrap")
        model.evaluate_bootstrap_dl()

        # 4. scores on test set using bootstrap and optimal threshold
        logging.info(
            "Evaluating ensemble model on test set using bootstrap and optimal threshold"
        )

        if config["clinic_filter"] == "none":
            model.evaluate_bootstrap_dl_none(
                opti_threshold=calc_threshold[config["clinic_filter"]]
            )
        else:
            model.evaluate_bootstrap_dl(
                opti_threshold=calc_threshold[config["clinic_filter"]]
            )
        exit(0)

    if config["run_cv"]:
        train_cv(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
