import logging
import pickle
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
import wandb
import xgboost as xgb
from scipy.stats import norm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.utils import resample

from app.data_handling.get_training_data import AutopilotDataset
from app.ml.train_helper import calculate_scores, calculate_mean
from app.ml.train_helper import return_train_validation_cv_split, get_dataset


class Trainer:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, train, val, test, random_state, config):
        self.x_train = train[0]
        self.y_train = train[1]
        self.x_val = val[0]
        self.y_val = val[1]
        self.x_test = test[0]
        self.y_test = test[1]
        self.random_state = random_state
        self.config = config

    # Loading params and building model
    def build_model(self, model_name):
        if "xgb" in model_name:
            xgb_params = {}
            clf = xgb.XGBClassifier(
                use_label_encoder=False,
                random_state=self.random_state,
                tree_method="gpu_hist",
                n_jobs=64,
                missing=np.nan,
                # callbacks=[WandbCallback(log_model=True)],
                **xgb_params,
            )

        elif "rf" in model_name:
            params = {
                "booster": self.config["booster"],
                "reg_lambda": self.config["reg_lambda"],
                "reg_alpha": self.config["reg_alpha"],
                "max_depth": self.config["max_depth"],
                "learning_rate": self.config["learning_rate"],
                "gamma": self.config["gamma"],
                "grow_policy": self.config["grow_policy"],
                "sampling_method": self.config["sampling_method"],
            }

            clf = xgb.XGBRFClassifier(
                random_state=self.random_state,
                missing=1024,
                tree_method="gpu_hist",
                n_jobs=64,
                # callbacks=[WandbCallback(log_model=True)], # not yet supported for xgboost
                **params,
            )

            logging.info(clf.get_params())
        elif "lstm" in model_name:
            return tf.keras.models.load_model()

        else:
            logging.info("model = dummy classifier")
            clf = DummyClassifier(strategy="stratified", random_state=self.random_state)
        return clf

    # Scores
    @staticmethod
    def calculate_scores(X, y, clf, is_deep=False, threshold=None):
        if is_deep:
            y_proba = clf.predict(X)
            y_preds = [1 if x >= 0.5 else 0 for x in y_proba]
        elif threshold:
            y_proba = clf.predict(X)
            y_preds = [1 if x >= threshold else 0 for x in y_proba]
        else:
            y_preds = clf.predict(X)
            y_proba = clf.predict_proba(X)

        f1 = f1_score(y, y_preds)
        ps = precision_score(y, y_preds)
        rs = recall_score(y, y_preds)
        # precision, recall, threshold = precision_recall_curve(y, y_proba)
        if is_deep or threshold:
            auc = average_precision_score(y, pd.DataFrame(y_proba))
        else:
            auc = average_precision_score(y, pd.DataFrame(y_proba).iloc[:, 1])

        def get_confusion_matrix_values(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            return (cm[0][0], cm[0][1], cm[1][0], cm[1][1])

        tn, fp, fn, tp = get_confusion_matrix_values(y, y_preds)
        tp = np.float64(tp)
        fp = np.float64(fp)
        tn = np.float64(tn)
        fn = np.float64(fn)
        x = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / x
        speci = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        return f1, ps, rs, auc, mcc, speci

    # Training without cv
    def train_plain(self, model_name: str):
        # X_train = self.x_train.drop(["pat_id"], axis=1)
        # X_test = self.X_test.drop(["pat_id"], axis=1)

        # upsamle here
        # X_train, y_train = self.oversample_gmm(
        #     x_split=X_train, y_split=y_train, rows_1=20
        # )

        clf = self.build_model(model_name)
        clf.fit(self.x_train, self.y_train)

        val_f1, val_ps, val_rs, val_auc, val_mcc, val_speci = calculate_scores(
            self.X_test, self.y_test, clf
        )

        logging.info(f"****{model_name}****")
        logging.info(f"AUCPR Score: {np.round(val_auc, 4)}")
        logging.info(f"MCC Score: {np.round(val_mcc, 4)}")
        logging.info(f"F1 Score: {np.round(val_f1, 4)}")
        logging.info(f"Precision Score: {np.round(val_ps, 4)}")
        logging.info(f"Specificity Score: {np.round(val_speci, 4)}")
        logging.info(f"Recall/Sensitivity Score: {np.round(val_rs, 4)}")

    # Scores

    def train_wandb(self):
        wandb.init(
            project=f"autopilot_{self.config['model_type']}_{self.config['clinic_filter']}",
            tags=[],
            mode="online",
            config=self.config,
        )
        wandb.run.log_code(".")

        clf = self.build_model(self.config["model_type"])
        clf.fit(self.x_train, self.y_train)

        model_file = self.config["classic_ml_dir"] / Path("tmp_model.pkl")
        with open(model_file, "wb") as file:
            pickle.dump(clf, file)

        # Create a Wandb artifact
        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_file(model_file)
        wandb.log_artifact(artifact)

        calculate_scores(
            self.x_train, self.y_train, clf, log_wandb=True, prefix="train"
        )
        calculate_scores(self.x_val, self.y_val, clf, log_wandb=True, prefix="val")

        calculate_scores(self.x_test, self.y_test, clf, log_wandb=True, prefix="test")

        wandb.run.finish()

    def evaluate_bootstrap(
        self,
        model,
        opti_threshold=None,
        n_iterations=1000,
        confidence_level=0.95,
        log_wandb=True,
    ):
        score_lists = {
            metric: [] for metric in ["auc", "mcc", "f1", "ps", "speci", "sensi", "rs"]
        }

        for i in range(n_iterations):
            x_test_resampled, y_test_resampled = resample(self.x_test, self.y_test)
            f1, ps, rs, auc, mcc, specificity, sensitivity = calculate_scores(
                x_test_resampled, y_test_resampled, model, threshold=opti_threshold
            )
            score_lists["auc"].append(auc)
            score_lists["mcc"].append(mcc)
            score_lists["f1"].append(f1)
            score_lists["ps"].append(ps)
            score_lists["speci"].append(specificity)
            score_lists["sensi"].append(sensitivity)
            score_lists["rs"].append(rs)

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

        if log_wandb:
            run_name = "best_auc_test_threshold" if opti_threshold else "best_auc_test"
            run = wandb.init(
                project=self.config["wb_project_name"],
                name=run_name,
                tags=["bootstrap"],
            )
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
                # model.evaluate_model(
                #     model.val_dataset,
                #     None,
                #     None,
                #     data_type="val",
                #     threshold=calc_threshold[config['clinic_filter']],
                # )
                "Precision",
                "Specificity",
                "Sensitivity",
            ]
            for metric, mean, ci_low, ci_upp in zip(metrics, means, ci_lower, ci_upper):
                table.add_data(
                    metric, mean, f"[{np.round(ci_low, 2)}, {np.round(ci_upp, 2)}]"
                )

            table_title = (
                f"Bootstrap Results on {self.config['clinic_filter']} using {opti_threshold}"
                if opti_threshold
                else f"Bootstrap Results on {self.config['clinic_filter']}"
            )
            wandb.log({table_title: table})
            run.finish()

        return means, stds, ci_lower, ci_upper

    def get_optimal_threshold(self, model):
        precision, recall, threshold = precision_recall_curve(
            self.y_val, model.predict_proba(self.x_val)[:, 1]
        )
        f1_scores = 2 * (precision * recall) / (precision + recall)
        return threshold[np.argmax(f1_scores)]


def load_data(config):
    config["classic_ml_dir"] = config["classic_ml_dir"] / Path(config["clinic_filter"])

    def load_data_split(split_name):
        data_file = f"{config['classic_ml_dir']}/{split_name}_data_array_list.pkl"
        if Path(data_file).exists():
            with open(data_file, "rb") as file:
                return pickle.load(file)
        else:
            ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
            ds.download_and_prepare()

            data = get_dataset(split_name, config)

            data_list_1 = []
            data_list_2 = []
            data_list_3 = []
            labels_list = []

            for item in data:
                (data_1, data_2, data_3), labels = item
                data_list_1.append(data_1.numpy())
                data_list_2.append(data_2.numpy())
                data_list_3.append(data_3.numpy())
                labels_list.append(labels.numpy())

            data_array_1 = np.concatenate(data_list_1)
            data_array_1 = np.reshape(data_array_1, (data_array_1.shape[0], -1))
            data_array_2 = np.concatenate(data_list_2)
            data_array_3 = np.concatenate(data_list_3)

            train_array = np.concatenate(
                [data_array_1, data_array_2, data_array_3], axis=1
            )
            labels_array = np.concatenate(labels_list)

            with open(data_file, "wb") as file:
                pickle.dump((train_array, labels_array), file)

            return train_array, labels_array

    # get split names
    train_split, val_split = return_train_validation_cv_split(cv_idx=1, cv_folds=5)
    train_data = load_data_split(train_split)
    val_data = load_data_split(val_split)
    test_data = load_data_split("test")

    return train_data, val_data, test_data


def main(config) -> None:
    if config["evaluate_bs"]:
        # Evaluation of the model using bootstrap on test data
        with open(config["model_path"], "rb") as file:
            model = pickle.load(file)
        train_data, val_data, test_data = load_data(config)
        trainer = Trainer(
            train_data,
            val_data,
            test_data,
            random_state=42,
            config=config,
        )
        means, stds, ci_lower, ci_upper = trainer.evaluate_bootstrap(model)
        print(f"Bootstrap Means: {means}")
        print(f"Bootstrap Stds: {stds}")
        print(f"Bootstrap 95% CI Lower: {ci_lower}")
        print(f"Bootstrap 95% CI Upper: {ci_upper}")
        threshold = trainer.get_optimal_threshold(model)
        trainer.evaluate_bootstrap(model, opti_threshold=threshold)
        print(f"Bootstrap Means: {means}")
        print(f"Bootstrap Stds: {stds}")
        print(f"Bootstrap 95% CI Lower: {ci_lower}")
        print(f"Bootstrap 95% CI Upper: {ci_upper}")
    else:
        # Training the model, evaluation on train and evaluation set
        train_data, val_data, test_data = load_data(config)

        trainer = Trainer(
            train_data,
            val_data,
            test_data,
            random_state=42,
            config=config,
        )
        # trainer.train_plain("rf")
        trainer.train_wandb()
