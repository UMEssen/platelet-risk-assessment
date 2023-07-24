from cmath import sqrt

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
)
import wandb
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Tuple, Any


from app.data_handling.get_training_data import AutopilotDataset


def calculate_mean(score_dict):
    auc_mean = np.mean(score_dict["auc"])
    mcc_mean = np.mean(score_dict["mcc"])
    f1_mean = np.mean(score_dict["f1"])
    ps_mean = np.mean(score_dict["ps"])
    speci_mean = np.mean(score_dict["speci"])
    sensi_mean = np.mean(score_dict["sensi"])
    rs_mean = np.mean(score_dict["rs"])

    auc_std = np.std(score_dict["auc"])
    mcc_std = np.std(score_dict["mcc"])
    f1_std = np.std(score_dict["f1"])
    ps_std = np.std(score_dict["ps"])
    speci_std = np.std(score_dict["speci"])
    sensi_std = np.std(score_dict["sensi"])
    rs_std = np.std(score_dict["rs"])

    return (
        auc_mean,
        mcc_mean,
        f1_mean,
        ps_mean,
        speci_mean,
        sensi_mean,
        rs_mean,
    ), (
        auc_std,
        mcc_std,
        f1_std,
        ps_std,
        speci_std,
        sensi_std,
        rs_std,
    )


def calculate_scores2(dataset, labels, model, is_deep=False, threshold=None):
    y_pred = model.predict(dataset)
    if is_deep and threshold:
        y_pred = [1 if pred >= threshold else 0 for pred in y_pred]

    f1 = f1_score(labels, y_pred)
    ps = precision_score(labels, y_pred)
    rs = recall_score(labels, y_pred)
    auc_pr = average_precision_score(labels, y_pred)
    mcc = matthews_corrcoef(labels, y_pred)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
    speci = tn / (tn + fp)  # specificity
    sensitivity = tp / (tp + fn)  # recall

    return f1, ps, rs, auc_pr, mcc, speci, sensitivity


def calculate_scores(
    X, y, clf, is_deep=False, threshold=None, log_wandb=False, prefix=""
):
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

    tn, fp, fn, tp = confusion_matrix(y, y_preds).ravel()

    tp = np.float64(tp)
    fp = np.float64(fp)
    tn = np.float64(tn)
    fn = np.float64(fn)
    x = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / x
    speci = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    if log_wandb:
        wandb.log(
            {
                f"{prefix}_f1": f1,
                f"{prefix}_precision": ps,
                f"{prefix}_recall": rs,
                f"{prefix}_auc": auc,
                f"{prefix}_mcc": mcc,
                f"{prefix}_specificity": speci,
                f"{prefix}_sensitivity": sensitivity,
            }
        )
        wandb.log(
            {
                f"{prefix}_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y,
                    preds=y_preds,
                    class_names=["0", "1"],
                    title=prefix,
                )
            }
        )
        wandb.log(
            {
                f"{prefix}_tn": tn,
                f"{prefix}_fp": fp,
                f"{prefix}_fn": fn,
                f"{prefix}_tp": tp,
            }
        )

    return f1, ps, rs, auc, mcc, speci, sensitivity


def return_train_validation_cv_split(cv_idx: int, cv_folds: int = 5) -> Tuple[str, str]:
    splits = [
        f"cv_{idx}_of_{cv_folds}" for idx in range(1, cv_folds + 1) if idx != cv_idx
    ]
    return "+".join(splits), f"cv_{cv_idx}_of_{cv_folds}"


# Todo Edit for categorical feature analysis
def get_dataset_cohort(config: dict):
    # 1. None model
    # 2. Hematoonco
    # 3. Heart Thorax

    ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
    ds.download_and_prepare()

    dataset = pd.DataFrame()
    dataset_names = {
        "heart_thorax": "0.3.1",
        "hematooncology": "0.2.0",
        "none": "0.1.3",
    }

    for dataset_name, dataset_version in dataset_names.items():
        split_names = ["test", "cv_1_of_5+cv_2_of_5+cv_3_of_5+cv_4_of_5+cv_5_of_5"]
        for split in split_names:
            if split == "test":
                is_test = True

            ds, info = tfds.load(
                f"autopilot_dataset:{dataset_version}",
                split=split,
                data_dir=config["dataset_dir"],
                with_info=True,
            )

            ds = ds.map(
                lambda x: get_gender_age(x, dataset_name, is_test),
                num_parallel_calls=64,
            )
            # turn the dataset into a pandas dataframe
            ds = tfds.as_dataframe(ds)
            # append to dataset
            dataset = dataset.append(ds, ignore_index=True)

    print(dataset.head())
    pd.to_pickle(dataset, config["age_gender_metas_model_based"])


def get_gender_age(sample, model_type: str, is_test=False) -> dict:
    patient_id = sample["patient_id"]
    age = tf.cast(sample["mt_features"][-1], tf.float32)
    gender = tf.cast(sample["mt_features"][-2], tf.float32)
    gender = tf.where(gender == -1, "male", "female")

    train_type = "test" if is_test else "train"

    return {
        "patient_id": patient_id,
        "type": train_type,
        "age": age,
        "gender": gender,
        "model_type": model_type,
    }


def get_dataset(
    split_names: str,
    config: dict,
    do_split_label=True,
    do_normalize_featuers=True,
    do_binarize_label=True,
    do_binarize_tk=True,
):
    ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
    ds.download_and_prepare()

    ds, info = tfds.load(
        f"autopilot_dataset:{config['dataset_name']}",
        split=split_names,
        data_dir=config["dataset_dir"],
        shuffle_files=config["shuffle_files"],
        with_info=True,
        as_supervised=False,
    )

    # ds = ds.map(lambda x: check_valid(x), num_parallel_calls=1)

    if do_normalize_featuers:
        scaler_dict = pd.read_pickle(config["scalars_path"])
        ds = ds.map(
            lambda x: normalize_features(x, scaler_dict),
            num_parallel_calls=60,
        )

    if do_binarize_label:
        ds = ds.map(lambda x: binarize_label(x), num_parallel_calls=64)

    if do_binarize_tk:
        ds = ds.map(lambda x: binarize_tk(x), num_parallel_calls=64)

    if do_split_label:
        ds = ds.map(lambda x: split_label(x), num_parallel_calls=64)

    if config["batch_size"] is not None:
        ds = ds.batch(config["batch_size"], drop_remainder=True)
    if config["prefetch"]:
        ds = ds.prefetch(None)
    return ds


def normalize_features(sample, scaler_dict):
    # Normalize Age
    age_scaler = scaler_dict["age_scaler"]
    age = tf.cast(sample["mt_features"][-1], tf.float32)
    age_norm = normalize_data(age, age_scaler)
    sample["mt_features"] = tf.concat(
        [sample["mt_features"][:-1], tf.reshape(age_norm, shape=(1,))], axis=0
    )

    # Normalize zthra
    zthra_scaler = scaler_dict["zthra_scaler"]
    zthra = tf.cast(sample["sq_features"][:, 0], tf.float32)

    if tf.reduce_any(zthra == 1024):
        zthra_mask = tf.ones(tf.size(tf.where(zthra == 1024))) * 1024
        zthra_mask = tf.cast(zthra_mask, dtype=tf.float32)

        zthra_raw = zthra[tf.size(zthra_mask) :]

        zthra_norm = normalize_data(zthra_raw, zthra_scaler)
        zthra_norm = tf.cast(zthra_norm, dtype=tf.float32)
        zthra_norm = tf.concat([zthra_mask, zthra_norm], 0)

    else:
        zthra_norm = normalize_data(zthra, zthra_scaler)

    zthra_norm = my_tf_round(zthra_norm, 2)

    # Normalize tk
    # tkz_scaler = scaler_dict["tkz_scaler"]
    # tkz = tf.cast(sample["sq_features"][:, 1], tf.float32)
    # tkz_norm = normalize_data(tkz, tkz_scaler)

    sample["sq_features"] = tf.concat(
        [
            tf.reshape(zthra_norm, shape=(-1, 1)),
            # tf.reshape(tkz_norm, shape=(None, 1)),
            sample["sq_features"][:, 1:],
        ],
        axis=1,
    )

    # tf.logging.info(sample["sq_features"])
    return sample


def binarize_label(sample):
    sample["label"] = tf.where(sample["label"] > 0, 1, 0)
    return sample


def binarize_tk(sample):
    tk_raw = sample["sq_features"][:, 1]

    # masked values shall not be binarized
    if tf.reduce_any(tk_raw == 1024):
        tk_mask = tf.ones(tf.size(tf.where(tk_raw == 1024))) * 1024
        tk_mask = tf.cast(tk_mask, dtype=tf.float32)

        tk_raw = tk_raw[tf.size(tk_mask) :]
        tk_raw = tf.cast(tk_raw, dtype=tf.float32)
        tk_raw = tf.cast(tf.where(tk_raw > 0, 1, -1), tf.float32)

        tkzs = tf.concat([tk_mask, tk_raw], 0)
    else:
        tkzs = tf.cast(tf.where(tk_raw > 0, 1, -1), tf.float32)
    zthra = tf.cast(sample["sq_features"][:, 0], tf.float32)

    sample["sq_features"] = tf.concat(
        [
            tf.reshape(zthra, shape=(-1, 1)),
            tf.reshape(tkzs, shape=(-1, 1)),
            sample["sq_features"][:, 2:],
        ],
        axis=1,
    )

    return sample


def split_label(sample) -> tuple:
    return (
        sample["sq_features"],
        sample["mt_features"],
        sample["station_features"],
    ), sample["label"]


def normalize_data(data: Any, n_scaler: sklearn.preprocessing.StandardScaler) -> Any:
    mean_n = tf.constant(n_scaler.mean_[0], dtype=tf.float32)
    std_n = tf.constant(n_scaler.scale_[0], dtype=tf.float32)
    n = tf.cast(data, tf.float32)
    n_norm = (n - mean_n) / std_n
    return n_norm


def inverse_normalize_data(
    data: Any, n_scaler: sklearn.preprocessing.StandardScaler
) -> Any:
    mean_n = tf.constant(n_scaler.mean_[0], dtype=tf.float32)
    std_n = tf.constant(n_scaler.scale_[0], dtype=tf.float32)
    n_norm = tf.cast(data, tf.float32)
    n = (n_norm * std_n) + mean_n
    return n


def my_tf_round(x, decimals=0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier
