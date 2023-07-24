import datetime
import logging
import pathlib
import random
from math import sqrt
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Zero balance calculator
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from colorlog import ColoredFormatter
from matplotlib.colors import LinearSegmentedColormap
from numpy import argmax
from scipy import stats
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from app.data_handling.create_sliding_window import (
    col_to_datetime,
    get_clinic_from_stations,
)
from app.data_handling.get_training_data import (
    calculate_age,
    transform_dict_constants,
    AutopilotDataset,
)
from app.fhir_extraction.extract_transform_validate import FHIRExtractor

#### logging
from app.ml.train_helper import (
    get_dataset,
    return_train_validation_cv_split,
    calculate_scores,
)

logger = logging.getLogger(__name__)
LOG_LEVEL = logging.DEBUG
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(
    "%(log_color)s[%(levelname).1s %(log_color)s%(asctime)s] - %(log_color)s%(name)s %(reset)s- "
    "%(message)s"
)

plt.rcParams["font.family"] = "DejaVu Sans"

with open("app/config/constants.yaml", "r") as stream:
    code_dict = yaml.safe_load(stream)


def filter_date(
    start: datetime, end: datetime, resource: pd.DataFrame, date_col: str
) -> pd.DataFrame:
    df = resource[
        ((start <= resource[date_col]) & (resource[date_col] <= end))
    ].sort_values([date_col])

    return df


def get_zero_balance(config):
    wins = pd.read_pickle(config["root_path"] + "/training_training_data/windows.pkl")
    all_labels = []

    for win in wins:
        for count, pat in enumerate(wins[win]):
            if count == 0:
                continue
            all_labels.append(pat["label"][0])

    zeros = pd.Series(all_labels).value_counts()
    print(f"Zero values: {zeros[0]}, Ones: {zeros[1]} ")
    print(f"Distribution: {zeros[1] / (zeros[0] + zeros[1])} ")


def get_labels(ds):
    return np.concatenate([y for x, y in ds], axis=0)


def cohort_selection(config):
    obs_raw = pd.read_feather(config["obs_path"])

    obs_raw.dropna(how="all", inplace=True)
    obs_raw.dropna(subset=["value"], inplace=True)
    obs_raw["date"] = col_to_datetime(obs_raw["observation_date"])
    obs_raw = filter_date(
        config["start_datetime"], config["end_datetime"], obs_raw, "date"
    )
    len_obs_raw = len(obs_raw["patient_id"].unique())
    print(f"Raw patients:{len_obs_raw}")

    obs = pd.read_feather(config["obs_path_filtered"])
    len_obs = len(obs["patient_id"].unique())
    print(f"Excluded patients (platelet count < 450/nl): {len_obs_raw - len_obs}")

    print(f"Patients (platelet count < 450/nl):{len_obs}")

    wins = pd.read_pickle(config["preprocessed_windows_path"])
    pats_windows = len(wins.keys())

    print(f"Excluded patients (5 crit): {len_obs - pats_windows}")
    print(f"Patients (5 crit): {pats_windows}")

    all_labels = []
    for win in wins:
        for count, pat in enumerate(wins[win]):
            if count == 0:
                continue
            all_labels.append(0)

    print(f"Number of training samples {len(all_labels)}")


def get_gender_age_distribution(config):
    if not Path(config["age_gender_metas"]).exists():
        wins = pd.read_pickle(config["preprocessed_windows_path"])
        # equals the shuffling for ts ds
        shuffled_pats = list(wins.keys())
        random.Random(9).shuffle(shuffled_pats)
        split_count = int(len(wins) * 0.8)
        train_pats, test_pats = shuffled_pats[:split_count], shuffled_pats[split_count:]
        all_pat_metas = pd.read_feather(config["patient_path_filtered"])
        all_pat_metas["age"] = [
            calculate_age(pd.to_datetime(x), datetime.datetime.today())
            for x in all_pat_metas["birth_date"]
        ]
        all_pat_metas.drop(columns=["birth_date"])
        train_metas = all_pat_metas.loc[all_pat_metas["patient_id"].isin(train_pats)]
        train_metas["type"] = "train"
        test_metas = all_pat_metas.loc[all_pat_metas["patient_id"].isin(test_pats)]
        test_metas["type"] = "test"
        pd.to_pickle((train_metas, test_metas), config["age_gender_metas"])
        return train_metas, test_metas
    else:
        return pd.read_pickle(config["age_gender_metas"])


def plot_characteristics(config, metas):
    my_pal = {"train": "#9CA0A6", "test": "#FC8181"}
    fig, (ax_boxplot, ax_pie, ax_pie_pats) = plt.subplots(
        1, 3, sharex=False, figsize=(15, 5), constrained_layout=True
    )

    fig.suptitle("Figure 3", fontsize=13, x=0, ha="left")
    # BOX
    sns.boxplot(
        palette=my_pal,
        ax=ax_boxplot,
        x="gender",
        y="age",
        hue="type",
        data=metas,
        showfliers=False,
    )
    sns.boxplot()
    handles, _ = ax_boxplot.get_legend_handles_labels()

    ax_boxplot.legend(handles, ["Train", "Test"], title="Dataset")
    ax_boxplot.set_title("Age Distribution by Gender")
    ax_boxplot.set_xlabel("Gender")
    ax_boxplot.set_ylabel("Age")
    ax_boxplot.spines["top"].set_visible(False)
    ax_boxplot.spines["right"].set_visible(False)
    # ax_boxplot.set_facecolor('#D3D3D3')
    ax_boxplot.annotate(
        "A", xy=(0, 1.1), xycoords="axes fraction", fontsize=13, fontweight="bold"
    )

    # Pie demand of bdp
    bdp = pd.read_feather(config["bdp_path_filtered"])
    station_index = get_clinic_from_stations(bdp.output_to_einscode, first_only=False)
    station_index = [x[0] for x in station_index]
    bdp["clinic"] = station_index
    # Count unique patients within patient_counts
    patient_counts = bdp.groupby("clinic")["patient_id"].nunique()

    bdp_counts = bdp.output_to_einscode.value_counts().sort_index()
    station_index = get_clinic_from_stations(bdp_counts.index, first_only=False)
    station_index = [x[0] for x in station_index]
    bdp_counts.index = station_index
    bdp_counts = bdp_counts.groupby(level=0).sum().sort_values(ascending=False)
    bdp_counts_max = bdp_counts.nlargest(2)
    bdp_counts_min = bdp_counts.nsmallest(len(bdp_counts) - 2)
    bdp_counts_min = pd.Series([sum(bdp_counts_min)], index=["other"])
    pie_counts = pd.concat([bdp_counts_max, bdp_counts_min])

    data = pie_counts.values
    labels = pie_counts.index
    print(labels)
    print(data)
    # colors = ["#2D3748", "#9CA0A6", "#FC8181", "#F2ACAC", "#F2D8D8"]
    colors = ["#9CA0A6", "#FC8181", "#F2ACAC", "#F2D8D8"]
    labels_tmp = [
        f"Hematology-Oncology",
        f"Cardio-Thoracic Surgery",
        f"Other",
    ]

    ax_pie.pie(data, labels=labels_tmp, colors=colors, autopct="%.0f%%")
    ax_pie.set_title("Platelet Demand by Clinic")
    ax_pie.annotate(
        "B",
        xy=(0, 1.1),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
    )

    # Define the specific clinics
    # specific_clinics = ["heart_thorax", "hematooncology", "pediatrics"]
    specific_clinics = ["heart_thorax", "hematooncology"]

    # Get the counts for the specific clinics
    specific_clinic_counts = patient_counts[patient_counts.index.isin(specific_clinics)]

    # Get the counts for the other clinics
    other_clinics = patient_counts[~patient_counts.index.isin(specific_clinics)]
    other_sum = pd.Series([other_clinics.sum()], index=["Other"])

    # Concatenate specific clinic counts and other
    final_patient_counts = pd.concat([specific_clinic_counts, other_sum])

    labels_tmp = [
        f"Hematology-Oncology",
        f"Cardio-Thoracic Surgery",
        f"Other",
    ]
    ax_pie_pats.pie(
        final_patient_counts, labels=labels_tmp, colors=colors, autopct="%.0f%%"
    )
    ax_pie_pats.set_title("Unique Patients Receiving \n Platelets by Dataset")
    ax_pie_pats.annotate(
        "C",
        xy=(0, 1.1),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
    )

    # pie number of pats
    plt.show()
    plt.savefig("3.3 Figure 3.pdf", format="pdf", dpi=600)
    # plt.savefig(config["image"], format="jpeg", dpi=600)


def find_clinic(station):
    for clinic, stations in code_dict["CLINIC_BY_STATION"].items():
        if station in stations:
            return clinic


def get_clinic_analysis(df_metas, df_bdp, gender: str):
    filtered_bdp = df_bdp[df_bdp["patient_id"].isin(df_metas["patient_id"])]
    merged_bdp = filtered_bdp.merge(
        df_metas[["patient_id", "gender"]], on="patient_id", how="left"
    )
    merged_bdp["station"] = merged_bdp["output_to_einscode"].apply(find_clinic)

    # Count the occurrences of each station
    station_counts = merged_bdp.station.where(
        merged_bdp.gender == gender
    ).value_counts()

    # Get the counts for 'hematooncology' and 'heart_thorax'
    count_hematooncology = station_counts.get("hematooncology", 0)
    count_heart_thorax = station_counts.get("heart_thorax", 0)

    # Sum of all stations excluding 'hematooncology' and 'heart_thorax'
    sum_other = station_counts[
        ~station_counts.index.isin(["hematooncology", "heart_thorax"])
    ].sum()

    # Calculate the total count
    total_count = count_hematooncology + count_heart_thorax + sum_other

    # Calculate the percentage in relation to the total count
    percentage_hematooncology = count_hematooncology / total_count * 100
    percentage_heart_thorax = count_heart_thorax / total_count * 100
    percentage_sum_other = sum_other / total_count * 100

    return (
        count_hematooncology,
        count_heart_thorax,
        sum_other,
        total_count,
        percentage_hematooncology,
        percentage_heart_thorax,
        percentage_sum_other,
    )


def get_cohort_characteristics(config, plot=False):
    (train_metas, test_metas) = get_gender_age_distribution(config)
    metas = pd.concat([train_metas, test_metas])

    # Perform Shapiro-Wilk test
    shapiro_result = stats.shapiro(metas.age)

    # Check if the p-value is less than 0.05 (common significance level)
    if shapiro_result.pvalue < 0.05:
        print("The age data is not normally distributed.")
    else:
        print("The age data is normally distributed.")

    print(metas["age"].describe())
    exit(0)

    # Meadiean is 50 %
    print(metas.groupby(["gender", "type"])["age"].describe())

    bdp = pd.read_feather(config["bdp_path_filtered"])
    # Create a dictionary to store the results
    results = {}
    results["Male Training"] = get_clinic_analysis(train_metas, bdp, "male")
    results["Male Test"] = get_clinic_analysis(test_metas, bdp, "male")
    results["Female Training"] = get_clinic_analysis(train_metas, bdp, "female")
    results["Female Test"] = get_clinic_analysis(test_metas, bdp, "female")

    df_results = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=[
            "Hematooncology",
            "Heart_Thorax",
            "Sum Other",
            "Total Count",
            "Percentage Hematooncology",
            "Percentage Heart_Thorax",
            "Percentage Sum Other",
        ],
    )

    print("Clinic Analysis Results:")
    print(df_results)

    if plot:
        plot_characteristics(config, metas)


def count_category_occurrences():
    # Initialize a dictionary to store the counts
    count_dict = {}

    # Loop through each category and count the number of occurrences
    for category in code_dict["CATEGORY_DICT_LIST"]:
        count_dict[category] = 0
        for dict_list in [
            "CONDITION_DICT_LIST",
            "PROCEDURE_DICT_LIST",
            "MEDICATION_DICT_LIST",
        ]:
            for key in code_dict[dict_list]:
                if key in code_dict["CATEGORY_DICT_LIST"][category]:
                    count_dict[category] += len(code_dict[dict_list][key])

    return count_dict


def get_ops_icd_medication_occurance(config, split_by_gender=False):
    medis = pd.read_feather(config["medication_merged_path"])
    pro = pd.read_feather(config["procedure_path_filtered"])
    cond = pd.read_feather(config["condition_path_filtered"])

    code_dict = transform_dict_constants()
    med_dict = code_dict["MEDICATION_DICT_LIST"]
    pro_dict = code_dict["PROCEDURE_DICT_LIST"]
    con_dict = code_dict["CONDITION_DICT_LIST"]

    data_set_names = ["train", "test"]
    if split_by_gender:
        data_set_names = [
            f"{ds}_{gender}" for ds in data_set_names for gender in ["male", "female"]
        ]

    medi_sets_df = pd.DataFrame(index=med_dict.keys(), columns=data_set_names, data=0)
    pro_sets_df = pd.DataFrame(index=pro_dict.keys(), columns=data_set_names)
    cond_sets_df = pd.DataFrame(index=con_dict.keys(), columns=data_set_names)

    (train_metas, test_metas) = get_gender_age_distribution(config)
    metas_dict = {"train": train_metas, "test": test_metas}

    for data_set in ["train", "test"]:
        for gender in ["male", "female"]:
            meta = metas_dict[data_set]
            pats = meta[meta["gender"] == gender] if split_by_gender else meta

            # Medications
            medis_scope = medis[medis["patient_id"].isin(pats.patient_id.values)]
            medis_counts = pd.Series(medis_scope.medicationName.values).value_counts()
            medis_counts = pd.DataFrame(medis_counts).reset_index()
            medis_counts.columns = ["med_name", "count"]
            for med in sum(code_dict["MEDICATION_DICT_LIST"].values(), []):
                medis_counts.index = medis_counts.med_name.apply(
                    lambda x: next(
                        iter(
                            [
                                med
                                for med in sum(
                                    code_dict["MEDICATION_DICT_LIST"].values(), []
                                )
                                if med.lower() in x.lower()
                            ]
                        ),
                        np.nan,
                    )
                )
            medis_counts = medis_counts[pd.notnull(medis_counts.index)]
            medis_counts = medis_counts.groupby(medis_counts.index).sum()

            for med_name, med_counts in medis_counts["count"].items():
                for med_dict_name in medi_sets_df.index:
                    if med_name == med_dict_name:
                        medi_sets_df.loc[med_dict_name][
                            f"{data_set}_{gender}"
                        ] += med_counts

            # Conditions
            cond_scope = cond[
                cond["patient_id"].isin(pats.patient_id.values)
            ].icd_code.values
            cond_keys = list()

            for element in cond_scope:
                key = list(
                    (
                        k
                        for k, v in con_dict.items()
                        if any(word in element for word in v)
                    )
                )
                if not len(key):
                    print("cannot classify")
                    exit()
                cond_keys.append(key)

            cond_counts = pd.DataFrame(cond_keys).value_counts()

            # Procedures
            pro_scope = pro[pro["patient_id"].isin(pats.patient_id.values)]
            codes = pro_scope.code.values
            pro_keys = list()
            for element in codes:
                element_og = element
                element = element.split(".")[0]
                key = list(
                    (
                        k
                        for k, v in pro_dict.items()
                        if any(word in element for word in v)
                    )
                )
                if not len(key):
                    key = list(
                        (
                            k
                            for k, v in pro_dict.items()
                            if any(word in element_og for word in v)
                        )
                    )
                    if not len(key):
                        print("cannot classify")
                        exit()
                    if len(key) > 1:
                        print(f"more than one match init: {element}, key:{key}")
                if len(key) > 1:
                    key = list(
                        (
                            k
                            for k, v in pro_dict.items()
                            if any(word == element for word in v)
                        )
                    )
                    if len(key) > 1:
                        print(f"more than one match init: {element}, key:{key}")
                pro_keys.append(key)

            pro_counts = pd.DataFrame(
                [item for sublist in pro_keys for item in sublist]
            ).value_counts()

            pro_reindexed = pd.DataFrame({f"{data_set}_{gender}": pro_counts})
            cond_reindexed = pd.DataFrame({f"{data_set}_{gender}": cond_counts})

            pro_reindexed.reset_index(drop=False, inplace=True)
            pro_reindexed.index = pro_reindexed[0]
            pro_reindexed.drop(columns=[0], inplace=True)

            cond_reindexed.reset_index(drop=False, inplace=True)
            cond_reindexed.index = cond_reindexed[0]
            cond_reindexed.drop(columns=[0], inplace=True)

            pro_sets_df = pd.concat([pro_sets_df, pro_reindexed], axis=1)
            cond_sets_df = pd.concat([cond_sets_df, cond_reindexed], axis=1)

            pro_sets_df = pro_sets_df.fillna(0).astype(int)
            cond_sets_df = cond_sets_df.fillna(0).astype(int)

    medi_sets_df.to_pickle(config["medication_counts"])
    pro_sets_df.to_pickle(config["procedure_counts"])
    cond_sets_df.to_pickle(config["condition_counts"])

    # Check for duplicate columns
    cond_sets_df = cond_sets_df.groupby(level=0, axis=1).sum()  # Drop duplicate columns
    pro_sets_df = pro_sets_df.groupby(level=0, axis=1).sum()

    # Drop duplicate columns

    logging.info("------MEDICATION------")
    logging.info(medi_sets_df)
    logging.info("------Conditions------")
    logging.info(cond_sets_df)
    logging.info("------Procedures------")
    logging.info(pro_sets_df)

    category_counts = {k: 0 for k in code_dict["CATEGORY_DICT_LIST"].keys()}
    category_counts_df = pd.DataFrame(category_counts, index=[0]).T
    category_counts_df.columns = ["test_female"]
    category_counts_df["test_male"] = 0  # Add 'test_male' column
    category_counts_df["train_female"] = 0  # Add 'train_female' column
    category_counts_df["train_male"] = 0

    if split_by_gender:
        with (pathlib.Path(__file__).parent.parent / "config" / "constants.yaml").open(
            "r"
        ) as stream:
            code_dict = yaml.safe_load(stream)

        for category, key_group in code_dict["CATEGORY_DICT_LIST"].items():
            print(category, key_group)
            for key in key_group:
                if key in medi_sets_df.index:
                    for set_gender in medi_sets_df.columns:
                        category_counts_df.loc[category][
                            set_gender
                        ] += medi_sets_df.loc[key][set_gender]
                if key in cond_sets_df.index:
                    for set_gender in cond_sets_df.columns:
                        category_counts_df.loc[category][
                            set_gender
                        ] += cond_sets_df.loc[key][set_gender]
                if key in pro_sets_df.index:
                    for set_gender in pro_sets_df.columns:
                        category_counts_df.loc[category][set_gender] += pro_sets_df.loc[
                            key
                        ][set_gender]

        logging.info("------Category Counts------")
        new_column_order = ["train_male", "test_male", "train_female", "test_female"]
        category_counts_df = category_counts_df.reindex(columns=new_column_order)

        logging.info(category_counts_df)
        category_counts_df.to_pickle(config["category_counts"])


def check_meds(config):
    encs = pd.read_pickle(
        "/nvme/shared/autopilot/training_training_data/preprocessed_by_encounter.pkl"
    )

    for pat in encs:
        for count, df in enumerate(encs[pat]):
            if count == 0:
                continue
            if not df.substance.isna().any():
                print("w8")


class EvaluateNnModel:
    def __init__(self, config):
        self.config = config
        self.model = (
            tf.keras.models.load_model(
                "app/config/model_"
                + self.config["clinic_filter"]
                + "/ensemble_model.h5"
            )
            if config["eval_ensemble_bs"]
            else tf.keras.models.load_model(
                f"./app/config/model_{config['clinic_filter']}/cv_5"
            )
        )
        # for index in range(1, 5 + 1):
        #     model_dir = Path(
        #         "app/config/model_" + self.config["clinic_filter"] + f"/cv_{index}"
        #     )
        #     self.model = tf.keras.models.load_model(model_dir)
        ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
        ds.download_and_prepare()
        train_splits, val_splits = return_train_validation_cv_split(
            cv_idx=1, cv_folds=5
        )
        self.train_dataset = get_dataset(train_splits, config)
        self.val_dataset = get_dataset(val_splits, config)
        self.test_dataset = get_dataset("test", config)

        print(f'****** {config["clinic_filter"]} ******')
        print(f'****** {config["eval_ensemble_bs"]} ******')
        print(self.model.summary())

    def get_scores(self, ds):
        metrics = self.model.evaluate(ds)
        tp = metrics[1]
        fp = metrics[2]
        tn = metrics[3]
        fn = metrics[4]
        precision = metrics[5]
        recall = metrics[6]
        auc_roc = metrics[7]
        auc_pr = metrics[8]

        f1 = tp / (tp + 0.5 * (fp + fn))
        x = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / x
        speci = tn / (tn + fp)

        return f1, precision, recall, auc_roc, mcc, speci, auc_pr

    def deep_nn_scores(self, threshold=None, return_val_scores=False) -> Optional[dict]:
        train_tup = ("TRAIN", self.train_dataset)
        val_tup = ("VAL", self.val_dataset)
        test_tup = ("TEST", self.test_dataset)

        print(
            f'Model for {self.config["clinic_filter"]}, Dataset_Version: {self.config["dataset_name"]}'
        )

        for ds in [train_tup, val_tup, test_tup]:
            print("    --------------------------")
            print(f"************{ds[0]}***************")
            print("    --------------------------")
            if threshold:
                labels = get_labels(ds[1])
                # labels = functools.reduce(operator.iconcat, labels, [])
                # x = Trainer(
                #     [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], None, None
                # )
                # (f1, ps, rs, auc_pr, mcc, speci) = x.calculate_scores(
                #     ds[1], labels, self.model, False, threshold
                # )
                (f1, ps, rs, auc_pr, mcc, speci, sensitivity) = calculate_scores(
                    ds[1], labels, self.model, is_deep=True, threshold=threshold
                )
                roc_auc = 0

            else:
                (f1, ps, rs, roc_auc, mcc, speci, auc_pr) = self.get_scores(ds[1])

            logging.info(f"AUCPR Score: {np.round(auc_pr, 4)}")
            logging.info(f"MCC Score: {np.round(mcc, 4)}")
            logging.info(f"F1 Score: {np.round(f1, 4)}")
            logging.info(f"Precision Score: {np.round(ps, 4)}")
            logging.info(f"Specificity Score: {np.round(speci, 4)}")
            logging.info(f"Recall/Sensitivity Score: {np.round(rs, 4)}")

            if return_val_scores and "VAL" in ds[0]:
                return {
                    "auc_pr": auc_pr,
                    "mcc": mcc,
                    "f1": f1,
                    "precision": ps,
                    "specificity": speci,
                    "recall": rs,
                }


def get_main_conditions_by_cohort(config):
    config["dataset_name"] = "0.2.0"
    config["input_dim_station_embedding"] = 21
    config["clinic_filter"] = "hematooncology"

    train_splits, val_splits = return_train_validation_cv_split(cv_idx=1, cv_folds=5)
    split_names = val_splits + "+" + train_splits + "+test"

    ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
    ds.download_and_prepare()

    ds = tfds.load(
        f"autopilot_dataset:{config['dataset_name']}",
        split=split_names,
        data_dir=config["dataset_dir"],
        shuffle_files=False,
    )

    patient_ids = []
    start_dates = []
    for example in ds.take(-1):
        patient_id = example["patient_id"].numpy().decode("utf-8")
        start_date = example["start_date"].numpy().decode("utf-8")
        patient_ids.append(patient_id)
        start_dates.append(pd.to_datetime(start_date).date())

    df = pd.DataFrame({"patient_id": patient_ids, "start_date": start_dates})

    if (
        not Path("/nvme/shared/autopilot/training_publish/main_diag.ftr").exists()
        or True
    ):
        extract = FHIRExtractor(config)
        extract.build_main_conditions_by_cohort(df)


class EvaluateCVNnModel(EvaluateNnModel):
    def __init__(self, config):
        super().__init__(config)

    def evaluate_cv(self):
        scores = []
        for i, cv_id in enumerate(range(1, 6), start=1):
            self.train_splits, self.val_splits = return_train_validation_cv_split(
                cv_idx=cv_id, cv_folds=5
            )
            self.val_dataset = get_dataset(self.val_splits, self.config)
            self.model = tf.keras.models.load_model(
                f"./app/config/model_{self.config['clinic_filter']}/cv_{i}"
            )
            scores.append(self.deep_nn_scores(return_val_scores=True))

        # print scores in a combinded manner AUCPR: Model CV1, MOdel CV2 etc
        keys = scores[0].keys()
        for key in keys:
            print(f"{key}: ", end="")
            for score_dict in scores:
                print(f"{np.round(score_dict[key], 4)}, ", end="")
            print()


def set_labels(ax, row, col):
    # Clear all labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Set the y-label for the leftmost column
    if col == 0 and row == 0:
        ax.set_ylabel("True Positive Rate")
    if col == 0 and row == 1:
        ax.set_ylabel("Precision")
    if col == 0 and row == 2:
        ax.set_ylabel("True Label")

    if col == 1 and row == 0:
        ax.set_xlabel("False Positive Rate")
    if col == 1 and row == 1:
        ax.set_xlabel("Recall")
    if col == 1 and row == 2:
        ax.set_xlabel("Predicted Label")

    return ax


def plot_roc_curve(y_test, y_pred, model_title, ax, annotation_text):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="black", label=f"ROC curve (area = {roc_auc:.2f})")
    idx = np.argwhere(np.diff(np.sign(tpr - fpr))).flatten()
    ax.fill_between(fpr, tpr, 0, alpha=0.2, color="gray")
    ax.plot([0, 1], [0, 1], "k--", label="No Skill")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.annotate(
        annotation_text,
        xy=(0, 1.1),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(model_title)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")

    return ax


def plot_precision_recall_curve(
    y_test, y_pred, model_title, ax, target_threshold, annotation_text
):
    precision, recall, threshold = precision_recall_curve(y_test, y_pred)
    ax.plot(recall, precision, color="black", label=f"AUC-PR by Threshold")
    ax.fill_between(recall, precision, 0, alpha=0.2, color="gray")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.1, 1.05])

    if not threshold.any():
        fscore = (2 * precision * recall) / (precision + recall)
        ix = argmax(pd.Series(fscore))
    else:
        # Your target threshold.
        diff = np.abs(np.array(threshold) - target_threshold)
        ix = np.argmin(diff)
    ax.scatter(
        recall[ix],
        precision[ix],
        marker="o",
        color="#FC8181",
        label="Optimal Threshold",
    )
    ax.annotate(f"   {threshold[ix]:0.2f}", (recall[ix], precision[ix]))
    sns.despine(ax=ax)

    ax.annotate(
        annotation_text,
        xy=(0, 1.1),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(model_title)

    # h line
    ax.axhline(0, color="black", ls="--", label="No Skill")

    return ax


def plot_confusion_matrix(y_test, y_pred, model_title, ax, annotation_text):
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#9CA0A6", "#FC8181"]
    )
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap=custom_cmap, values_format="d", text_kw={"color": "black"})

    ax.annotate(
        annotation_text,
        xy=(0, 1.1),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(model_title, fontsize=13)
    return ax


def generate_model_performance_figures(config):
    # Model configurations
    model_configs = [
        {
            "clinic_filter": "heart_thorax",
            "dataset_name": "0.3.1",
            "input_dim_station_embedding": 18,
            "title": "Cardio-Thoracic Surgery Patient Model",
        },
        {
            "clinic_filter": "hematooncology",
            "dataset_name": "0.2.0",
            "input_dim_station_embedding": 21,
            "title": "Hematology Oncology Patient Model",
        },
        {
            "clinic_filter": "none",
            "dataset_name": "0.1.3",
            "input_dim_station_embedding": 291,
            "title": "All Other Clinics and Wards Model",
        },
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)  # adjust width and height space
    fig.suptitle("Figure 5", fontsize=13, x=0, ha="left")

    for ax in axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)  # Adjust as needed
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)  # Adjust as needed

    annot_letters = ["A", "D", "G", "B", "E", "H", "C", "F", "I"]
    a = 0
    # Replace the following line with the optimal threshold values you have determined for each model
    thresholds = [0.39, 0.41, 0.47]
    for m, model_config in enumerate(model_configs):
        config.update(model_config)

        test_dataset = get_dataset("test", config)
        model_dir = Path(
            f"app/config/model_{config['clinic_filter']}/ensemble_model.h5"
        )

        model = tf.keras.models.load_model(model_dir)
        print(model.summary())

        # Separate features and labels in the dataset
        sq_features_test = []
        mt_features_test = []
        station_features_test = []
        y_test = []

        for (sq_features, mt_features, station_features), labels in test_dataset:
            sq_features_test.append(sq_features.numpy())
            mt_features_test.append(mt_features.numpy())
            station_features_test.append(station_features.numpy())
            y_test.append(labels.numpy())

        # Convert list to numpy arrays
        sq_features_test = np.concatenate(sq_features_test)
        mt_features_test = np.concatenate(mt_features_test)
        station_features_test = np.concatenate(station_features_test)
        y_test = np.concatenate(y_test)

        x_test = [sq_features_test, mt_features_test, station_features_test]

        # Generate predictions
        y_pred = model.predict(x_test)
        y_pred_binary = np.where(y_pred > thresholds[m], 1, 0)

        # Plot ROC curve for this model
        axes[0, m] = plot_roc_curve(
            y_test, y_pred, model_config["title"], axes[0, m], annot_letters[a]
        )
        axes[0, m] = set_labels(axes[0, m], 0, m)
        a += 1

        # Plot Precision-Recall curve for this model
        axes[1, m] = plot_precision_recall_curve(
            y_test,
            y_pred,
            model_config["title"],
            axes[1, m],
            thresholds[m],
            annot_letters[a],
        )
        axes[1, m] = set_labels(axes[1, m], 1, m)
        a += 1

        # Plot Confusion Matrix for this model
        axes[2, m] = plot_confusion_matrix(
            y_test, y_pred_binary, model_config["title"], axes[2, m], annot_letters[a]
        )
        axes[2, m] = set_labels(axes[2, m], 2, m)
        a += 1

        axes[0, m].legend(loc="lower right")
    axes[1, -1].legend(loc="upper right")

    fig.tight_layout()
    # Save the figure
    plt.savefig(
        config["image_test_results"], format="pdf", dpi=600, bbox_inches="tight"
    )


def main(config) -> None:
    get_zero_balance(config)
    cohort_selection(config)

    get_cohort_characteristics(config)
    get_main_conditions_by_cohort(config)

    get_ops_icd_medication_occurance(config, split_by_gender=True)
    check_meds(config)

    # Calculate the performance of cross validation models on validation set
    EvaluateCVNnModel(config).evaluate_cv()

    # Generate a Figure wiht AURPR AUCROC and Confusion Matrix for all models
    generate_model_performance_figures(config)


if __name__ == "__main__":
    main()
