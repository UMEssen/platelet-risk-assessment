import datetime
import logging
import pathlib
import random
from typing import Any, Generator, List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml

from app.data_handling.create_sliding_window import find_clinic


def get_stations_int_encoding(
    pat_stationed_at: str, clinic: str, clinic_by_station_dict: dict
) -> np.array:
    if pat_stationed_at not in clinic_by_station_dict[clinic]:
        return np.array([0])
    return np.array([clinic_by_station_dict[clinic].index(pat_stationed_at) + 1])


def get_clinic_station_index(
    pat_stationed_at: str,
    clinic: str,
    station_index_dict: dict,
    clinic_index_dict: dict,
) -> np.array:
    if clinic not in clinic_index_dict or pat_stationed_at not in station_index_dict:
        return np.array([18, 6])
    return np.array([clinic_index_dict[clinic], station_index_dict[pat_stationed_at]])


def get_resource_one_hot(args: Any) -> pd.DataFrame:
    (dict_name, code_dict, df) = args
    # Fill all columns with -1 -> not occuring
    zero_list = [-1] * len(code_dict[dict_name].keys())
    df_res = pd.DataFrame(zero_list).transpose()
    df_res.columns = code_dict[dict_name].keys()

    # Find the occuering codes
    if dict_name == "CONDITION_DICT_LIST":
        scope = "icd_code"
    elif dict_name == "PROCEDURE_DICT_LIST":
        scope = "ops_code"
    elif dict_name == "MEDICATION_DICT_LIST":
        scope = "substance"
    else:
        scope = ["icd_code", "ops_code", "substance"]

    try:
        if not pd.Series(scope).isin(df.columns).all():
            return df_res

        for scope_element in scope:
            if not df[scope_element].any():
                continue
            oc_resources = np.hstack(df[scope_element].dropna().values)
            oc_resources = [x for x in oc_resources if x != 1024]
            for code in oc_resources:
                columns = [
                    group_dict_name
                    for group_dict_name, group_list in code_dict[dict_name].items()
                    for list_code in group_list
                    if list_code in code
                ]
                for group in columns:
                    df_res[group] = 1
    except Exception as e:
        logging.info(e.message, e.args)

        # df_merged = pd.concat([df, df_res], ignore_index=True)
    return df_res


def calculate_age(born: datetime, start_date: datetime) -> int:
    return int(
        start_date.year
        - born.year
        - ((start_date.month, start_date.day) < (born.month, born.day))
    )


def encode_cyclical_time(data: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    data[col + "_sin"] = np.round(
        np.sin(2 * np.pi * data[col].astype(float) / max_val), 1
    )
    data[col + "_cos"] = np.round(
        np.cos(2 * np.pi * data[col].astype(float) / max_val), 3
    )
    return data


def preprocess_dfs(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if not pd.Series(["weekday"]).isin(df.columns).any():
            return df

        # exlude masked values
        if (df.value_zthra == 1024).any():
            for index, row in df.iterrows():
                if row.value_zthra != 1024:
                    break

            mask = df.iloc[0:index]
            df = df.iloc[index:]

            df = encode_cyclical_time(df, "weekday", 6)

            # concat with masked values
            df = pd.concat([mask, df], axis=0)
            df.fillna(1024, inplace=True)
        else:
            df = encode_cyclical_time(df, "weekday", 6)

        df.drop(
            columns=[
                "weekday",
                "substance",
                "ops_code",
                "icd_code",
                "output_to",
                "einscode_from_dep",
            ],
            inplace=True,
        )
        if df.value_zthra.isna().any() and df.value_zthra.sum() > 0:
            df.value_zthra.fillna(method="ffill", inplace=True)
        elif df.value_zthra.isna().any():
            df.value_zthra.fillna(value=-1, inplace=True)
        df.value_tkz.fillna(value=0, inplace=True)

        # cast to int
        df["value_zthra"] = df["value_zthra"].astype(int)
        df["value_tkz"] = df["value_tkz"].astype(int)

        # turn 2d df sample into 1d flat df and add to train df
        df.reset_index(drop=True, inplace=True)
        df.index = df.index.astype(str)
        return df  # pd.DataFrame(flat_df).T
    except Exception as e:
        logging.info(e.args)


def transform_dict_constants():
    with (pathlib.Path(__file__).parent.parent / "config" / "constants.yaml").open(
        "r"
    ) as stream:
        code_dict = yaml.safe_load(stream)

    # adding medication key to values
    for key, value in code_dict["MEDICATION_DICT_LIST"].items():
        code_dict["MEDICATION_DICT_LIST"][key].append(key)

    # translating categorry_dict_list into ICD, OPS and medication names
    dict_combo = (
        code_dict["CONDITION_DICT_LIST"]
        | code_dict["PROCEDURE_DICT_LIST"]
        | code_dict["MEDICATION_DICT_LIST"]
    )
    for key, value in code_dict["CATEGORY_DICT_LIST"].items():
        codes_in_class = list()
        for item in value:
            codes_in_class.append(dict_combo[item])
        codes_in_class = [item for sublist in codes_in_class for item in sublist]
        code_dict["CATEGORY_DICT_LIST"][key] = codes_in_class
    return code_dict


class AutopilotDataset(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for patch_dataset dataset.
    """

    VERSION = tfds.core.Version("0.2.1")
    RELEASE_NOTES = {
        "0.1.3": "all clinic",
        "0.2.0": "hematooncology",
        "0.2.1": "hematooncology obs limit 50",
        "0.3.1": "heart-thorax fixed",
    }

    def __init__(self, config, **kwargs):
        self.VERSION = tfds.core.Version(config["dataset_name"])
        tfds.core.GeneratorBasedBuilder.__init__(self, **kwargs)
        self.code_dict = transform_dict_constants()
        self.config = config

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="increasing ds",
            features=tfds.features.FeaturesDict(
                {
                    "sq_features": tfds.features.Tensor(
                        shape=(60, 7), dtype=tf.float32
                    ),
                    "sq_features_mask": tfds.features.Sequence(tf.bool, length=60),
                    "mt_features": tfds.features.Sequence(tf.float32, length=16),
                    "station_features": tfds.features.Sequence(tf.uint8, length=2),
                    "patient_id": tfds.features.Text(),
                    "start_date": tfds.features.Text(),
                    "clinic": tfds.features.Text(),
                    "stationed_at": tfds.features.Text(),
                    "label": tfds.features.ClassLabel(num_classes=2),
                }
            ),
            supervised_keys=(("sq_features", "mt_features"), "label"),
        )

    def _split_generators(self, _: Any) -> List[tfds.core.SplitGenerator]:
        pat_dict = pd.read_pickle(self.config["preprocessed_windows_path"])
        # pat_dict = dict(itertools.islice(pat_dict.items(), int(len(pat_dict)*0.5)))
        for pat_key, pat_value in pat_dict.items():
            gender = pat_value[0]["gender"]
            birth_date = datetime.datetime.fromisoformat(pat_value[0]["birth_date"])
            # the first entry is skipped as it contains only metadata
            for dict_index, item_pat_value in enumerate(pat_value[1:]):
                start_date = datetime.datetime.fromisoformat(
                    item_pat_value["start_date"]
                )
                pat_value[dict_index].update(
                    {
                        "gender": gender,
                        "age": calculate_age(birth_date, start_date),
                        "patient_id": pat_key,
                    }
                )

        shuffled_pats = list(pat_dict.keys())

        if self.config["is_live_prediction"]:
            return [
                tfds.core.SplitGenerator(
                    name="live",
                    gen_kwargs={
                        "patient_dict": {
                            pat_id: pat_dict[pat_id] for pat_id in shuffled_pats
                        }
                    },
                )
            ]

        else:
            # subject based
            total_count = len(pat_dict)
            train_count = int(total_count * self.config["train_size"])

            # shuffle patients
            random.Random(9).shuffle(shuffled_pats)

            folds = np.array_split(shuffled_pats[:train_count], 5)
            assert all(
                p not in [other_p for fold in folds for other_p in fold]
                for p in shuffled_pats[train_count:]
            )
            return [
                tfds.core.SplitGenerator(
                    name=f"cv_{fold_id + 1}_of_{5}",
                    gen_kwargs={
                        "patient_dict": {pat_id: pat_dict[pat_id] for pat_id in fold},
                    },
                )
                for fold_id, fold in enumerate(folds)
            ] + [
                tfds.core.SplitGenerator(
                    name="test",
                    gen_kwargs={
                        "patient_dict": {
                            pat_id: pat_dict[pat_id]
                            for pat_id in shuffled_pats[train_count:]
                        }
                    },
                )
            ]

    def _generate_examples(self, patient_dict) -> Generator:
        for patient_id, patient_list in patient_dict.items():
            # assert patient_list[0]["gender"] in {"male", "female"}
            gender = -1 if patient_list[0]["gender"] == "male" else 1
            age = patient_list[0]["age"]
            # the first entry is skipped as it contains only metadata
            for i, pat_dict in enumerate(patient_list[1:]):
                # One hot encoding categories
                args = ["CATEGORY_DICT_LIST", self.code_dict, pat_dict["pat_df"]]
                cat_df = get_resource_one_hot(args)

                # # One hot encoding station by a spesific clinic
                pat_dict["clinic"] = find_clinic(pat_dict["stationed_at"])
                # [18, 6] = 'other' , 'other
                pat_dict["clinic"] = (
                    pat_dict["clinic"] if pat_dict["clinic"] else "other"
                )

                # filter for only heart_thorax or heamatooncology samples
                if self.config["clinic_filter"] != "None":
                    if pat_dict["clinic"] != self.config["clinic_filter"]:
                        continue
                    stations_by_clinic = get_stations_int_encoding(
                        pat_dict["stationed_at"],
                        pat_dict["clinic"],
                        self.code_dict["CLINIC_BY_STATION"],
                    )[0]
                    clinic_station_index = np.array([-1, stations_by_clinic])
                # exclude heart thorax and heamatooncology samples
                else:
                    if (
                        pat_dict["clinic"] == "heart_thorax"
                        or pat_dict["clinic"] == "heamatooncology"
                    ):
                        continue

                    station_index_dict = {}
                    for j, station in enumerate(
                        [
                            item
                            for sublist in list(
                                (self.code_dict["CLINIC_BY_STATION"].values())
                            )
                            for item in sublist
                        ]
                    ):
                        station_index_dict[station] = j

                    clinic_index_dict = {}
                    for x, clinic in enumerate(
                        self.code_dict["CLINIC_BY_STATION"].keys()
                    ):
                        clinic_index_dict[clinic] = x

                    clinic_station_index = get_clinic_station_index(
                        pat_dict["stationed_at"],
                        pat_dict["clinic"],
                        station_index_dict,
                        clinic_index_dict,
                    )

                # print(
                #     f"found clinic: {pat_dict['clinic']}, stationed at: {pat_dict['stationed_at']} ,station_id: {clinic_station_index}"
                # )
                assert (
                    len(pat_dict["clinic"]) != 1
                ), f"invalid size clinic: {pat_dict['clinic']}"
                assert len(pat_dict["stationed_at"]) != 1, "invalid size station"
                assert isinstance(pat_dict["stationed_at"], str)
                assert isinstance(pat_dict["clinic"], str)

                # adding gender and age
                cat = cat_df.iloc[0].tolist()
                cat += [gender, age]
                seq_df = preprocess_dfs(pat_dict["pat_df"])

                if seq_df is None:
                    continue

                yield f"{patient_id}_{i}", {
                    "sq_features": seq_df,
                    "sq_features_mask": pat_dict["pat_df_mask"],
                    "mt_features": cat,
                    "station_features": clinic_station_index,
                    "patient_id": patient_id,
                    "start_date": pat_dict["start_date"],
                    "clinic": pat_dict["clinic"],
                    "stationed_at": pat_dict["stationed_at"],
                    "label": pat_dict["label"][0],
                }
