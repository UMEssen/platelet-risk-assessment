import logging
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, List, Optional

import holidays as holidays
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

with open("app/config/constants.yaml", "r") as stream:
    code_dict = yaml.safe_load(stream)


def col_to_datetime(date_series: pd.Series) -> pd.Series:
    # some procedure dates are just YYYY-MM-DD and will result in nan values => I don't fucking care right now
    if date_series.any():
        date_series = pd.to_datetime(
            date_series, format="%Y-%m-%dT%H:%M:%S.%f%z", utc=True, errors="coerce"
        ).dt.tz_convert("CET")

        if date_series is pd.NaT:
            date_series = pd.to_datetime(
                date_series, format="%Y-%m-%d %H:%M:%S.%f%z", utc=True, errors="coerce"
            ).dt.tz_convert("CET")

        date_series = date_series.dt.tz_localize(None)
    return date_series


def get_stations(
    slide_df: pd.DataFrame,
) -> tuple:
    stations_bdp = slide_df["output_to"].reset_index(drop=True).dropna()
    stations_obs = slide_df["einscode_from_dep"].reset_index(drop=True).dropna()
    unique_stations = (
        pd.concat([stations_bdp, stations_obs], ignore_index=True)
        .drop_duplicates()
        .values
    )

    # where was the final location of a patient
    def get_max_station_index(df_column: pd.Series) -> tuple[list, Optional[str]]:
        if pd.isna(df_column).all():
            return 0
        return np.nonzero(df_column)[0][-1] if np.nonzero(df_column)[0].any() else 0

    slide_df = slide_df.where(pd.notnull(slide_df), None)
    station_dict = {
        "output_to": get_max_station_index(slide_df.output_to.values),
        "einscode_from_dep": get_max_station_index(slide_df.einscode_from_dep.values),
    }

    col = max(station_dict, key=station_dict.get)
    max_station = slide_df[col].iloc[station_dict[col]]
    return unique_stations, max_station


def find_clinic(station: str) -> str:
    for clinic, stations in code_dict["CLINIC_BY_STATION"].items():
        if station in stations:
            return clinic


def get_clinic_from_stations(stations: pd.Series, first_only=True) -> Any:
    columns = []
    if stations is np.nan or isinstance(stations, float):
        return ["other"]

    for station in stations:
        columns.append(
            [
                group_dict_name
                for group_dict_name, group_list in code_dict[
                    "CLINIC_BY_STATION"
                ].items()
                for list_code in group_list
                if list_code == station
            ]
        )
    columns = [x if x.__len__() != 0 else ["other"] for x in columns]
    if first_only:
        return columns[0][0]
    return columns


def make_store_global(store: Any) -> None:
    global global_store
    global_store = store


def list_without_nans(series: pd.Series) -> Optional[List[str]]:
    ll = set()
    for val in series.values:
        if not pd.isnull(val):
            ll.add(val)
    if not len(ll):
        return None
    return list(ll)


@dataclass
class DataStore:
    bdp: Optional[pd.DataFrame]
    obs: Optional[pd.DataFrame]
    pro: Optional[pd.DataFrame]
    enc: pd.DataFrame
    con: Optional[pd.DataFrame]
    med: Optional[pd.DataFrame]
    pat: pd.DataFrame

    def select_resources(
        self, resource_df: pd.DataFrame, column: str, patient_id: str, filter_date
    ):
        if len(resource_df) == 0:
            return resource_df

        return resource_df.loc[
            (resource_df.patient_id == patient_id)
            & (col_to_datetime(resource_df[column]) >= filter_date)
        ]

    def filter_patient(self, patient_id: str, filter_date=None):
        if filter_date:
            date = pd.to_datetime(filter_date)

            return DataStore(
                self.select_resources(
                    resource_df=self.bdp,
                    column="date",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.obs,
                    column="date",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.pro,
                    column="procedure_start",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.enc,
                    column="encounter_start",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.con,
                    column="condition_date",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.med,
                    column="event_time",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.pat[self.pat.patient_id == patient_id],
            )

        return DataStore(
            self.bdp[self.bdp.patient_id == patient_id],
            self.obs[self.obs.patient_id == patient_id],
            self.pro[self.pro.patient_id == patient_id],
            self.enc[self.enc.patient_id == patient_id].sort_values(
                ["start_date"], inplace=False
            ),
            self.con[self.con.patient_id == patient_id],
            self.med[self.med.patient_id == patient_id],
            self.pat[self.pat.patient_id == patient_id],
        )


def get_pat_by_encounter_helper(
    substore: DataStore, enc_start: str, enc_end: str
) -> pd.DataFrame:
    # patient is dead
    if substore.pat.deceasedDateTime.any():
        if len(substore.pat[(substore.pat.deceasedDateTime <= enc_end)]):
            # patient died during encounter but encounter lasts longer
            if len(
                substore.obs[
                    (substore.obs.date >= enc_start) & (substore.obs.date <= enc_end)
                ]
            ):
                enc_end = substore.pat.deceasedDateTime.values[0]
            else:
                return pd.DataFrame()
    # get observations for encounter cell
    obs_scope = substore.obs[
        (substore.obs.date >= enc_start) & (substore.obs.date <= enc_end)
    ]
    obs_scope["is_real_zthra_value"] = 1

    # if we don't have two observations in the encounter there is nothing I can predict
    if len(obs_scope) >= 2:
        if min(obs_scope.value_zthra) >= 50:
            return pd.DataFrame()
        bdp_scope = substore.bdp[
            (substore.bdp.date >= enc_start) & (substore.bdp.date <= enc_end)
        ]
        pro_scope = substore.pro[
            (substore.pro.start_date >= enc_start)
            & (substore.pro.start_date <= enc_end)
        ]
        med_scope = substore.med[
            (substore.med.event_time >= enc_start)
            & (substore.med.event_time <= enc_end)
        ]
        con_scope = substore.con[
            (substore.con.date >= enc_start) & (substore.con.date <= enc_end)
        ]
        # 1. combine resources , interpolate obs, resample
        # 2. creating sliding windows starting from enc_start + (7d)
        pat_hist = obs_scope
        if len(bdp_scope):
            bdp_scope["value_tkz"] = 1
            pat_hist = pd.concat([pat_hist, bdp_scope], join="outer")
        if len(pro_scope):  # good
            pro_scope.rename(columns={"start_date": "date"}, inplace=True)
            pat_hist = pd.concat([pat_hist, pro_scope], join="outer")
        if len(med_scope):  # good
            med_scope.rename(columns={"event_time": "date"}, inplace=True)
            pat_hist = pd.concat([pat_hist, med_scope], join="outer")
        if len(con_scope):
            pat_hist = pd.concat([pat_hist, con_scope], join="outer")
        # in put frames for lower / upper time limit
        data = {"date": [enc_start, enc_end]}
        pat_hist = pd.merge(pat_hist, pd.DataFrame(data), on="date", how="outer")
        agg_rules = {
            "value_zthra": pd.NamedAgg(column="value_zthra", aggfunc="min"),
            "value_tkz": pd.NamedAgg(column="value_tkz", aggfunc="sum"),
            "ops_code": pd.NamedAgg(column="code", aggfunc=list_without_nans),
            "output_to": pd.NamedAgg(column="output_to_einscode", aggfunc="last"),
            "einscode_from_dep": pd.NamedAgg(
                column="einscode_from_dep", aggfunc="last"
            ),
            "substance": pd.NamedAgg(
                column="medicationName", aggfunc=list_without_nans
            ),
            "is_real_zthra_value": pd.NamedAgg(
                column="is_real_zthra_value", aggfunc="last"
            ),
            "icd_code": pd.NamedAgg(column="icd_code", aggfunc=list_without_nans),
        }
        df = (
            pat_hist.set_index("date", drop=True)
            .resample("12H")
            .agg(**{k: v for k, v in agg_rules.items() if v.column in pat_hist.columns})
            .reindex(agg_rules.keys(), axis=1)
        )

        df.value_tkz.fillna(0, inplace=True)
        # add column weekday 0 -> Monday, 6 -> Sunday
        df["weekday"] = df.index.dayofweek
        # add column is_weekend
        df["is_weekend"] = df.weekday.apply(lambda x: 1 if x > 4 else -1)
        # add column holiday
        de_NW_holidays = holidays.DE(prov="NW")
        time = pd.DataFrame()
        time["timenoindex"] = df.index.to_pydatetime()
        df["is_holiday"] = [
            1 if val in de_NW_holidays else -1 for val in time.timenoindex
        ]
        df.value_zthra.interpolate("linear", inplace=True)
        df.value_zthra.fillna(method="bfill", inplace=True)
        df.value_zthra.fillna(method="ffill", inplace=True)
        df.value_zthra = df.value_zthra.astype(int)
        df.value_tkz = df.value_tkz.astype(int)
        df.is_real_zthra_value.fillna(-1, inplace=True)
        df.is_real_zthra_value = df.is_real_zthra_value.astype(int)
        return df
    else:
        return pd.DataFrame()


class BuildAggEncounters:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config
        self.store = self.get_source_data()

    def get_source_data(self) -> DataStore:
        # BiologicallyDerivedProduct
        bdp = pd.read_feather(Path(self.config["bdp_path_filtered"]))
        obs = pd.read_feather(Path(self.config["obs_path_filtered"]))
        # Procedure
        pro = pd.read_feather(Path(self.config["procedure_path_filtered"]))
        # Encounter
        enc = pd.read_feather(Path(self.config["encounter_path_filtered"]))
        # Conditions
        con = pd.read_feather(Path(self.config["condition_path_filtered"]))
        # Patients
        pats = pd.read_feather(Path(self.config["patient_path_filtered"]))
        # pat_index_rand = [randint(0, len(pats) - 1) for p in range(0, 20000)]
        # pats = pats.iloc[pat_index_rand]
        # Medciation
        med = pd.read_feather(Path(self.config["medication_merged_path_filtered"]))

        # transform to datetime
        bdp["date"] = col_to_datetime(bdp["date"])
        obs["date"] = col_to_datetime(obs["date"])
        pro["start_date"] = col_to_datetime(pro["procedure_start"])
        pro["end_date"] = col_to_datetime(pro["procedure_end"])
        enc["start_date"] = col_to_datetime(enc["encounter_start"])
        enc["end_date"] = col_to_datetime(enc["encounter_end"])
        pats["deceasedDateTime"] = col_to_datetime(pats["deceasedDateTime"])
        med["event_time"] = col_to_datetime(med["event_time"])
        con["date"] = col_to_datetime(con["condition_date"])

        enc.drop(
            columns=[
                "next_start",
                "overlap",
                "encounter_type",
                "status",
                "all_encounters",
            ],
            inplace=True,
        )

        obs.rename(columns={"value": "value_zthra"}, inplace=True)

        # Filter time scope
        # enc.index = enc.start_date
        # enc = enc.loc["2017-01-01":"2019-12-31"]
        # enc = enc.reset_index(drop=True).sort_values(by="start_date")

        return DataStore(bdp, obs, pro, enc, con, med, pats)

    def get_pat_by_encounter(self) -> dict:
        pat_dict = {}

        args = [
            (patient, self.config["is_live_prediction"])
            for patient in self.store.pat.patient_id
        ]

        with mp.Pool(64, initializer=make_store_global, initargs=(self.store,)) as pool:
            results_iter = tqdm(
                pool.imap_unordered(
                    self.get_pat_by_encounter_worker,
                    args,
                ),
                total=len(self.store.pat),
            )
            for item in results_iter:
                if item is None:
                    continue
                for pat_id, list_of_dfs in item.items():
                    for value in list_of_dfs:
                        if pat_id in pat_dict:
                            pat_dict[pat_id].append(value)
                        else:
                            substore = DataStore.filter_patient(
                                self.store, patient_id=pat_id
                            )
                            pat_dict[pat_id] = [
                                {
                                    "gender": str(substore.pat.gender.values[0]),
                                    "birth_date": str(
                                        substore.pat.birth_date.values[0]
                                    ),
                                }
                            ]
                            pat_dict[pat_id].append(value)

        with open(self.config["preprocessed_by_encounter_path"], "wb+") as file:
            pickle.dump(pat_dict, file)
        return pat_dict

    @staticmethod
    def get_pat_by_encounter_worker(args: Any) -> Optional[dict]:
        # Unpack globals
        store = global_store
        dfs = list()
        df = pd.DataFrame()
        (pat, is_live_prediction) = args
        substore = DataStore.filter_patient(store, patient_id=pat)
        # patient needs valid encounter + has at least an observation
        if len(substore.enc) and len(substore.obs):
            for _, enc_cell in substore.enc.iterrows():
                enc_start = enc_cell.start_date
                enc_end = enc_cell.end_date
                if is_live_prediction:
                    today = datetime.today().date()
                    max_time = time.max
                    max_datetime_today = datetime.combine(today, max_time)

                    # future encounters are set to the current time
                    enc_end = (
                        datetime.now() if enc_end > max_datetime_today else enc_end
                    )

                    # encounter end must be today or there shall not be a prediction
                    if enc_end.date() != today:
                        continue

                    # mapping encounter end to 12:00 or 24:00 to have clear predictions cut-offs
                    enc_end = (
                        max_datetime_today
                        if enc_end.hour > 12
                        else pd.to_datetime(enc_end.date()) + pd.Timedelta(hours=12)
                    )

                    df = get_pat_by_encounter_helper(substore, enc_start, enc_end)
                else:
                    enc_end = (
                        pd.Timestamp("2022-01-01 00:00:00")
                        if enc_end > pd.to_datetime("2022-01-01 00:00:00")
                        else enc_end
                    )
                    # inclusion criteria: Patient needs to be at least two days in the hospital
                    if enc_end - enc_start >= pd.Timedelta(2, "d"):
                        df = get_pat_by_encounter_helper(substore, enc_start, enc_end)
                if len(df):
                    dfs.append(df)
        ret = {pat: dfs} if len(dfs) else None
        return ret


class BuildAggSlidingWindows:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config

    def generate_windows_from_encounter_dfs(
        self, enc_pat_dfs: pd.DataFrame
    ) -> Optional[dict]:
        pat_dict = {}

        args = [
            (pat, data, self.config["is_live_prediction"])
            for (pat, data) in enc_pat_dfs.items()
        ]

        with mp.Pool(80) as pool:
            results_iter = tqdm(
                pool.imap_unordered(
                    # self.generate_sliding_windows_from_encounter_dfs_worker,
                    self.generate_growing_windows_from_encounter_dfs_worker,
                    args,
                ),
                total=len(enc_pat_dfs),
            )

            for el in results_iter:
                for pat, data in el.items():
                    pat_dict[pat] = data

        with open(
            self.config["preprocessed_windows_path"],
            "wb+",
        ) as file:
            pickle.dump(pat_dict, file)
        return pat_dict

    @staticmethod
    def generate_growing_windows_from_encounter_dfs_worker(
        args: Any,
    ) -> Optional[dict]:
        (pat, data, is_live_prediction) = args
        pat_dict = dict()

        for idx, df in enumerate(data):
            if idx == 0:
                continue

            enc_start = df.index[0]
            enc_end = df.index[-1]

            if is_live_prediction:
                # for live predictions only the last day is relevant
                date_range = pd.date_range(
                    start=enc_end,
                    end=enc_end,
                    freq="1D",
                )
            else:
                date_range = pd.date_range(
                    start=enc_start + pd.Timedelta(1, "d"),
                    end=enc_end - pd.Timedelta(1, "d"),
                    freq="1D",
                )

            # extact by encounter as some sliding windows do not have station information
            unique_stations_enc, max_station_enc = get_stations(df)

            # d is the day of prediction if d.hour == 0
            for d in date_range:
                if is_live_prediction:
                    d += timedelta(days=1)
                    train_slice = df.iloc[-60:]
                    labels = [0]
                else:
                    train_slice = df[(df.index <= d - pd.Timedelta(0.1, "d"))]

                    # Take next 24 hours into account
                    label_slice = df[
                        ((d <= df.index) & (df.index < d + pd.Timedelta(1, "d")))
                    ].value_tkz

                    labels = [sum(label_slice)]

                    assert (
                        train_slice.index[0].hour == d.hour
                    ), "Hours don't match -> fix sliding function"

                unique_stations, last_station = get_stations(train_slice)
                last_station = (
                    max_station_enc if pd.isna(last_station) else last_station
                )
                clinic = get_clinic_from_stations(last_station)

                # if there was no observation or min zTHRA is not below reference value sliding window is not relevant
                if (
                    len(train_slice[train_slice.is_real_zthra_value == 1]) == 0
                    or min(train_slice.value_zthra) >= 150
                ):
                    continue

                # padding, masking:
                # 1. cut to 60 cells for more than 60 data points
                # 2. time-series-data which has less than 60 data points
                if len(train_slice) > 60:
                    train_slice = train_slice.iloc[-60:]
                    mask = np.full(train_slice.shape[0], True)
                else:
                    len_df = len(train_slice)
                    empty_df = pd.DataFrame(
                        columns=train_slice.columns, index=range(60 - len_df)
                    )
                    empty_df.fillna(value=1024, inplace=True)
                    # mask which will later be applied to the model, int represents number of rows
                    true_mask = np.full(train_slice.shape[0], True)
                    false_mask = np.full(empty_df.shape[0], False)
                    mask = np.concatenate((false_mask, true_mask))
                    train_slice = pd.concat([empty_df, train_slice]).reset_index(
                        drop=True
                    )
                assert train_slice.shape == (60, 11)

                # todo challenge for live predictions
                # final filter: 48 h no real observation and last got no bdp
                if (
                    not (train_slice.is_real_zthra_value[-4:] == 1).any()
                    and labels[0] == 0
                ):
                    continue

                assert clinic
                pat_sliding_dict = {
                    "pat_df": train_slice,
                    "pat_df_mask": mask,
                    "label": [x if x <= 1 else 1 for x in labels],
                    "start_date": str(d),
                    "clinic": str(clinic),
                    "stationed_at": str(last_station),
                }
                if pat in pat_dict:
                    pat_dict[pat].append(pat_sliding_dict)
                else:
                    pat_dict[pat] = [
                        {
                            "gender": str(data[0]["gender"]),
                            "birth_date": data[0]["birth_date"],
                        }
                    ]
                    pat_dict[pat].append(pat_sliding_dict)
            return pat_dict
        return None

    @staticmethod
    def generate_sliding_windows_from_encounter_dfs_worker(
        args: Any,
    ) -> Optional[dict]:
        (pat, data, is_live_prediction) = args
        pat_dict = dict()

        for idx, df in enumerate(data):
            if idx == 0:
                continue

            enc_start = df.index[0]
            enc_end = df.index[-1]

            if is_live_prediction:
                # for live predictions only the last day is relevant
                date_range = pd.date_range(
                    start=enc_end,
                    end=enc_end,
                    freq="1D",
                )
            else:
                date_range = pd.date_range(
                    start=enc_start + pd.Timedelta(7, "d"),
                    end=enc_end - pd.Timedelta(3, "d"),
                    freq="1D",
                )

            # extact by encounter as some sliding windows do not have station information
            unique_stations_enc, max_station_enc = get_stations(df)

            # d is the day of prediction if d.hour == 0
            for d in date_range:
                if is_live_prediction:
                    d += timedelta(days=1)
                    train_slice = df.iloc[-60:]
                    labels = [0]
                else:
                    train_slice = df[
                        ((d - pd.Timedelta(7.1, "d")) <= df.index)
                        & (df.index <= d - pd.Timedelta(0.1, "d"))
                    ]
                    if d.hour == 0:
                        label_slice = df[
                            (d - pd.Timedelta(0.5, "d") < df.index)
                            & (df.index < d + pd.Timedelta(3, "d"))
                        ].value_tkz

                    else:
                        label_slice = df[
                            (d < df.index) & (df.index < d + pd.Timedelta(3.5, "d"))
                        ].value_tkz

                    label_slice = pd.DataFrame(label_slice)
                    agg_rules = {
                        "labels": pd.NamedAgg(column="value_tkz", aggfunc="sum"),
                    }
                    labels = (
                        (
                            label_slice.resample("24H")
                            .agg(
                                **{
                                    k: v
                                    for k, v in agg_rules.items()
                                    if v.column in df.columns
                                }
                            )
                            .reindex(agg_rules.keys(), axis=1)
                        )
                        .stack()
                        .values.tolist()
                    )

                    assert (
                        train_slice.index[0].hour == d.hour
                    ), "Hours don't match -> fix sliding function"

                assert train_slice.shape[0] == 60, f"invalid shape"
                unique_stations, last_station = get_stations(train_slice)
                last_station = (
                    max_station_enc if pd.isna(last_station) else last_station
                )
                clinic = get_clinic_from_stations(last_station)

                # if there was no observation or min zTHRA is not below reference value sliding window is not relevant
                if (
                    len(train_slice[train_slice.is_real_zthra_value == 1]) == 0
                    or min(train_slice.value_zthra) >= 150
                ):
                    continue

                # in case of mid-day: day of prediction is the next day
                if d.hour == 12:
                    d += timedelta(hours=12)

                pat_sliding_dict = {
                    "pat_df": train_slice,
                    "label": labels,
                    "start_date": str(d),
                    "clinic": str(clinic),
                    "stationed_at": str(last_station),
                }
                if pat in pat_dict:
                    pat_dict[pat].append(pat_sliding_dict)
                else:
                    pat_dict[pat] = [
                        {
                            "gender": str(data[0]["gender"]),
                            "birth_date": data[0]["birth_date"],
                        }
                    ]
                    pat_dict[pat].append(pat_sliding_dict)
            return pat_dict
        return None

    def bdp_window_validator(self) -> None:
        bdp = pd.read_feather(self.config["bdp_path_filtered"])
        windows = pd.read_pickle(self.config["preprocessed_windows_path"])

        labels = []
        start_dates = []
        pat_ids = []
        for pat_id, win_dict_list in windows.items():
            i = 0
            for dict in win_dict_list:
                if i == 0:
                    i += 1
                    continue

                pat_ids.append(pat_id)
                labels.append(dict["label"][0])
                start_dates.append(
                    datetime.strptime(dict["start_date"], "%Y-%m-%d %H:%M:%S").date()
                )

        wins_df = pd.DataFrame(data=[pat_ids, labels, start_dates]).transpose()
        wins_df.columns = ["pat_id", "label", "start_date"]

        # dates = ['18.01.2018', '26.06.2019', '30.09.2021', '09.09.2019', '02.02.2020']
        date_range = pd.date_range(
            start=pd.Timestamp("2020-01-28"),
            end=pd.Timestamp("2020-03-28"),
            freq="1D",
        )
        bdp["date_date"] = pd.Series(
            [x.strftime("%Y-%m-%d 00:00:00") for x in bdp.date]
        )

        for date in date_range:
            # Filter BDP to a relevant time
            bdp_filtered = bdp[bdp.date_date == str(date)]
            bdp_raw_x_day = bdp_filtered.count()[0]

            logging.info(f"\n{date}")
            logging.info(f"Raw bdp count: {bdp_raw_x_day}")

            # Filter window raw_data to / start_date = Date of prediciton
            wins_df_filtered = wins_df[wins_df.start_date == date]

            logging.info(f"Window bdp count: {wins_df_filtered.count()[0]}")

        logging.info(
            pd.concat(
                [bdp_filtered, bdp_filtered.patient_id.isin(wins_df_filtered.pat_id)],
                axis=1,
            )
        )


def main(config) -> None:
    if not Path(config["preprocessed_by_encounter_path"]).exists():
        enc_builder = BuildAggEncounters(config)
        enc_pat_dfs = enc_builder.get_pat_by_encounter()
    else:
        enc_pat_dfs = pd.read_pickle(config["preprocessed_by_encounter_path"])

    # Generate a dict with pat vs encounter windows
    # enc_pat_dfs = dict(itertools.islice(enc_pat_dfs.items(), 50))
    win_builder = BuildAggSlidingWindows(config)
    if not Path(config["preprocessed_windows_path"]).exists():
        win_builder.generate_windows_from_encounter_dfs(enc_pat_dfs)
    else:
        win_builder.bdp_window_validator()


if __name__ == "__main__":
    main()
