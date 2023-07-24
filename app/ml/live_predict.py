import datetime as dt
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Union, Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import tensorflow as tf
from fhir.resources import riskassessment, meta
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.period import Period
from fhir.resources.reference import Reference
from requests.adapters import Retry, PoolManager

from app.data_handling.create_sliding_window import DataStore
from app.data_handling.get_training_data import AutopilotDataset
from app.ml.train_helper import get_dataset

# REQUIRED PATHS
SEARCH_URL = os.environ["SEARCH_URL"]
BASIC_AUTH = os.environ["BASIC_AUTH"]
REFRESH_AUTH = os.environ["REFRESH_AUTH"]
FHIR_USER = os.environ["FHIR_USER"]
FHIR_PASSWORD = os.environ["FHIR_PASSWORD"]


def custom_json_serializer(obj: object) -> Union[float, str]:
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="milliseconds")
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return round(float(obj), 3)
    raise TypeError(f"Type {type(obj)} not JSON serializable")


def get_time_zone_format(time: datetime) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S+00:00")


def fhir_login() -> str:
    """
    FHIR login to get token.
    Reads USERNAME and PASSWORD from .env file
    Parameters
    ----------
    Returns
    ----------
    token as string
    """
    client = requests.Session()
    response = client.post(
        BASIC_AUTH,
        auth=(FHIR_USER, FHIR_PASSWORD),
    )
    query_args = parse_qs(urlparse(response.url).query)
    if "result" in query_args and query_args["result"][0] == "loginFailed":
        print("Invalid login!")
    if "503 Service Unavailable" in response.text:
        print("503 Service Unavailable")
    if "internal server error" in response.text:
        print("There was an internal server error.")
    return response.text


class TkPredict:
    def __init__(self, config, token):
        self.token = token
        self.config = config

    def get_predictions(self) -> Any:
        print(self.config["model_path"] + self.config["clinic_filter"])

        model = tf.keras.models.load_model(
            self.config["model_path"] + self.config["clinic_filter"]
        )

        try:
            ds = AutopilotDataset(
                data_dir=self.config["dataset_dir"], config=self.config
            )
            ds.download_and_prepare()
        except Exception as e:
            logging.info(e)
            return [], []

        live_split = get_dataset(
            "live",
            self.config,
        )
        live_data = get_dataset(
            "live",
            self.config,
            do_split_label=False,
        )

        if len(live_split) == 0:
            return [], []

        pred = model.predict(live_split)
        return pred, live_data

    def push_predictions(self, predictions, live_data):
        pat_ids = []
        for sample in zip(live_data):
            pat_ids.append(pd.Series(sample[0]["patient_id"]))

        flat_list = [item for sublist in pat_ids for item in sublist]
        pat_ids = [x.decode("utf-8") for x in flat_list]

        pat_predicts = pd.DataFrame(predictions)
        pat_predicts.index = pat_ids
        pat_predicts.reset_index(inplace=True)
        pat_predicts.columns = ["pat_id", "day1"]

        bdp = pd.read_feather(Path(self.config["bdp_path_filtered"]))
        obs = pd.read_feather(Path(self.config["obs_path_filtered"]))
        pro = pd.read_feather(Path(self.config["procedure_path_filtered"]))
        enc = pd.read_feather(Path(self.config["encounter_path_filtered"]))
        con = pd.read_feather(Path(self.config["condition_path_filtered"]))
        pats = pd.read_feather(Path(self.config["patient_path_filtered"]))
        med = pd.read_feather(Path(self.config["medication_merged_path_filtered"]))

        meta_rq = meta.Meta(
            source="urn:ship-app:autopilot",
            profile=["https://uk-essen.de/fhir/StructureDefinition/autopilot"],
            extension=[
                {
                    "url": "http://uk-essen.de/fhir/extension-source-version",
                    "valueString": "0.0.1",
                },
            ],
        )

        store = DataStore(bdp, obs, pro, enc, con, med, pats)
        dtnow = (
            datetime.now(tz=timezone.utc)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S+00:00")
        )
        dtnow_date = datetime.utcnow().date()

        for index, pat in pat_predicts.iterrows():
            pat_store = store.filter_patient(
                patient_id=pat.pat_id,
                filter_date=str(
                    dtnow_date - pd.DateOffset(days=self.config["time_delta"])
                ),
            )
            model_name = (
                "department_none"
                if self.config["clinic_filter"] == "None"
                else self.config["clinic_filter"]
            )
            code_name = "24_hour_forecast"
            rsk_id = hashlib.sha256(
                "".join(
                    [
                        "autopilot",
                        pat.pat_id,
                        model_name,
                        code_name,
                        str(dt.datetime.now()),
                    ]
                ).encode()
            ).hexdigest()

            print(f"rsk id {rsk_id}")
            rsk_pat = riskassessment.RiskAssessment(
                id=rsk_id,
                status="final",
                meta=meta_rq,
                subject=Reference(reference=f"Patient/{pat.pat_id}"),
            )

            bdps = [
                Reference(
                    reference=f"BiologicallyDerivedProduct/{x}",
                )
                for x in pat_store.bdp.resource_id
            ]
            conds = [
                Reference(reference=f"Condition/{x}") for x in pat_store.con.resource_id
            ]
            encs = [
                Reference(reference=f"Encounter/{x}") for x in pat_store.enc.resource_id
            ]
            meds = [
                Reference(reference=f"Medication/{x}")
                for x in pat_store.med.resource_id
            ]
            obs = [
                Reference(reference=f"Observation/{x}")
                for x in pat_store.obs.request_id
            ]
            pros = [
                Reference(reference=f"Procedure/{x}")
                for x in pat_store.pro.procedure_id
            ]
            rsk_pat.basis = bdps + encs + conds + meds + obs + pros

            rsk_pat.occurrenceDateTime = dtnow

            rsk_pat.method = CodeableConcept(
                coding=[
                    Coding(
                        code=model_name,
                        system="https://uk-essen.de/fhir/autopilot/models",
                    )
                ]
            )

            # "The algorithm, process or mechanism used to evaluate the risk."
            rsk_pat.code = CodeableConcept(
                coding=[
                    Coding(
                        code="24_hour_forecast",
                        display="Probability of TKZ consumption within the next 24 hours",
                        system="https://uk-essen.de/fhir/autopilot/risk_assessment",
                    )
                ]
            )
            rsk_pat.prediction = [
                {
                    "probabilityDecimal": pat.day1,
                    "whenPeriod": Period(
                        start=dtnow,
                        end=get_time_zone_format(
                            pd.to_datetime(dtnow) + pd.Timedelta(days=1)
                        ),
                    ),
                },
            ]
            enc_unfiltered = pd.read_feather(self.config["encounter_path_filtered"])
            enc_pat = enc_unfiltered[enc_unfiltered.patient_id == pat.pat_id]
            enc_pat.sort_values(by=["encounter_end"], ascending=False)
            rsk_pat.encounter = Reference(
                reference=f"Encounter/{enc_pat.iloc[0]['resource_id']}"
            )
            retries = Retry(
                total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
            )
            http = PoolManager(retries=retries)
            response = http.request(
                "PUT",
                f"https://shipdev.uk-essen.de/app/FHIR/r4/RiskAssessment/{rsk_id}",
                body=json.dumps(
                    rsk_pat.dict(),
                    indent=2,
                    default=custom_json_serializer,
                ).encode(),
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/fhir+json",
                },
            )

            if response.status == 201:
                logging.debug(
                    f"Successfully created RiskAssessment object for patient {pat.pat_id}"
                )
            else:
                logging.warning(
                    f"Error cerating resource for {pat.pat_id}, code {response.status}, error msg {response.msg}"
                )
        logging.info(f"Predictions generated for {dtnow_date}")


def predict_push(token, config):
    tk_pred = TkPredict(config, token)
    predictions, live_data = tk_pred.get_predictions()
    if len(predictions) == 0:
        logging.info(f"no predictions for model: {config['clinic_filter']}")
        return
    tk_pred.push_predictions(predictions, live_data)


def main(config) -> None:
    token = fhir_login()

    # all
    config["dataset_name"] = "1.0.0"
    config["clinic_filter"] = "None"
    predict_push(token, config)

    # hematooncology
    config["dataset_name"] = "2.0.0"
    config["clinic_filter"] = "hematooncology"
    predict_push(token, config)

    # heart_thorax
    config["dataset_name"] = "3.0.0"
    config["clinic_filter"] = "heart_thorax"
    predict_push(token, config)


if __name__ == "__main__":
    main()
