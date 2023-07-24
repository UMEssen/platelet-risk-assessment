import datetime
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fhir_pyrate.ahoy import Ahoy
from fhir_pyrate.pirate import Pirate
from requests.adapters import Retry
from tqdm import tqdm

from app.data_handling.create_sliding_window import col_to_datetime

# REQUIRED PATHS
from app.data_handling.get_training_data import (
    transform_dict_constants,
)

SEARCH_URL = os.environ["SEARCH_URL"]
BASIC_AUTH = os.environ["BASIC_AUTH"]
REFRESH_AUTH = os.environ["REFRESH_AUTH"]
FHIR_USER = os.environ["FHIR_USER"]
FHIR_PASSWORD = os.environ["FHIR_PASSWORD"]

# Authentication
auth = Ahoy(
    auth_method="env",
    username=FHIR_USER,
    auth_url=BASIC_AUTH,
    refresh_url=REFRESH_AUTH,
)

with open("app/config/constants.yaml", "r") as stream:
    code_dict = yaml.safe_load(stream)


# Class FHIRExtractor
class FHIRExtractor:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config
        # Pirate
        if config["read_from_fhir_cache"]:
            self.search = Pirate(
                print_request_url=True,
                auth=auth,
                base_url=SEARCH_URL,
                num_processes=1,
                cache_folder=self.config["bundle_cache_folder_path"],
                retry_requests=Retry(
                    total=3,  # Retries for a total of three times
                    backoff_factor=0.5,
                    # A backoff factor to apply between attempts, such that the requests are not run directly one after the other
                    status_forcelist=[
                        500,
                        502,
                        503,
                        504,
                    ],  # HTTP status codes that we should force a retry on
                    allowed_methods=[
                        "GET"
                    ],  # Set of uppercased HTTP method verbs that we should retry on
                ),
            )
        else:
            self.search = Pirate(
                auth=auth,
                base_url=SEARCH_URL,
                num_processes=5,
            )

    # BDP
    @staticmethod
    def extract_bdp(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            # ResourceType: BDP
            if resource.resourceType == "BiologicallyDerivedProduct":
                try:
                    resource_id = resource.id
                except Exception as e:
                    resource_id = np.nan

                try:
                    request_id = resource.request[0].reference.split("ServiceRequest/")[
                        -1
                    ]
                except Exception as e:
                    request_id = np.nan

                try:
                    ausgabe_datetime = resource.storage[0].duration.end
                except Exception as e:
                    ausgabe_datetime = pd.NaT

                try:
                    extensions = resource.extension
                    output_to = next(
                        (
                            e.valueString
                            for e in extensions
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/AUSGABEAN"
                        ),
                        None,
                    )
                except Exception as e:
                    extensions = np.nan
                    output_to = np.nan

                try:
                    ausgabe_type = next(
                        (
                            e.valueString
                            for e in extensions
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/VERBRAUCH"
                        ),
                        None,
                    )
                except Exception as e:
                    ausgabe_type = np.nan

                try:
                    product_code = resource.productCode.coding
                    code = next(
                        (
                            e.code
                            for e in product_code
                            if e.system
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/KONSART"
                        ),
                        None,
                    )
                except Exception as e:
                    code = np.nan

                elements = {
                    "resource_type": "bdp",
                    "resource_id": resource_id,
                    "request_id": request_id,
                    "ausgabe_datetime": ausgabe_datetime,
                    "ausgabe_type": ausgabe_type,
                    "code": code,
                    "output_to": output_to,
                }
                records.append(elements)

            # ResourceType: Service Request
            if resource.resourceType == "ServiceRequest":
                request_id = resource.id
                patient_id = resource.subject.reference.split("Patient/")[-1]
                try:
                    output_to_einskurz = resource.requester.extension[0].valueString
                    output_to_einscode = resource.requester.extension[1].valueString
                except Exception as e:
                    output_to_einskurz = None
                    output_to_einscode = None

                elements = {
                    "resource_type": "sr",
                    "request_id": request_id,
                    "patient_id": patient_id,
                    "output_to_einskurz": output_to_einskurz,
                    "output_to_einscode": output_to_einscode,
                }
                records.append(elements)

        return records

    def build_main_conditions_by_cohort(self, pats):
        df_enc = self.search.trade_rows_for_dataframe(
            df=pats,
            resource_type="Encounter",
            request_params={"_count": 100, "_sort": "-date"},
            df_constraints={
                "subject": "patient_id",
                "date": [("ge", "start_date"), ("le", "start_date")],
            },
        )

        # Drop NaN values in 'meta_extension_0_valueString'
        df_enc.dropna(subset=["meta_extension_0_valueString"], inplace=True)
        df_enc["meta_extension_0_valueString"] = [
            x.split(".")[-1] for x in df_enc["meta_extension_0_valueString"]
        ]

        # Filter DataFrame by 'Case' or 'Stay' in 'meta_extension_0_valueString'
        enc_filtered = df_enc[
            df_enc["meta_extension_0_valueString"].str.contains("Case|Stay")
        ]
        enc_filtered = enc_filtered[
            enc_filtered["meta_extension_0_valueString"] == "Case"
        ]

        df_conds = self.search.trade_rows_for_dataframe(
            df=enc_filtered,
            resource_type="Condition",
            request_params={
                "_count": 100,
                "_sort": "-recorded-date",
                "category": "med",
            },
            df_constraints={"encounter": "id"},
        )

        if len(df_conds):
            df_conds.reset_index(drop=True).to_feather(
                "/nvme/shared/autopilot/training_publish/main_diag.ftr"
            )
        else:
            print("No conditions found.")

    # Build BDP DataFrame
    def build_bdp(self):
        df = self.search.sail_through_search_space_to_dataframe(
            process_function=self.extract_bdp,
            resource_type="BiologicallyDerivedProduct",
            request_params=self.config["bdp_params"],
            time_attribute_name="shipStorageEnd",
            date_init=self.config["start_datetime"],
            date_end=self.config["end_datetime"],
        )

        bdp = df[df["resource_type"] == "bdp"]
        sr = df[df["resource_type"] == "sr"]

        merged = bdp.merge(sr, on="request_id", how="left")
        merged.drop(
            [
                "resource_type_x",
                "resource_type_y",
                "patient_id_x",
                "output_to_einskurz_x",
                "output_to_einscode_x",
                "resource_id_y",
                "ausgabe_datetime_y",
                "ausgabe_type_y",
                "code_y",
                "output_to_y",
            ],
            axis=1,
            inplace=True,
        )
        merged.columns = [
            "resource_id",
            "request_id",
            "ausgabe_datetime",
            "ausgabe_type",
            "code",
            "output_to",
            "patient_id",
            "output_to_einskurz",
            "output_to_einscode",
        ]

        merged.drop_duplicates(inplace=True)
        merged.reset_index(drop=True, inplace=False).to_feather(self.config["bdp_path"])

    # Observations
    @staticmethod
    def extract_observations(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            if resource.resourceType == "Observation":
                patient_id = resource.subject.reference.split("Patient/")[-1]

                try:
                    request_id = resource.basedOn[0].reference.split("ServiceRequest/")[
                        -1
                    ]
                except Exception as e:
                    request_id = np.nan

                code = resource.code.coding[0].code

                try:
                    value = resource.valueQuantity.value
                except Exception as e:
                    value = np.nan

                try:
                    unit = resource.valueQuantity.unit
                except Exception as e:
                    unit = np.nan

                # observed date
                if resource.effectiveDateTime is not None:
                    observation_date = resource.effectiveDateTime
                elif resource.issued is not None:
                    observation_date = resource.issued
                else:
                    observation_date = pd.NaT

                elements = {
                    "resource_type": "obs",
                    "patient_id": patient_id,
                    "request_id": request_id,
                    "code": code,
                    "value": value,
                    "unit": unit,
                    "observation_date": observation_date,
                }
                records.append(elements)

            if resource.resourceType == "ServiceRequest":
                request_id = resource.id
                req = resource.requester.extension
                if req is not None:
                    einskurz_from_dep = next(
                        (
                            e.valueString
                            for e in req
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/EINSENDER/EINSKURZ"
                        ),
                        None,
                    )
                    einscode_from_dep = next(
                        (
                            e.valueString
                            for e in req
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/EINSENDER/EINSCODE"
                        ),
                        None,
                    )
                else:
                    einskurz_from_dep = np.nan
                    einscode_from_dep = np.nan
                elements = {
                    "resource_type": "sr",
                    "request_id": request_id,
                    "einskurz_from_dep": einskurz_from_dep,
                    "einscode_from_dep": einscode_from_dep,
                }
                records.append(elements)

        return records

    # Build Observation DataFrame
    def build_obs(self):
        df = self.search.sail_through_search_space_to_dataframe(
            process_function=self.extract_observations,
            resource_type="Observation",
            request_params=self.config["obs_params"],
            time_attribute_name="date",
            date_init=self.config["start_datetime"],
            date_end=self.config["end_datetime"],
        )

        # df = pd.read_feather("/nvme/shared/autopilot/archive/obs.ftr")

        obs_obs = df[df.resource_type == "obs"]
        obs_sr = df[df.resource_type == "sr"]

        obs_merged = obs_obs.merge(obs_sr, on="request_id", how="left")
        obs_merged.dropna(how="all", inplace=True)
        obs_merged.dropna(subset=["value_x"], inplace=True)
        obs_merged = obs_merged[
            [
                "patient_id_x",
                "request_id",
                "value_x",
                "observation_date_x",
                "einscode_from_dep_y",
            ]
        ]

        obs_merged.columns = [
            "patient_id",
            "request_id",
            "value",
            "observation_date",
            "einscode_from_dep",
        ]

        obs_merged.reset_index(drop=True, inplace=False).to_feather(
            self.config["obs_path"]
        )

    # Build Patient df
    def build_patient(self):
        base_df = pd.read_feather(self.config["base_for_patient"])
        base_df.drop_duplicates(subset="patient_id", inplace=True)
        logging.info(f"numb of patients: {len(base_df)}")
        df = self.search.trade_rows_for_dataframe(
            df=base_df,
            resource_type="Patient",
            df_constraints=self.config["patient_constraints"],
            request_params=self.config["patient_params"],
            fhir_paths=[
                ("resource_type", "resourceType"),
                ("patient_id", "id"),
                ("gender", "gender"),
                ("birth_date", "birthDate"),
                ("deceasedDateTime", "deceasedDateTime"),
            ]
            # read_from_cache=self.config["read_from_fhir_cache"],
        )
        df.drop_duplicates(subset=["patient_id"], inplace=True)
        logging.info(f"numb of patients: {len(df)}")
        df.reset_index(drop=True, inplace=False).to_feather(self.config["patient_path"])

    # Procedure
    @staticmethod
    def extract_procedure(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            patient_id = resource.subject.reference.split("Patient/")[-1]
            procedure_id = resource.id

            status = resource.status

            try:
                code = resource.code.coding[0].code
            except Exception as e:
                code = np.nan

            if resource.performedPeriod is not None:
                procedure_start = resource.performedPeriod.start
            elif resource.performedDateTime is not None:
                procedure_start = resource.performedDateTime
            else:
                procedure_start = pd.NaT

            try:
                procedure_end = resource.performedPeriod.end
            except Exception as e:
                procedure_end = pd.NaT

            elements = {
                "patient_id": patient_id,
                "procedure_id": procedure_id,
                "status": status,
                "code": code,
                "procedure_start": procedure_start,
                "procedure_end": procedure_end,
            }
            records.append(elements)

        return records

    # Build Procedure DataFrame
    def build_procedure(self):
        base_df = pd.read_feather(self.config["base_for_procedure"])
        if self.config["is_live_prediction"]:
            dates = {
                "date": "ge"
                + str(self.config["start_datetime"])
                + "&date=le"
                + str(self.config["end_datetime"])
            }
        else:
            dates = {}
        procedure_params = self.config["procedure_params"] | dates

        df = self.search.trade_rows_for_dataframe(
            df=base_df,
            resource_type="Procedure",
            df_constraints=self.config["procedure_constraints"],
            process_function=self.extract_procedure,
            request_params=procedure_params,
            # read_from_cache=self.config["read_from_fhir_cache"],
        )
        df.procedure_start = df.procedure_start.astype(str)
        df.procedure_end = df.procedure_end.astype(str)
        df.reset_index(drop=True, inplace=False).to_feather(
            self.config["procedure_path"]
        )

    # Condition
    @staticmethod
    def extract_condition(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            patient_id = resource.subject.reference.split("Patient/")[-1]
            try:
                resource_id = resource.id
            except Exception as e:
                resource_id = np.nan
            try:
                condition_date = resource.recordedDate
            except Exception as e:
                condition_date = np.nan

            try:
                icd_code = resource.code.coding[0].code
            except Exception as e:
                icd_code = np.nan

            try:
                icd_display = resource.code.coding[0].display
            except Exception as e:
                icd_display = np.nan

            try:
                icd_code_root = icd_code.split(".")[0]
            except Exception as e:
                icd_code_root = np.nan

            elements = {
                "patient_id": patient_id,
                "condition_date": condition_date,
                "resource_id": resource_id,
                "icd_code": icd_code,
                "icd_display": icd_display,
                "icd_code_root": icd_code_root,
            }
            records.append(elements)

        return records

    # Building Condition DataFrame
    def build_condition(self):
        base_df = pd.read_feather(self.config["base_for_condition"])
        if self.config["is_live_prediction"]:
            dates = {"recorded-date": "ge" + str(self.config["start_datetime"])}
        else:
            dates = {}
        self.config["condition_params"] = self.config["condition_params"] | dates
        df = self.search.trade_rows_for_dataframe(
            df=base_df,
            resource_type="Condition",
            df_constraints=self.config["condition_constraints"],
            process_function=self.extract_condition,
            request_params=self.config["condition_params"],
            # read_from_cache=self.config["read_from_fhir_cache"],
        )
        # df = df[df.icd_code.str.contains('|'.join(condition_code))]
        df.reset_index(drop=True, inplace=False).to_feather(
            self.config["condition_path"]
        )

    # Encounter
    @staticmethod
    def extract_encounter(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            resource_id = resource.id
            patient_id = resource.subject.reference.split("Patient/")[-1]
            status = resource.status
            try:
                encounter_start = resource.period.start
            except Exception as e:
                encounter_start = pd.NaT
            try:
                encounter_end = resource.period.end
            except Exception as e:
                encounter_end = pd.NaT

            try:
                if "Case" in resource.meta.extension[0].valueString:
                    encounter_type = "case"
                elif "Stay" in resource.meta.extension[0].valueString:
                    encounter_type = "stay"
                else:
                    encounter_type = np.nan
            except:
                encounter_type = np.nan

            elements = {
                "resource_id": resource_id,
                "patient_id": patient_id,
                "status": status,
                "encounter_start": encounter_start,
                "encounter_end": encounter_end,
                "encounter_type": encounter_type,
            }
            records.append(elements)

        return records

    # Building Encounter DataFrame
    def build_encounter(self):
        base_df = pd.read_feather(self.config["base_for_encounter"])
        if self.config["is_live_prediction"]:
            dates = {"date": "ge" + str(self.config["start_datetime"])}
        else:
            dates = {}
        self.config["encounter_params"] = self.config["encounter_params"] | dates
        df = self.search.trade_rows_for_dataframe(
            df=base_df,
            resource_type="Encounter",
            df_constraints=self.config["encounter_constraints"],
            process_function=self.extract_encounter,
            request_params=self.config["encounter_params"],
        )
        df.encounter_start = df.encounter_start.astype(str)
        df.encounter_end = df.encounter_end.astype(str)
        df.reset_index(drop=True, inplace=False).to_feather(
            self.config["encounter_path"]
        )

    # Medication
    @staticmethod
    def extract_medication(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            importer = next(
                (
                    e.valueString
                    for e in resource.meta.extension
                    if e.url == "http://uk-essen.de/fhir/extension-importer-name"
                ),
                None,
            )
            try:
                if "cato" in importer:
                    medicationName = resource.code.text
                    source = "cato"
                elif "medico" in importer:
                    source = "medico"
                    medicationName = resource.code.coding[0].display
                else:
                    source = "other"
                    medicationName = (
                        resource.code.coding[0].display
                        if resource.code.coding[0].display
                        else np.nan
                    )
            except Exception as e:
                medicationName = resource.code.text
                source = np.nan

            lastUpdated = resource.meta.lastUpdated
            medication_id = resource.id

            elements = {
                "medicationName": medicationName,
                "source": source,
                "lastUpdated": lastUpdated,
                "medication_id": medication_id,
            }

            records.append(elements)

        return records

    @staticmethod
    def extract_medication_request(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            try:
                importer = next(
                    (
                        e.valueString
                        for e in resource.meta.extension
                        if e.url == "http://uk-essen.de/fhir/extension-importer-name"
                    ),
                    None,
                )
                if "cato" in importer:
                    source = "cato"
                elif "medico" in importer:
                    source = "medico"
                else:
                    source = np.nan
            except Exception:
                source = np.nan

            status = resource.status
            try:
                medication_id = resource.medicationReference.reference.split(
                    "Medication/"
                )[-1]
            except:
                medication_id = np.nan
            patient_id = resource.subject.reference.split("Patient/")[-1]
            try:
                event_time = (
                    resource.dosageInstruction[0].timing.event[0]
                    if resource.dosageInstruction[0].timing
                    else np.nan
                )
            except:
                event_time = np.nan
            resource_id = resource.id

            elements = {
                "resource_id": resource_id,
                "medication_id": medication_id,
                "patient_id": patient_id,
                "event_time": event_time,
                "source": source,
                "status": status,
            }

            records.append(elements)

        return records

    @staticmethod
    def extract_medication_administration(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            try:
                importer = next(
                    (
                        e.valueString
                        for e in resource.meta.extension
                        if e.url == "http://uk-essen.de/fhir/extension-importer-name"
                    ),
                    None,
                )
                if "cato" in importer:
                    source = "cato"
                elif "medico" in importer:
                    source = "medico"
                else:
                    source = np.nan
            except Exception as e:
                source = np.nan

            status = resource.status

            medication_id = (
                resource.medicationReference.reference.split("Medication/")[-1]
                if resource.medicationReference is not None
                else None
            )
            patient_id = resource.subject.reference.split("Patient/")[-1]
            try:
                event_time = resource.effectiveDateTime
            except Exception as e:
                event_time = resource.effectivePeriod.start
            except Exception as e:
                event_time = pd.NaT

            resource_id = resource.id

            elements = {
                "resource_id": resource_id,
                "medication_id": medication_id,
                "patient_id": patient_id,
                "event_time": event_time,
                "source": source,
                "status": status,
            }
            records.append(elements)

        return records

    # Build Medication DataFrame
    def build_medication(self):
        base_df = pd.read_feather(self.config["base_for_medication"])
        base_dict = transform_dict_constants()

        # only pull medications if they are older than 30 days in live prediciton
        if (
            self.config["is_live_prediction"]
            and Path(self.config["medication_path_raw"]).exists()
        ):
            modified_date = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.config["medication_path_raw"])
            )
            duration = datetime.datetime.today() - modified_date
            re_download = True if duration.days > 30 else False
        else:
            re_download = True

        # 1. Get all medications by ID
        if re_download:
            df_medications = self.search.sail_through_search_space_to_dataframe(
                process_function=self.extract_medication,
                request_params=self.config["medication_params"],
                resource_type="Medication",
                time_attribute_name="_lastUpdated",
                date_init="01-01-2000",
                date_end=datetime.datetime.now().date(),
            )
            df_medications.columns = df_medications.columns.astype(str)

            # Simplify medication labels e.g. Temozolomid HEXAL 100mg -> Temozolomid
            df_medications["substance"] = np.nan
            df_medications.dropna(subset=["medicationName"], inplace=True)
            # translating brand names to acutal substances -> keys in medication dict
            df_medications["substance"] = df_medications.medicationName.apply(
                lambda x: [
                    [
                        med
                        for med in sum(base_dict["MEDICATION_DICT_LIST"].values(), [])
                        if med.lower() in x.lower()
                    ]
                    or [np.nan]
                ][0]
            )

            df_medications.dropna(subset=["substance"]).reset_index(
                inplace=False, drop=True
            ).to_feather(self.config["medication_path_raw"])
        else:
            df_medications = pd.read_feather(self.config["medication_path_raw"])

        # 2. MedicationRequest by medications in scope
        if self.config["is_live_prediction"]:
            # todo verify in live before prod live prediction
            dates = {"date": "ge" + str(self.config["start_datetime"])}
        else:
            dates = {}
        self.config["medication_req_params"] = (
            self.config["medication_req_params"] | dates
        )
        df_request = self.search.trade_rows_for_dataframe(
            df=base_df,
            process_function=self.extract_medication_request,
            request_params=self.config["medication_req_params"],
            df_constraints=self.config["medication_constraints"],
            resource_type="MedicationRequest",
        )
        df_req_filtered = df_medications.merge(
            df_request, on="medication_id", how="inner"
        )

        # 3. MedicationAdministration by medications in scope
        if self.config["is_live_prediction"]:
            # todo verify in live before prod live prediction
            dates = {"_lastUpdated": "ge" + str(self.config["start_datetime"])}
        else:
            dates = {}
        self.config["medication_admin_params"] = (
            self.config["medication_admin_params"] | dates
        )
        df_admin = self.search.trade_rows_for_dataframe(
            df=df_medications,
            process_function=self.extract_medication_administration,
            request_params=self.config["medication_admin_params"],
            df_constraints={"medication": "medication_id"},
            resource_type="MedicationAdministration",
        )
        df_admin_filtered = df_medications.merge(
            df_admin, on="medication_id", how="inner"
        )

        df_merged = pd.concat([df_req_filtered, df_admin_filtered], axis=0)

        df_merged.lastUpdated = df_merged.lastUpdated.astype(str)
        df_merged.event_time = df_merged.event_time.astype(str)
        df_merged.reset_index(drop=True, inplace=False).to_feather(
            self.config["medication_merged_path"]
        )

    def build_procedure_op_plan(self):
        if self.config["is_live_prediction"]:
            dates = {"date": "ge" + str(self.config["start_datetime"])}

        else:
            dates = {}

        self.config["procedure_params_op_plan"] = (
            self.config["procedure_params_op_plan"] | dates
        )

        base_df = pd.read_feather(self.config["base_for_procedure_op_plan"])
        df = self.search.trade_rows_for_dataframe(
            df=base_df,
            df_constraints=self.config["procedure_op_plan_constraints"],
            resource_type="Procedure",
            request_params=self.config["procedure_params_op_plan"],
        )

        df.reset_index(drop=True, inplace=False).to_feather(
            self.config["procedure_op_plan_path"]
        )

    def build_bdp_all(self):
        df = self.search.sail_through_search_space_to_dataframe(
            # process_function=self.extract_bdp,
            resource_type="BiologicallyDerivedProduct",
            request_params=self.config["bdp_params"],
            time_attribute_name="shipStorageEnd",
            # read_from_cache=self.config["read_from_fhir_cache"],
        )
        df.reset_index(drop=True, inplace=False).to_feather(
            "/nvme/shared/autopilot/live/bdp_all_raw.ftr"
        )


# Class FHIRExtractor
class FHIRFilter:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def filter_date(
        start: datetime, end: datetime, resource: pd.DataFrame, date_col: str
    ) -> pd.DataFrame:
        df = resource[
            ((start <= resource[date_col]) & (resource[date_col] <= end))
        ].sort_values([date_col])

        return df

    def filter_bdp(self) -> None:
        bdp = pd.read_feather(self.config["bdp_path"])
        bdp["date"] = col_to_datetime(bdp["ausgabe_datetime"])
        if not self.config["is_live_prediction"]:
            bdp = self.filter_date(
                self.config["start_datetime"], self.config["end_datetime"], bdp, "date"
            )
        bdp = bdp.drop_duplicates(subset=["resource_id"], keep="first")
        bdp.dropna(subset=["output_to"], inplace=True)
        bdp = bdp[bdp.ausgabe_type.str.contains("AUSGABE")]
        bdp.drop(
            columns=[
                "code",
                "ausgabe_type",
                "output_to",
                "output_to_einskurz",
                "ausgabe_datetime",
            ],
            inplace=True,
        )
        bdp.reset_index(drop=True).to_feather(self.config["bdp_path_filtered"])

    @staticmethod
    def format_iso8601(x):
        if x <= datetime.datetime.utcnow():
            return str(x.replace(tzinfo=datetime.timezone.utc).isoformat())
        else:
            tmp = str(
                datetime.datetime.utcnow()
                .replace(tzinfo=datetime.timezone.utc)
                .isoformat()
            )
            tmp = tmp.split(".")
            tmp_2 = tmp[1].split("+")[-1]
            tmp_final = f"{tmp[0]}+{tmp_2}"
            return tmp_final

    def filter_observations(self) -> None:
        obs = pd.read_feather(self.config["obs_path"])
        obs.dropna(how="all", inplace=True)
        obs.dropna(subset=["value"], inplace=True)
        obs["date"] = col_to_datetime(obs["observation_date"])
        if not self.config["is_live_prediction"]:
            obs = self.filter_date(
                self.config["start_datetime"], self.config["end_datetime"], obs, "date"
            )
        obs.drop(columns=["observation_date"], inplace=True)
        obs = obs[obs.value <= 450]
        obs.reset_index(drop=True).to_feather(self.config["obs_path_filtered"])

    def filter_patients(self) -> None:
        pats = pd.read_feather(self.config["patient_path"])
        obs = pd.read_feather(self.config["obs_path_filtered"])
        # only take patients that have relevant observations
        pats = pats.merge(
            pd.DataFrame(obs.patient_id.unique(), columns=["patient_id"]),
            on="patient_id",
        )
        pats.drop(columns=["resource_type"], inplace=True)
        pats.dropna(subset=["birth_date"], inplace=True)
        # patients which passed are irrelevant on live prediciton
        if self.config["is_live_prediction"]:
            pats = pats[pats.deceasedDateTime.isnull()]
        # only allow binary gender
        pats.gender = ["female" if x == "female" else "male" for x in pats.gender]
        pats.reset_index(drop=True).to_feather(self.config["patient_path_filtered"])

    def filter_procedures(self) -> None:
        pro = pd.read_feather(self.config["procedure_path"])
        if not self.config["is_live_prediction"]:
            pro["procedure_start"] = col_to_datetime(pro["procedure_start"])
            pro["procedure_end"] = col_to_datetime(pro["procedure_end"])
            pro = self.filter_date(
                self.config["start_datetime"],
                self.config["end_datetime"],
                pro,
                "procedure_start",
            )
            # pro["duration"] = pro["procedure_end"] - pro["procedure_start"]
        pro = pro[pro.status.isin(["completed", "in-progress", "preparation"])]
        pro.drop(columns=["status"], inplace=True)

        pro_list = list()
        pro_dict = code_dict["PROCEDURE_DICT_LIST"]
        for pat_id, value in pro_dict.items():
            if type(value) is list:
                for ele in value:
                    pro_list.append(ele)
            else:
                pro_list.append(value)

        pro = pro[~pd.isna(pro.code)]
        pro_filtered = pro[pro.code.str.contains("|".join(pro_list))]
        pro_filtered.reset_index(drop=True).to_feather(
            self.config["procedure_path_filtered"]
        )

    def filter_procedures_op_plan(self) -> None:
        prod_plan = pd.read_feather(self.config["procedure_op_plan_path"])
        op_plan_filtered = prod_plan[
            [
                "id",
                "status",
                "extension_0_valueDateTime",
                "identifier_0_value",
                "basedOn_0_reference",
                "category_coding_0_code",
                "category_coding_0_display",
                "code_coding_0_code",
                "code_coding_0_display",
                "subject_reference",
                "performedPeriod_start",
                "performedPeriod_end",
            ]
        ]
        op_plan_filtered.columns = [
            "id",
            "status",
            "creation_DateTime",
            "medico_id",
            "ServiceRequest",
            "sct_code",
            "sct_display",
            "medico_code",
            "medico_display",
            "patient_id",
            "performedPeriod_start",
            "performedPeriod_end",
        ]
        op_plan_filtered_clean = op_plan_filtered.dropna(
            subset=["performedPeriod_start"]
        ).reset_index(drop=True)
        op_plan_filtered_clean = op_plan_filtered_clean.drop_duplicates(
            subset=["performedPeriod_start", "performedPeriod_end", "patient_id"],
            keep="first",
        )
        op_plan_filtered_clean.reset_index(drop=True).to_feather(
            self.config["procedure_op_plan_path_filtered"]
        )

    def filter_conditions(self) -> None:
        con = pd.read_feather(self.config["condition_path"])
        if not self.config["is_live_prediction"]:
            con["condition_date"] = col_to_datetime(con["condition_date"])
            con = self.filter_date(
                self.config["start_datetime"],
                self.config["end_datetime"],
                con,
                "condition_date",
            )

        con_list = list()
        con_dict = code_dict["CONDITION_DICT_LIST"]
        for pat_id, value in con_dict.items():
            if type(value) is list:
                for ele in value:
                    con_list.append(ele)
            else:
                con_list.append(value)

        con = con[~pd.isna(con.icd_code)]
        con_filtered = con[con.icd_code.str.contains("|".join(con_list))]
        con_filtered.reset_index(drop=True).to_feather(
            self.config["condition_path_filtered"]
        )

    def filter_medications(self) -> None:
        # status: active, completed, in-progess
        med = pd.read_feather(self.config["medication_merged_path"])
        med = med[med.status.isin(["active", "completed", "in-progress", "draft"])]
        med.drop(
            columns=["source_x", "lastUpdated", "source_y", "status"], inplace=True
        )

        if not self.config["is_live_prediction"]:
            med["event_time"] = col_to_datetime(med["event_time"])
            med = self.filter_date(
                self.config["start_datetime"],
                self.config["end_datetime"],
                med,
                "event_time",
            )

        med.substance = [x[0] for x in med.substance]
        med.dropna(subset=["substance"], inplace=True)

        med.reset_index(drop=True).to_feather(
            self.config["medication_merged_path_filtered"]
        )

    @staticmethod
    def to_datetime(df, col_format):
        for k, v in tqdm(col_format.items(), desc=f"Converting DateTime"):
            df[k] = pd.to_datetime(df[k], format=v, utc=True, errors="coerce")
        return df

    @staticmethod
    def clean_encounter(args):
        is_live_prediction, enc, pat = args

        enc_uniq = enc[enc.patient_id == pat]

        enc_uniq.drop_duplicates(
            subset=["encounter_start", "encounter_end"], inplace=True
        )
        enc_uniq.sort_values(["encounter_start", "encounter_end"], inplace=True)
        enc_uniq.reset_index(drop=True, inplace=True)

        enc_uniq["next_start"] = enc_uniq["encounter_start"].shift(-1)

        mask = (enc_uniq["next_start"] >= enc_uniq["encounter_start"]) & (
            enc_uniq["next_start"] <= enc_uniq["encounter_end"]
        )
        enc_uniq["overlap"] = enc_uniq.resource_id.isin(
            enc_uniq.loc[mask].resource_id.tolist()
        )

        enc_uniq = enc_uniq.reindex(
            columns=[
                "encounter_start",
                "next_start",
                "encounter_end",
                "overlap",
                "patient_id",
                "resource_id",
                "encounter_type",
                "status",
            ]
        )
        enc_uniq.reset_index(inplace=True, drop=True)
        enc_uniq["all_encounters"] = enc_uniq["resource_id"]

        true_df = enc_uniq[enc_uniq.overlap == True]
        while len(true_df) > 0:
            curr_resource = true_df.resource_id.iloc[0]
            idx = enc_uniq[enc_uniq.resource_id == curr_resource].index.values[0]

            curr_end = enc_uniq.encounter_end.iloc[idx]
            next_end = enc_uniq.encounter_end.iloc[idx + 1]

            if next_end <= curr_end:
                enc_uniq.encounter_end.iloc[idx] = curr_end
            else:
                enc_uniq.encounter_end.iloc[idx] = next_end

            enc_uniq["all_encounters"].iloc[
                idx
            ] = f"{enc_uniq.all_encounters.iloc[idx]},{enc_uniq.all_encounters.iloc[idx + 1]}"
            enc_uniq.drop([idx + 1], axis=0, inplace=True)
            enc_uniq.reset_index(inplace=True, drop=True)
            enc_uniq["next_start"] = enc_uniq["encounter_start"].shift(-1)

            mask = (enc_uniq["next_start"] >= enc_uniq["encounter_start"]) & (
                enc_uniq["next_start"] <= enc_uniq["encounter_end"]
            )
            enc_uniq["overlap"] = enc_uniq.resource_id.isin(
                enc_uniq.loc[mask].resource_id.tolist()
            )

            enc_uniq = enc_uniq.reindex(
                columns=[
                    "encounter_start",
                    "next_start",
                    "encounter_end",
                    "overlap",
                    "patient_id",
                    "resource_id",
                    "encounter_type",
                    "status",
                    "all_encounters",
                ]
            )
            enc_uniq.reset_index(inplace=True, drop=True)
            true_df = enc_uniq[enc_uniq.overlap == True]

        enc_uniq.sort_values(by="encounter_start", inplace=True)
        enc_uniq_filtered = enc_uniq.query('status in ["finished", "in-progress"]')
        # for live predictions planned encounters shall not be combinded
        enc_uniq = enc_uniq.query('status in ["planned"]')

        # Additional filtering to fusion encounters with a distance of less than 3 days apart
        if len(enc_uniq_filtered) > 1:
            enc_uniq_filtered.sort_values(by=["encounter_start"], axis=0, inplace=True)
            enc_uniq_filtered.reset_index(drop=True, inplace=True)
            enc_uniq_filtered.overlap = True
            index = 0
            while sum(enc_uniq_filtered.overlap.values):
                if len(enc_uniq_filtered) <= index + 1:
                    enc_uniq_filtered.at[index - 1, "overlap"] = False
                    break
                if (
                    enc_uniq_filtered.iloc[index + 1].encounter_start
                    - enc_uniq_filtered.iloc[index].encounter_end
                ).days <= 2:
                    enc_uniq_filtered.at[index, "encounter_end"] = enc_uniq_filtered.at[
                        index + 1, "encounter_end"
                    ]
                    if isinstance(
                        enc_uniq_filtered.at[index, "all_encounters"], str
                    ) or isinstance(
                        enc_uniq_filtered.at[index, "all_encounters"], float
                    ):
                        # convert all encounters to list and then append
                        enc_uniq_filtered.at[index, "all_encounters"] = list(
                            str(enc_uniq_filtered.at[index, "all_encounters"])
                        )
                    enc_uniq_filtered.at[
                        index, "all_encounters"
                    ] = enc_uniq_filtered.at[index, "all_encounters"].append(
                        pd.Series(enc_uniq_filtered.at[index + 1, "all_encounters"])
                    )
                    enc_uniq_filtered.drop(index=index + 1, inplace=True)
                else:
                    enc_uniq_filtered.at[index, "overlap"] = False
                index += 1
            # only live prediction should care about planned procedures
            if is_live_prediction:
                enc_uniq_filtered = pd.concat([enc_uniq_filtered, enc_uniq])

        return enc_uniq_filtered.reset_index(drop=True).sort_values(
            by="encounter_start"
        )

    def filter_encounter(self) -> None:
        enc = pd.read_feather(self.config["encounter_path"])

        enc["date"] = col_to_datetime(enc["encounter_start"])
        if not self.config["is_live_prediction"]:
            enc = self.filter_date(
                self.config["start_datetime"], self.config["end_datetime"], enc, "date"
            )

        enc_format = {
            "encounter_start": "%Y-%m-%dT%H:%M:%S.%f",
            "encounter_end": "%Y-%m-%dT%H:%M:%S.%f",
        }
        enc = self.to_datetime(enc, enc_format)
        patient_ids = enc.patient_id.unique().tolist()
        args = [(self.config["is_live_prediction"], enc, pat) for pat in patient_ids]
        logging.info(f"Starting Pool...")
        pool = Pool(processes=64)
        results = pool.map(self.clean_encounter, args)
        pool.close()
        df = pd.concat(results, axis=0).reset_index(inplace=False, drop=True)
        encounter_end_dt = col_to_datetime(df.encounter_end)
        if self.config["is_live_prediction"]:
            df["encounter_end"] = encounter_end_dt.apply(
                lambda x: self.format_iso8601(x)
            )
        df.reset_index(drop=True).to_feather(self.config["encounter_path_filtered"])
        logging.info(f"Final shape: {df.shape}")


# Class FHIRExtractor
class FHIRValidator:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def na_checker(field_name: str, na_counts: pd.Series, is_error: bool) -> None:
        if na_counts[field_name] and is_error:
            logging.error(f"At least one {field_name} is zero")
            raise ValueError(f"At least one {field_name} is zero")
        elif na_counts[field_name]:
            logging.warning(f"At least one {field_name} is zero")
        else:
            logging.info(f"Validation for {field_name} passed")

    def validate_bdp(self) -> None:
        bdp = pd.read_feather(self.config["bdp_path_filtered"])
        bdp["ausgabe_datetime"] = col_to_datetime(bdp.date)
        bdp_count = bdp.ausgabe_datetime.value_counts().sort_index()[:-1]
        if (bdp_count == 0).any():
            logging.warning("BDP count for one or more imported days = 0")

    def validate_observations(self) -> None:
        obs = pd.read_feather(self.config["obs_path_filtered"])
        na_counts = obs.isna().sum()
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("value", na_counts, True)

    def validate_patients(self) -> None:
        pats = pd.read_feather(self.config["patient_path_filtered"])
        if pd.Series(pats.gender).value_counts().index.size > 2:
            logging.error("Gender is not binary in data")

        na_counts = pats.isna().sum()
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("gender", na_counts, True)
        self.na_checker("birth_date", na_counts, False)

    def validate_procedures(self) -> None:
        pros = pd.read_feather(self.config["procedure_path_filtered"])
        if pros.patient_id.isna().sum() != 0:
            logging.warning("Some patient ids are na")

    def validate_condition(self) -> None:
        cond = pd.read_feather(self.config["condition_path_filtered"])
        na_counts = cond.isna().sum()

        self.na_checker("patient_id", na_counts, True)
        self.na_checker("icd_code", na_counts, True)
        self.na_checker("condition_date", na_counts, True)

    def validate_medicaitons(self) -> None:
        meds = pd.read_feather(self.config["medication_merged_path_filtered"])
        na_counts = meds.isna().sum()
        self.na_checker("medicationName", na_counts, True)
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("event_time", na_counts, True)

    def validate_encounter(self) -> None:
        enc = pd.read_feather(self.config["encounter_path_filtered"])
        na_counts = enc.isna().sum()
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("encounter_start", na_counts, True)
