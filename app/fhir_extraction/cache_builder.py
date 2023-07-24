import datetime
import logging
import os
import sys

PATH = [x[0] for x in os.walk(os.getcwd())]
sys.path += PATH

from pathlib import Path

from app.fhir_extraction.extract_transform_validate import (
    FHIRExtractor,
    FHIRFilter,
    FHIRValidator,
)


# Main
def main(config):
    bdp_path = Path(config["bdp_path"])
    bdp_path_filtered = Path(config["bdp_path_filtered"])
    obs_path = Path(config["obs_path"])
    obs_path_filtered = Path(config["obs_path_filtered"])
    patient_path = Path(config["patient_path"])
    patient_path_filtered = Path(config["patient_path_filtered"])
    procedure_path = Path(config["procedure_path"])
    procedure_path_filtered = Path(config["procedure_path_filtered"])
    procedure_op_plan_path = Path(config["procedure_op_plan_path"])
    procedure_op_plan_path_filtered = Path(config["procedure_op_plan_path_filtered"])
    encounter_path = Path(config["encounter_path"])
    encounter_path_filtered = Path(config["encounter_path_filtered"])
    condition_path = Path(config["condition_path"])
    condition_path_filtered = Path(config["condition_path_filtered"])
    medication_path = Path(config["medication_merged_path"])
    medication_path_filtered = Path(config["medication_merged_path_filtered"])

    extract = FHIRExtractor(config)
    filter = FHIRFilter(config)
    validator = FHIRValidator(config)

    # extract.build_bdp_all(config['bdp_params'])
    # exit()

    if config["is_live_prediction"]:
        config["start_datetime"] = (
            datetime.datetime.today() - datetime.timedelta(days=config["time_delta"])
        ).date()
        config["end_datetime"] = (
            datetime.datetime.today() + datetime.timedelta(days=1)
        ).date()

    if not bdp_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting BDP...")
        extract.build_bdp()

    if not bdp_path_filtered.exists() or config["reload_cache"]:
        logging.info(f"Fitering BDP...")
        filter.filter_bdp()
        logging.info("Validating BDP...")
        validator.validate_bdp()

    if not obs_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Observations...")
        extract.build_obs()

    if not obs_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering Observations...")
        filter.filter_observations()
        logging.info("Valditating Observations...")
        validator.validate_observations()

    if not patient_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Patient Data...")
        extract.build_patient()

    if not patient_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering Patients...")
        filter.filter_patients()
        validator.validate_patients()

    if not procedure_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Procedure Data")
        extract.build_procedure()

    if not procedure_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering Procedures")
        filter.filter_procedures()
        logging.info("Validating Procedures")
        validator.validate_procedures()

    # op plan is not yet implemented
    # if not procedure_op_plan_path.exists() or config["reload_cache"]:
    #     logging.info(f"Extracting Procedure OP plan data")
    #     extract.build_procedure_op_plan()
    #
    # if not procedure_op_plan_path_filtered.exists() or config["reload_cache"]:
    #     logging.info("Filtering procedure op plan data")
    #     filter.filter_procedures_op_plan()
    #     logging.info("Validation procedure op plan data")
    #     validator.validate_procedures()

    if not encounter_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Encounter Data")
        extract.build_encounter()

    if not encounter_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering and combining encounters")
        filter.filter_encounter()

    if not condition_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Condition Data")
        extract.build_condition()

    if not condition_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering relevant conditions")
        filter.filter_conditions()
        logging.info("Validating conditions")
        validator.validate_procedures()

    if not medication_path.exists() or config["reload_cache"]:
        logging.info("Extracting medications...")
        extract.build_medication()

    if not medication_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering medications")
        filter.filter_medications()
        logging.info("Validating medications")
        validator.validate_medicaitons()
