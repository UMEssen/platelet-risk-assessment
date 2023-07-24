""" Main file where one can define what functions to run """
import argparse
import logging

# Imports
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)
LOG_LEVEL = logging.ERROR
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(
    "%(log_color)s[%(levelname).1s %(log_color)s%(asctime)s] - %(log_color)s%(name)s %(reset)s- "
    "%(message)s"
)
external_loggers = [logger]
for package in ["fhir_pyrate", "urllib3"]:
    package_logger = logging.getLogger(package)
    package_logger.setLevel(LOG_LEVEL)
    external_loggers.append(package_logger)

log_path = Path(Path.cwd() / "app/logs/run_{datetime.datetime.now()}.log")
handlers = [logging.StreamHandler(), logging.FileHandler(log_path)]
for h in handlers:
    h.setFormatter(formatter)
    for log in external_loggers:
        log.addHandler(h)

from app.fhir_extraction import cache_builder
from app.data_handling import create_sliding_window
from app.ml import (
    live_predict,
    train_nn,
    public_helper,
    train,
    explain_train_nn_lime,
    explain_train,
)

# Config file(s)
config = yaml.safe_load((Path.cwd() / "app/config/config_training.yaml").open())


def clear_process_data():
    if not config["is_live_prediction"]:
        if input("Do you really want to delete the cache (y:yes)?:") != "y":
            exit()
    folders = config["folders_to_clear"]
    for folder in folders:
        for filename in os.listdir(folder):
            # raw medication file shall only be extracted every 30 days to speed up pipline so we don't delete it here
            if filename == config["medication_path_raw"].parts[-1]:
                continue
            file_path = os.path.join(folder, filename)
            try:
                logger.info(f"deleting: {file_path}")
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.info("Failed to delete %s. Reason: %s" % (file_path, e))


# FHIR
def build_cache():
    logger.info(f"Checking and building cache...")
    start = time.time()
    cache_builder.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def build_time_series():
    logging.info("Creating sliding windows")
    create_sliding_window.main(config)


def launch_live_prediction():
    logging.info("Prediciton on live data")
    live_predict.main(config)


def launch_live_prediction():
    logging.info("Prediciton on live data")
    live_predict.main(config)


def launch_dl_training():
    logging.info("Starting model training with DL")
    train_nn.main(config)


def launch_ml_training():
    logging.info("Launching training for classical ML models")
    set_paths()
    train.main(config)


def set_paths() -> None:
    def helper_set_paths(folder_name: str) -> str:
        return config["root_dir"] / Path(folder_name)

    for key, value in config.items():
        if isinstance(value, str) and value.__contains__("./"):
            config[key] = helper_set_paths(value)
        if isinstance(value, list):
            if value[0].__contains__("./"):
                config[key] = [helper_set_paths(x) for x in config["folders_to_clear"]]
    return


def live_pipeline(init=False):
    if init:
        set_paths()
    blocks = [
        clear_process_data,
        build_cache,
        build_time_series,
        launch_live_prediction,
    ]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {np.round((end - start) / 60, 2)} minutes")


# Main
def nn_training_pipeline():
    blocks = [
        set_paths,
        # clear_process_data,
        build_cache,
        build_time_series,
        launch_dl_training,
    ]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {np.round((end - start) / 60, 2)} minutes")


def launch_publish():
    logging.info("Starting publish")
    public_helper.main(config),


def publish_pipline():
    blocks = [set_paths, launch_publish]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {np.round((end - start) / 60, 2)} minutes")


def explain_rnn_pipline():
    logging.info("Starting explain LSTM")
    set_paths()
    explain_train_nn_lime.main(config)


def parse_args_nn() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # create the top-level parser
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # create parser for "nn" command
    nn_parser = subparsers.add_parser("dl", help="Neural network related arguments.")
    nn_parser.add_argument("--run_cv", type=bool, required=False, default=False)
    nn_parser.add_argument(
        "--eval_ensemble_bs",
        type=bool,
        required=False,
        default=False,
        help="Generate ensemble model and evaluate",
    )
    nn_parser.add_argument(
        "--explain_lstm",
        type=bool,
        required=False,
        default=False,
        help="Explainable of NN",
    )
    nn_parser.add_argument(
        "--launch_training",
        type=bool,
        required=False,
        default=False,
        help="Start NN training, either with or without CV and build ensemble model + evaluation",
    )
    nn_parser.add_argument(
        "--launch_publish",
        type=bool,
        required=False,
        default=False,
        help="Launch analysis to publish results",
    )

    nn_parser.add_argument(
        "--root_dir",
        type=Path,
        required=False,
        default="/local/work/merengelke/autopilot/",
    )
    nn_parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=config["batch_size"],
        help="batch size",
    )
    nn_parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=config["learning_rate"],
        help="learning_rate",
    )
    nn_parser.add_argument(
        "--in_lstm", type=int, required=False, default=config["in_lstm"], help="in_lstm"
    )
    nn_parser.add_argument(
        "--out_lstm",
        type=int,
        required=False,
        default=config["out_lstm"],
        help="out_lstm",
    )
    nn_parser.add_argument(
        "--dense", type=int, required=False, default=config["dense"], help="dense"
    )
    nn_parser.add_argument(
        "--norm_type",
        type=str,
        required=False,
        default=config["norm_type"],
        help="norm_type",
    )
    nn_parser.add_argument(
        "--apply_weight_balancing",
        type=lambda x: (str(x).lower() == "true"),
        required=False,
        default=config["apply_weight_balancing"],
        help="apply_weight_balancing",
    )
    nn_parser.add_argument(
        "--loss_function",
        type=str,
        required=False,
        default=config["loss_function"],
        help="loss_function",
    )
    nn_parser.add_argument(
        "--use_sam",
        type=lambda x: (str(x).lower() == "true"),
        required=False,
        default=config["use_sam"],
        help="use sam algo",
    )
    nn_parser.add_argument(
        "--clinic_filter",
        type=str,
        required=False,
        default=config["clinic_filter"],
        help="clinic_filter",
    )
    nn_parser.add_argument(
        "--is_live_prediction",
        type=lambda x: (str(x).lower() == "true"),
        required=False,
        default=config["is_live_prediction"],
        help="trigger live pipline",
    )

    # create parser for "ml" command
    ml_parser = subparsers.add_parser("ml", help="Machine learning related arguments.")
    ml_parser.add_argument(
        "--explain_rf",
        type=bool,
        help="Do explainablity of RF",
        default=False,
    )
    ml_parser.add_argument(
        "--root_dir",
        type=Path,
        required=False,
        default="/local/work/merengelke/autopilot/",
    )
    ml_parser.add_argument(
        "--booster", type=str, help="The booster to use", default="gbtree"
    )
    ml_parser.add_argument(
        "--clinic_filter", type=str, help="Dataset to load", default="none"
    )
    ml_parser.add_argument("--reg_lambda", type=float, help="Lambda", default=1.0)
    ml_parser.add_argument("--reg_alpha", type=float, help="Alpha", default=0.0)
    ml_parser.add_argument("--max_depth", type=int, help="Max depth", default=None)
    ml_parser.add_argument(
        "--learning_rate", type=float, help="Learning rate", default=0.1
    )
    ml_parser.add_argument("--gamma", type=float, help="Gamma", default=0.0)
    ml_parser.add_argument(
        "--grow_policy", type=str, help="Grow policy", default="depthwise"
    )
    ml_parser.add_argument(
        "--sampling_method", type=str, help="Sampling method", default="uniform"
    )
    ml_parser.add_argument(
        "--evaluate_bs",
        type=bool,
        help="Do bootstrapping on the trained/saved model",
        default=False,
    )

    args = parser.parse_args()
    config.update(vars(args))
    return args


def define_ds_name(ai_type: str):
    if ai_type == "ml":
        ai_type = "rf"
    if config["clinic_filter"] == "heart_thorax":
        config["dataset_name"] = "0.3.1"
        config["input_dim_station_embedding"] = 18
        config["wb_project_name"] = (
            config["wb_project_name"] + "_" + ai_type + "_heart_thorax"
        )
    elif config["clinic_filter"] == "hematooncology":
        config["input_dim_station_embedding"] = 21
        config["dataset_name"] = "0.2.0"
        config["wb_project_name"] = (
            config["wb_project_name"] + "_" + ai_type + "_hematooncology"
        )
    elif config["clinic_filter"] == "none":
        config["dataset_name"] = "0.1.3"
        config["input_dim_station_embedding"] = 291
        config["wb_project_name"] = config["wb_project_name"] + "_" + ai_type + "_none"
    else:
        raise ValueError("Wrong clinic_filter value")


def define_pretrained_dl_model_config():
    if config["clinic_filter"] == "heart_thorax":
        dl_config = yaml.load(
            open("./app/config/model_heart_thorax/config.yaml"), Loader=yaml.FullLoader
        )
    elif config["clinic_filter"] == "hematooncology":
        dl_config = yaml.load(
            open("./app/config/model_hematooncology/config.yaml"),
            Loader=yaml.FullLoader,
        )
    elif config["clinic_filter"] == "none":
        dl_config = yaml.load(
            open("./app/config/model_none/config.yaml"), Loader=yaml.FullLoader
        )
    else:
        raise ValueError("Wrong clinic_filter value")
    if config["eval_ensemble_bs"]:
        config["shuffle_files"] = False

    config["in_lstm"] = dl_config["in_lstm"]["value"]
    config["out_lstm"] = dl_config["out_lstm"]["value"]
    config["batch_size"] = dl_config["batch_size"]["value"]
    config["epochs"] = dl_config["epochs"]["value"]
    config["learning_rate"] = dl_config["learning_rate"]["value"]
    config["weight_decay_score"] = dl_config["weight_decay_score"]["value"]
    config["prefetch"] = dl_config["prefetch"]["value"]
    config["train_size"] = dl_config["train_size"]["value"]
    config["shuffle_files"] = dl_config["shuffle_files"]["value"]
    config["as_supervised"] = dl_config["as_supervised"]["value"]
    config["dense"] = dl_config["dense"]["value"]
    config["norm_type"] = dl_config["norm_type"]["value"]
    config["apply_weight_balancing"] = dl_config["apply_weight_balancing"]["value"]
    config["loss_function"] = dl_config["loss_function"]["value"]
    config["use_sam"] = dl_config["use_sam"]["value"]


def define_pretrained_ml_model_config():
    if config["clinic_filter"] == "heart_thorax":
        dl_config = yaml.load(
            open("./app/config/model_heart_thorax/rf/base_config.yaml"),
            Loader=yaml.FullLoader,
        )
    elif config["clinic_filter"] == "hematooncology":
        dl_config = yaml.load(
            open("./app/config/model_hematooncology/rf/base_config.yaml"),
            Loader=yaml.FullLoader,
        )
    elif config["clinic_filter"] == "none":
        dl_config = yaml.load(
            open("./app/config/model_none/rf/base_config.yaml"), Loader=yaml.FullLoader
        )
    else:
        raise ValueError("Wrong clinic_filter value")

    config["booster"] = dl_config["booster"]["value"]
    config["gamma"] = dl_config["gamma"]["value"]
    config["grow_policy"] = dl_config["grow_policy"]["value"]
    config["learning_rate"] = dl_config["learning_rate"]["value"]
    config["max_depth"] = dl_config["max_depth"]["value"]
    config["reg_alpha"] = dl_config["reg_alpha"]["value"]
    config["reg_lambda"] = dl_config["reg_lambda"]["value"]
    config["sampling_method"] = dl_config["sampling_method"]["value"]
    config["model_path"] = Path(
        "./app/config/model_" + config["clinic_filter"] + "/rf/base_model.pkl"
    )


if __name__ == "__main__":
    args = parse_args_nn()

    if config["is_live_prediction"]:
        config = yaml.safe_load(
            (Path.cwd() / "app/config/config_live_prediction.yaml").open()
        )
        parse_args_nn()
        live_pipeline(init=True)

    else:
        define_ds_name(args.command)
        if args.command == "ml":
            define_pretrained_ml_model_config()
            if config["explain_rf"]:
                explain_train.main(config)
            else:
                launch_ml_training()
        elif args.command == "dl":
            define_pretrained_dl_model_config()
            if config["launch_training"]:
                nn_training_pipeline()
            if config["launch_publish"]:
                publish_pipline()
            if config["explain_lstm"]:
                explain_rnn_pipline()
