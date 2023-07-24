import ast
import csv
import glob
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import tensorflow as tf
import yaml

from lime import lime_tabular
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib import cm
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import itertools


from app.data_handling.get_training_data import AutopilotDataset
from app.ml.train_helper import get_dataset

logger = logging.getLogger(__name__)

with open("app/config/constants.yaml", "r") as stream:
    CONSTANTS = yaml.safe_load(stream)


LOOKUP = {
    "heart_thorax": "Cardio-Thoracic Surgery Patient Model",
    "hematooncology": "Hematology Oncology Patient Model",
    "none": "All Other Clinics and Wards Model",
}


def plot_feature_contributions(
    top_positive_features: Dict,
    top_negative_features: Optional[Dict] = None,
    ax: plt.Axes = None,
    annotation_text: Optional[str] = None,
    set_lim: bool = True,
    m: int = 2,
    title: Optional[str] = None,
) -> None:
    grouped_features = [
        "zTHRA",
        "TK",
        "is_real_zTHRA",
        "is_weekend",
        "is_holiday",
        "weekday",
    ]

    # Extract feature names and contributions for positive features
    positive_features = list(top_positive_features.keys())
    positive_contributions = list(top_positive_features.values())

    if top_negative_features not in [None, {}]:
        # Extract feature names and contributions for negative features
        negative_features = list(top_negative_features.keys())
        negative_contributions = list(top_negative_features.values())

    # Create y-axis values for the bar plot
    y_pos = (
        range(len(positive_features) + len(negative_features))
        if top_negative_features not in [None, {}]
        else range(len(positive_features))
    )

    # Plot positive contributions
    colors_pos = [
        "#EE8080" if feature not in grouped_features else "#9CA0A6"
        for feature in positive_features
    ]
    ax.barh(
        y_pos[: len(positive_features)],
        positive_contributions,
        align="center",
        color=colors_pos,
    )

    if top_negative_features not in [None, {}]:
        logging.info("Plotting also negative features")
        # Plot negative contributions
        colors_neg = [
            "#9CA0A6" if feature not in grouped_features else "#EE8080"
            for feature in negative_features
        ]
        ax.barh(
            y_pos[len(positive_features) :],
            negative_contributions,
            align="center",
            color=colors_neg,
        )

    if m == 2:
        # Make legend explaining that #EE8080 is for grouped features and #FE787C for non-grouped features
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color="#9CA0A6", label="Sequential Features"),
            plt.Rectangle((0, 0), 1, 1, color="#EE8080", label="Categorical Features"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        positive_features + negative_features
        if top_negative_features not in [None, {}]
        else positive_features
    )
    if annotation_text is not None:
        ax.annotate(
            annotation_text,
            xy=(0, 1.1),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
        )

    ax.invert_yaxis()  # Invert the y-axis to show higher contributions at the top
    ax.set_xlabel("Feature Importance")

    ax.set_title(f"Most Relevant Features for {title}")
    if set_lim:
        ax.set_xlim(xmin=0, xmax=1.05)

    return ax


def normalize_range(feature_contributions: Dict) -> Dict:
    """Scale contributions using feature scaling

    Args:
        feature_contributions (Dict): Dictionary containing the feature contributions

    Returns:
        Dict: Dictionary containing the scaled feature contributions
    """
    # Get the minimum and maximum contribution
    min_contribution = min(feature_contributions.values())
    max_contribution = max(feature_contributions.values())

    # Scale the contributions using feature scaling
    for feature in feature_contributions:
        scaled_contribution = (feature_contributions[feature] - min_contribution) / (
            max_contribution - min_contribution
        )
        feature_contributions[feature] = scaled_contribution

    return feature_contributions


def _check_repeated_feature_names(feature_names: Dict) -> Dict:
    """As 7 features are represented at multiple points in time, after flattening they are represented
    redundantly. This function checks if there are any repeated feature names and if so,
    it assigns them a suffix with _1, _2, etc.

    Args:
        feature_names ([Dict]): feature names dict without suffixes

    Returns:
        [Dict]: feature names dict with suffixes
    """
    feature_names_dict = {}
    for i, feature_name in enumerate(feature_names):
        if feature_name in feature_names_dict:
            feature_names[i] = (
                feature_name + "_" + str(feature_names_dict[feature_name])
            )
            feature_names_dict[feature_name] += 1
        else:
            feature_names_dict[feature_name] = 1

    return feature_names


def assign_feature_names(FEATURE_MAPPING: Dict) -> Dict:
    """Assign feature names to the feature contributions, matching the flattened shape of the data

    Args:
        FEATURE_MAPPING (Dict): Dictionary containing the feature mapping

    Returns:
        Dict: Adjusted Dictionary containing the feature names
    """
    feature_names = []
    num_features = len(FEATURE_MAPPING)

    # Assign feature names for the first 7 features (replicated 60 times), so first 420 indices
    for i in range(7):
        feature_name = FEATURE_MAPPING[str(i)]
        feature_names.extend([feature_name] * 60)

    # Assign feature names for the remaining features where index > 420
    for i in range(7, num_features):
        feature_name = FEATURE_MAPPING[str(i)]
        feature_names.append(feature_name)

    # Check if feature names are redundant and if so, assign them a suffix with _1, _2, etc.
    feature_names = _check_repeated_feature_names(feature_names)

    return feature_names


def get_top_features(
    df: pd.DataFrame,
    positive_contribution: bool,
    positive_only: bool = False,
) -> List[Dict]:
    """AI is creating summary for get_top_features

    Args:
        df (pd.DataFrame): DataFrame containing all feature contributions resulting from LIME
        positive_contribution (bool): If True, positive contributions are considered, otherwise negative contributions
        positive_only (bool, optional): If True, only positive contributions are considered by making all values absolute and removing minuses. Defaults to False.

    Returns:
        List[Dict]: List of dictionaries containing the top features and their contributions
    """

    top_features_list = []

    # Iterate over all rows in the DataFrame
    for index, row in df.iterrows():
        feature_contributions = {}

        # Iterate over all items in the row
        for item in row:
            tuple_str = item.strip()
            tuple_data = ast.literal_eval(tuple_str)

            feature_match = re.search(r"(?:[0-9.-]+\s*<\s*)?([^\s<>]+)", tuple_data[0])
            if feature_match:
                feature = feature_match.group(1)
                contribution = tuple_data[1]

                # If positive_only is True, make all values absolute to remove minuses
                if positive_only:
                    contribution = abs(contribution)

                feature_contributions[feature] = contribution

        sorted_features = sorted(
            feature_contributions,
            key=feature_contributions.get,
            reverse=positive_contribution,
        )

        # if positive_contribution is True, remove all negative contributions as sorting alone does not guarantee that positive contributions are at the top
        if positive_contribution:
            sorted_features = [
                feature
                for feature in sorted_features
                if feature_contributions[feature] > 0
            ]
        # Also, if positive_contribution is False, remove all positive contributions
        else:
            sorted_features = [
                feature
                for feature in sorted_features
                if feature_contributions[feature] < 0
            ]

        top_features = {
            str(feature): np.round(feature_contributions[feature], 3)
            for feature in sorted_features
        }
        top_features_list.append(top_features)

    return top_features_list


def list_to_csv(my_list, file_path):
    """Writes a list to a csv file

    Args:
        my_list ([list]): List to be written to csv
        file_path ([string]): Path to the csv file
    """
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(my_list)


def split_into_ranges(data, num_ranges):
    min_value = 0
    max_value = len(data)
    logging.info(min_value)
    logging.info(max_value)
    range_width = (max_value - min_value) / num_ranges

    ranges = []
    for i in range(num_ranges):
        lower_bound = min_value + (i * range_width)
        upper_bound = min_value + ((i + 1) * range_width)
        ranges.append((int(lower_bound), int(upper_bound)))

    return ranges


def explain_nn(
    config: Dict, train_data_list: List, labels: List, save_tsv: bool = False
) -> None:
    """Function to explain the predictions of a given neural network model using LIME explanations

    Args:
        config ([Dict]): config dict

    Raises:
        ValueError: Is raised if the train_data_list is None
        ValueError: Is raised if the model is None

    Returns:
        None
    """
    model = tf.keras.models.load_model(
        Path("app/config/model_" + config["clinic_filter"] + "/ensemble_model.h5")
    )

    # The structure of the data is as follows:
    # train_data_list[0]: MT_FEATURES (x, 60, 7)
    # train_data_list[1]: CLINIC_FEATURES (x, 16, 2)
    # train_data_list[2]: LOCATION_FEATURES (x, 2)
    # labels: binary labels (x)
    labels = labels.T
    for data in train_data_list:
        logging.info(data.shape)

    logging.info("Labels shape: ", labels.shape)

    if train_data_list is None:
        raise ValueError("train_data_list is None")
    if model is None:
        raise ValueError("Model is not loaded correctly")

    # Batch size is irrelevant -> (60, 7), (16,), (2,)]
    original_shapes = [data.shape[1:] for data in train_data_list]

    # Reshape your training data if needed. The format expected by LIME is a 2D array-like object.
    # (x, 420) / (x, 16) / (x , 2)
    train_data_list = [data.reshape(data.shape[0], -1) for data in train_data_list]
    # (x, 438)
    train_data = np.concatenate(train_data_list, axis=1)

    def custom_predict(input_data: np.ndarray) -> np.ndarray:
        """Custom predict function for LIME explainer

        # LIME explanation
        explainer = lime_tabular.LimeTabularExplainer(train_data, mode="regression", training_labels=labels)
        Args:
            input_data ([np.ndarray]): Input data to be predicted

        Returns:
            [np.ndarray]: Predicted output
        """
        split_points = np.cumsum([np.product(shape) for shape in original_shapes])
        input_data_split = np.hsplit(input_data, split_points)
        input_data_reshaped = [
            input_data_part.reshape((-1,) + original_shape)
            for input_data_part, original_shape in zip(
                input_data_split, original_shapes
            )
        ]

        return model.predict(input_data_reshaped)

    logging.info("Train data shape: ", train_data.shape)

    adjusted_feature_names = assign_feature_names(CONSTANTS["FEATURE_MAPPING"])

    # LIME explainer object
    explainer = lime_tabular.LimeTabularExplainer(
        train_data,
        mode="regression",
        feature_names=adjusted_feature_names,
        random_state=42,
        verbose=True,
        kernel_width=10,  # usually sqrt(438)*0.75 = 15.6
    )  # Regression as workaround for labels not being one-hot encoded

    explanations = []
    # y = [range(0,10), range(10,20)]
    num_ranges = split_into_ranges(train_data, 20)
    i = 0
    for num_range in tqdm(num_ranges):
        lower_bound, upper_bound = num_range

        pkl_file_path = f"{config['explanation_dir']}/explanations_{i}.pkl"
        # If this file already exists, skip this iteration.
        if Path(pkl_file_path).exists():
            i += 1
            continue

        for count, instance in enumerate(train_data[lower_bound:upper_bound]):
            logging.info(f"Calculating {count} sample on batch {i}")
            exp = explainer.explain_instance(
                instance, custom_predict, num_features=instance.shape[0]
            )
            explanations.append(exp.as_list())

        with open(pkl_file_path, "wb") as file:
            pickle.dump(explanations, file)
        i += 1

    explanation_files = glob.glob(
        os.path.join(config["explanation_dir"], "explanations_*.pkl")
    )

    # Generates a tsv file with all the explanations
    if save_tsv:
        explanations_df = pd.DataFrame()
        for file_path in explanation_files:
            with open(file_path, "rb") as file:
                explanation = pickle.load(file)
                df_exp = pd.DataFrame(explanation)
                explanations_df = pd.concat([explanations_df, df_exp])
        df_exp.to_csv(
            f"{config['explanation_dir']}/explanations_sum.tsv", index=False, sep="\t"
        )


def plot_lines_weekday(
    config,
    top_positive_features_all: pd.DataFrame,
    ax,
    annotation_text: Optional[str] = None,
):
    # Function to extract number from the feature name
    print(top_positive_features_all.head())
    df = top_positive_features_all
    df.columns = ["values"]

    def extract_number(feature_name):
        return int("".join(filter(str.isdigit, feature_name)))

    # List of features to be summarized
    features_to_summarize = ["weekday_cos", "weekday_sin"]

    # Dictionary to store features and their values for each category
    feature_dict = {}

    # Group features by category and extract their values
    for category in features_to_summarize:
        features_in_category = [
            col
            for col in df.index
            if category in col and any(char.isdigit() for char in col)
        ]
        features_in_category_sorted = sorted(features_in_category, key=extract_number)
        values = df.loc[features_in_category_sorted, "values"].values
        feature_dict[category] = (features_in_category_sorted, values)

    # Summarize weekday_cos and weekday_sin
    df_weekday_sin_values = [
        df.loc[feature, "values"]
        for feature in feature_dict[features_to_summarize[0]][0]
    ]
    df_weekday_cos_values = [
        df.loc[feature, "values"]
        for feature in feature_dict[features_to_summarize[1]][0]
    ]

    weekday_values = np.array(df_weekday_sin_values) + np.array(df_weekday_cos_values)
    feature_dict = {}
    weekday_features = ["weekday_" + str(i) for i in range(1, len(weekday_values) + 1)]
    feature_dict["weekday"] = (weekday_features, weekday_values)

    # remove weekday_sin from dict
    feature_dict.pop("weekday_sin", None)
    feature_dict.pop("weekday_cos", None)

    # generate color map
    num_features = len(feature_dict)
    colors = cm.Greys(np.linspace(0.3, 1, num_features))

    # Assign different color for each category and apply moving average
    for i, (category, (features, values)) in enumerate(feature_dict.items()):
        # Add a small constant to values before applying log to avoid log(0)
        log_values = np.log(values + 1e-5)
        # Apply moving average
        smoothed_values = uniform_filter1d(log_values, size=5)
        ax.plot(
            [extract_number(f) for f in features],
            smoothed_values,
            label=category,
            color=colors[i],
        )

    if annotation_text is not None:
        ax.annotate(
            annotation_text,
            xy=(0, 1.1),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
        )

    ax.set(
        xlabel="Time",
        ylabel="Feature Importance (Log Scale)",
    )
    ax.set_title(f"Weekday Feature on {LOOKUP[config['clinic_filter']]}")
    ax.legend()
    ax.grid(True)

    return ax


def combined_plot(config):
    # Initialize subplot grid
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle("Figure 6", fontsize=13, x=0, ha="left")
    # Model configurations

    annot_letters = ["A", "B", "C"]
    # annot_letters = ["A", "B", "C", "D", "E", "F"]
    a = 0
    for m, (clinic_filter, clinic_filter_name) in enumerate(LOOKUP.items()):
        top_positive_features_all_grouped = pd.read_csv(
            f"app/top_positive_features_all_sum_{clinic_filter}_grouped.csv"
        ).to_dict(orient="records")[0]
        top_positive_features_all_sequential = pd.read_csv(
            f"app/top_positive_features_all_sum_{clinic_filter}_sequential.csv"
        )

        print(clinic_filter)
        print(clinic_filter_name)

        plot_feature_contributions(
            top_positive_features=dict(
                itertools.islice(top_positive_features_all_grouped.items(), 15)
            ),
            top_negative_features=None,
            ax=axs[m],
            annotation_text=annot_letters[a],
            m=m,
            title=clinic_filter_name,
        )

        a += 1
        # plot_lines_weekday(
        #     config, top_positive_features_all_sequential, axs[m, 1], annot_letters[a]
        # )
        # a += 1

    # Save the combined plot
    file_name = (
        str(Path(config["root_dir"]) / "xai_lstm" / "plots" / "combined_plot")
        + "_"
        + datetime.now().strftime("%Y%m%d-%H%M")
        + ".pdf"
    )
    fig.tight_layout()
    plt.savefig(
        file_name,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def run_analysis(config, df_exp: pd.DataFrame) -> None:
    if not tf.config.list_physical_devices("GPU"):
        raise ValueError("GPU not available")
        sys.exit(1)

    if (
        not Path(
            f"app/top_positive_features_all_sum_{config['clinic_filter']}_grouped.csv"
        ).exists()
        or not Path(
            f"app/top_positive_features_all_sum_{config['clinic_filter']}_sequential.csv"
        ).exists()
    ):
        print("Generating top positive features")

        # Run once with summarize_sequential_features and once without
        runs = {"grouped": True, "sequential": False}
        for kind, is_grouped in runs.items():
            top_positive_features_per_samle = get_top_features(
                df_exp, positive_contribution=True
            )
            print(is_grouped)
            top_positive_features_all = summarize_feature_contributions(
                feature_contributions_list=top_positive_features_per_samle,
                number_of_features=-1,
                summarize_sequential_features=is_grouped,
            )
            top_positive_features_all = normalize_range(
                feature_contributions=top_positive_features_all
            )
            print_feature_contributions(top_positive_features_all, "positive")
            df = pd.DataFrame(top_positive_features_all, index=range(1))
            df.reset_index(drop=True).to_csv(
                f"app/top_positive_features_all_sum_{config['clinic_filter']}_{kind}.csv",
                index=False,
            )

    model_names = LOOKUP.keys()
    # only do this if all the model name files exist
    if all(
        Path(f"app/top_positive_features_all_sum_{model_name}_sequential.csv").exists()
        for model_name in model_names
    ):
        combined_plot(config)


def summarize_feature_contributions(
    feature_contributions_list: List[Dict],
    number_of_features: int = 5,
    summarize_sequential_features: bool = False,
) -> Dict:
    feature_summary = {}

    # Define the list of features to be summarized
    features_to_summarize = [
        "zTHRA",
        "TK",
        "is_real_zTHRA",
        "is_weekend",
        "is_holiday",
        "weekday",
    ]

    # Iterate over each dictionary in the list
    for feature_dict in feature_contributions_list:
        # Iterate over each feature and its contribution in the dictionary
        for feature, contribution in feature_dict.items():
            if summarize_sequential_features and any(
                feature.startswith(prefix) for prefix in features_to_summarize
            ):
                summarized_feature = next(
                    prefix
                    for prefix in features_to_summarize
                    if feature.startswith(prefix)
                )

                if summarized_feature not in feature_summary:
                    feature_summary[summarized_feature] = []

                feature_summary[summarized_feature].append(contribution)
            else:
                if feature not in feature_summary:
                    feature_summary[feature] = []

                feature_summary[feature].append(contribution)

    for feature, contributions in feature_summary.items():
        avg_contribution = round(sum(contributions) / len(contributions), 3)
        feature_summary[feature] = avg_contribution

    # Calculate the average contribution for each feature

    # Sort the dictionary by the absolute value of the contribution
    feature_summary = dict(
        sorted(feature_summary.items(), key=lambda item: abs(item[1]), reverse=True)[
            :number_of_features
        ]
    )

    print(feature_summary)
    return feature_summary


def print_feature_contributions(
    feature_contributions: Dict, contribution_type: str
) -> None:
    logging.info(
        f"Top five features with {contribution_type} contribution across all samples: "
    )
    for feature in feature_contributions:
        logging.info(f"{feature}: {feature_contributions[feature]}")


def load_data(config):
    if (Path(config["explanation_dir"]) / Path("test_data_array_list")).exists():
        with open(
            (Path(config["explanation_dir"]) / Path("test_data_array_list")),
            "rb",
        ) as file:
            return pickle.load(file)

    else:
        ds = AutopilotDataset(data_dir=config["dataset_dir"], config=config)
        ds.download_and_prepare()

    # In case you want to load the train vali split
    # train_splits, val_splits = return_train_validation_cv_split(
    #     cv_idx=1, cv_folds=5
    # )
    # train_data = get_dataset(train_splits, config)

    test_data = get_dataset("test", config)

    # Optional: Compute the number of samples to take
    # num_samples = tf.data.experimental.cardinality(test_data).numpy()
    # num_samples_10_percent = int(num_samples * 0.01)
    # print("samples: ", num_samples)
    # test_data_sample = test_data.take(num_samples_10_percent)

    # Initialize an empty list for each part of the tuple
    test_data_list_1 = []
    test_data_list_2 = []
    test_data_list_3 = []
    labels_list = []

    # Iterate over the TensorFlow Dataset and append each item to the respective list
    for item in test_data:
        (test_data_1, test_data_2, test_data_3), labels = item
        test_data_list_1.append(test_data_1.numpy())
        test_data_list_2.append(test_data_2.numpy())
        test_data_list_3.append(test_data_3.numpy())
        labels_list.append(labels.numpy())

    test_data_array_1 = np.concatenate(test_data_list_1)
    test_data_array_2 = np.concatenate(test_data_list_2)
    test_data_array_3 = np.concatenate(test_data_list_3)
    labels_array = np.concatenate(labels_list)

    # Combine the data arrays into a single list
    test_data_list = [test_data_array_1, test_data_array_2, test_data_array_3]

    # # Convert the list to a CSV file
    # # Save the list of arrays
    # with open(f"{config['explanation_dir']}/test_data_array_list.pkl", "wb") as file:
    #     pickle.dump((test_data_list, labels_array), file)

    return test_data_list, labels_array


def reduce_data(test_data_list, labels, numb_samples=10):
    np.random.seed(42)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)  # in-place shuffle of the indices
    train_data_list0 = [test_data_list[0][i] for i in indices]
    train_data_list1 = [test_data_list[1][i] for i in indices]
    train_data_list2 = [test_data_list[2][i] for i in indices]
    x = slice(0, numb_samples)
    test_data_list = [
        np.array(train_data_list0[x]),
        np.array(train_data_list1[x]),
        np.array(train_data_list2[x]),
    ]
    labels = [labels[i] for i in indices]
    labels = np.array(labels)
    labels = labels[x]
    return test_data_list, labels


def explainer(config: Dict):
    config["explanation_dir"] = config["explanation_dir"] / Path(
        config["clinic_filter"]
    )
    if not Path(Path(config["explanation_dir"]) / "explanations_sum.tsv").exists():
        logging.info("Explanation summary file does not exist. Creating it now.")
        test_data_list, labels = load_data(config)

        # test_data_list, labels = reduce_data(test_data_list, labels, numb_samples=10)

        logging.info(f"test_data_list: {test_data_list[0][0].shape}")
        logging.info(f"test_data_list: {type(test_data_list)}")
        logging.info(f"test_data_list: {type(test_data_list[0])}")
        explain_nn(config, test_data_list, labels, True)

    logging.info("Loading explanations_sum.tsv")
    df_exp = pd.read_csv(
        Path(config["explanation_dir"]) / "explanations_sum.tsv", sep="\t"
    )

    run_analysis(df_exp=df_exp, config=config)


def main(config):
    explainer(config)
