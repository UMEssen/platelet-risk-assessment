from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt, cm
from scipy.ndimage import uniform_filter1d

from app.ml.explain_train_nn_lime import (
    assign_feature_names,
    plot_feature_contributions,
)

LOOKUP = {
    "heart_thorax": "Cardio-Thoracic Surgery Patient Model",
    "hematooncology": "Hematology Oncology Patient Model",
    "none": "All Other Clinics and Wards Model",
}

with open("app/config/constants.yaml", "r") as stream:
    CONSTANTS = yaml.safe_load(stream)


def plot_lines_sequential(
    config,
    top_positive_features_all: pd.DataFrame,
    ax,
    annotation_text: Optional[str] = None,
):
    print(top_positive_features_all.head())
    df = top_positive_features_all
    df.columns = ["values"]

    def extract_number(feature_name):
        return int("".join(filter(str.isdigit, feature_name)))

    features_to_summarize = [
        "weekday_cos",
        "weekday_sin",
        "zTHRA",
        "is_holiday",
        "is_weekend",
        "TK",
        "is_real_zTHRA",
    ]

    feature_dict = {}

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
    if "weekday_cos" in feature_dict and "weekday_sin" in feature_dict:
        df_weekday_sin_values = [
            df.loc[feature, "values"] for feature in feature_dict["weekday_sin"][0]
        ]
        df_weekday_cos_values = [
            df.loc[feature, "values"] for feature in feature_dict["weekday_cos"][0]
        ]

        weekday_values = np.array(df_weekday_sin_values) + np.array(
            df_weekday_cos_values
        )
        weekday_features = [
            "weekday_" + str(i) for i in range(1, len(weekday_values) + 1)
        ]
        feature_dict["weekday"] = (weekday_features, weekday_values)

    feature_dict.pop("weekday_cos", None)
    feature_dict.pop("weekday_sin", None)

    # generate color map
    num_features = len(feature_dict)
    colors = cm.Greys(np.linspace(0.3, 1, num_features))

    for i, (category, (features, values)) in enumerate(feature_dict.items()):
        log_values = np.log(values + 1e-5)
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
    ax.set_title(f"Sequential Features on {LOOKUP[config['clinic_filter']]}")
    ax.legend()
    ax.grid(True)

    return ax


def plot_rf_explainability(config):
    adjusted_feature_names = assign_feature_names(CONSTANTS["FEATURE_MAPPING"])

    # Initialize subplot grid
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    # fig.suptitle("Supplementary Figure 2", fontsize=13, x=0, ha="left")
    # Model configurations

    # annot_letters = ["A", "B", "C", "D", "E", "F"]
    annot_letters = ["A", "B", "C"]
    a = 0
    for m, (clinic_filter, clinic_filter_name) in enumerate(LOOKUP.items()):
        # model = pd.read_pickle(Path('app/config/model_' / clinic_filter / 'rf/base_model.pkl'))
        path = f"app/config/model_{clinic_filter}" / Path("rf/base_model.pkl")
        model = pd.read_pickle(path)
        print(path)
        model = pd.read_pickle("app/config/model_none/rf/base_model.pkl")
        imp = model.get_booster().get_score(importance_type="weight")

        print(imp.keys())
        print(len(imp.keys()))

        named_imp = {}

        for count, (key, values) in enumerate(imp.items()):
            named_imp[adjusted_feature_names[count]] = values

        print(named_imp.keys())
        print(len(named_imp.keys()))

        ungrouped_importances_df = pd.DataFrame.from_dict(
            named_imp, orient="index", columns=["importance"]
        )
        # group features that start with 'weekday', 'is_holiday', 'is_weekend', 'is_real_zTHRA'
        filter_group = [
            "weekday",
            "is_holiday",
            "is_weekend",
            "is_real_zTHRA",
            "TK",
            "zTHRA",
        ]
        # Create a new DataFrame with just the columns you're interested in
        grouped_importances = {group: 0 for group in filter_group}

        print(ungrouped_importances_df.tail())
        print(len(ungrouped_importances_df.columns))

        for group in filter_group:
            for feature in ungrouped_importances_df.index:
                if feature.startswith(group):
                    grouped_importances[group] += ungrouped_importances_df.loc[
                        feature, "importance"
                    ]

        # devide by 60 to get the average importance
        grouped_importances = {
            group: grouped_importances[group] / 60 for group in filter_group
        }

        # filter out the features that start with 'weekday', 'is_holiday', 'is_weekend', 'is_real_zTHRA'
        grouped_importances_df = ungrouped_importances_df[
            ~ungrouped_importances_df.index.str.startswith(tuple(filter_group))
        ]
        # add the grouped features to the DataFrame
        grouped_importances_df = grouped_importances_df.append(
            pd.DataFrame.from_dict(
                grouped_importances, orient="index", columns=["importance"]
            )
        )
        # sort the df by value descending
        grouped_importances_df = grouped_importances_df.sort_values(
            by="importance", ascending=False
        )
        print(grouped_importances_df.to_dict()["importance"])

        plot_feature_contributions(
            top_positive_features=grouped_importances_df.to_dict()["importance"],
            top_negative_features=None,
            ax=axs[m],
            annotation_text=annot_letters[a],
            m=m,
            title=clinic_filter_name,
            set_lim=False,
        )
        a += 1

        # plot_lines_sequential(
        #     config,
        #     ungrouped_importances_df,
        #     axs[m, 1],
        #     annot_letters[a],
        # )
        # a += 1

        # plot lines generic ....

    # Save the combined plot
    file_name = (
        str(Path(config["root_dir"]) / "xai_lstm" / "plots" / "combined_plot")
        + "_"
        + datetime.now().strftime("%Y%m%d-%H%M")
        + ".png"
    )
    fig.tight_layout()
    plt.savefig(
        file_name,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def main(config):
    plot_rf_explainability(config)
