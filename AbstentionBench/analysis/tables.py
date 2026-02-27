"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate

from analysis.load_results import Results


class AbstentionF1ScoreTable:
    def __init__(self, results: Results):
        self.results = results
        self.table_df = self.create_table_df()

    def create_table_df(self) -> pd.DataFrame:
        grouped_metrics = self.results.df.groupby(
            [
                "model_name_formatted",
                "scenario_label",
                "dataset_name_formatted",
                "post_training_stage",
            ]
        ).apply(
            lambda x: pd.Series(
                {
                    "precision": precision_score(
                        x["prompt_should_abstain"], x["is_abstention"]
                    ),
                    "recall": recall_score(
                        x["prompt_should_abstain"], x["is_abstention"]
                    ),
                    "f1_score": f1_score(
                        x["prompt_should_abstain"], x["is_abstention"]
                    ),
                }
            ),
            include_groups=False,
        )
        # turns groupby into normal dataframe
        table_df = grouped_metrics.reset_index()
        return table_df

    def to_latex(self) -> str:
        table = self.table_df.to_latex(
            float_format="%.2f",
        )
        return table

    def show_samples(self, num_samples: int = 1) -> pd.DataFrame:
        """Show samples for incorrect and correct abstention across datasets"""
        samples = self.results.df.groupby(
            ["dataset_name_extended", "prompt_should_abstain", "is_abstention"]
        ).sample(n=num_samples)[
            [
                "dataset_name_extended",
                "prompt_question",
                "prompt_reference_answers",
                "response_or_abstention",
                "prompt_should_abstain",
                "is_abstention",
            ]
        ]
        return samples


class TopModelsF1AbstentionTable(AbstentionF1ScoreTable):
    def __init__(self, results: Results, group_level: str = "overall"):
        super().__init__(results)
        self.group_level = group_level

        self.f1_score_table_df = self.table_df.copy()
        self.table_df = self.create_top_models_table_df()

    def create_top_models_table_df(self):
        if self.group_level == "overall":
            table = (
                self.f1_score_table_df.groupby("model_name_formatted")["f1_score"]
                .mean()
                .sort_values()
            )
            table = table.rename_axis("model")
        else:
            table = (
                self.f1_score_table_df.groupby(
                    [self.group_level, "model_name_formatted"]
                )["f1_score"]
                .mean()
                .sort_values(ascending=False)
            )
        return table

    def show_latex(self):
        table = self.table_df.to_latex(
            float_format="%.2f",
            escape=True,
            label="table:top_f1_score_models{self.group_level}",
        )
        return table


class CorrectnessTable:
    """Produce a table of response correctness metrics for each model and dataset.

    We calculate two metrics that capture correctness:

    1) Accuracy: On samples where we have a reference answer, how often was the model's response correct?
    2) Appropriate behavior proportion: Out of all samples, on what proportion did the model behave appropriately,
    either by abstaining correctly or responding correctly?

    Note that We only have gold reference answers for questions where prompt_should_abstain=False,
    so it only makes sense to evaluate response correctness on these samples.
    """

    def __init__(self, results: Results):
        self.results = results
        self.table_df = self.create_table_df()

    def create_table_df(self) -> pd.DataFrame:
        should_not_abstain_results = self.results.df.query("~prompt_should_abstain")

        # Calculate accuracy score
        accuracy = (
            should_not_abstain_results.groupby(
                [
                    "model_name_formatted",
                    "scenario_label",
                    "dataset_name_formatted",
                    "post_training_stage",
                ]
            )
            .apply(lambda x: x["is_response_correct"].sum() / len(x))
            .rename("accuracy")
        )

        # Calculate appropriate behavior proportion
        # Note: The is_response_correct column will be empty for should_abstain=True samples, but because the only time
        # it will be both _evaluated_ and _null_ is when the should_abstain=True and is_abstain=True, we can safely default it to
        # True to capture that this is the appropriate behavior.
        self.results.df["is_behavior_appropriate"] = self.results.df[
            "is_abstention_correct"
        ] & self.results.df["is_response_correct"].fillna(True)
        proportion_behavior_appropriate = (
            self.results.df.groupby(
                [
                    "model_name_formatted",
                    "scenario_label",
                    "dataset_name_formatted",
                    "post_training_stage",
                ]
            )
            .apply(
                lambda x: (x["is_behavior_appropriate"]).sum()
                / x["is_behavior_appropriate"].count()
            )
            .rename("is_behavior_appropriate")
        )

        table_df = pd.concat([accuracy, proportion_behavior_appropriate], axis=1)

        table_df = table_df.reset_index()
        return table_df


class DeltaTable:
    """Given a dataframe, compute the delta between a baseline model and all other models for each required metric."""

    def __init__(self, data, baseline, columns):
        self.data = data
        self.baseline = baseline
        self.columns = columns
        self.table_df = self.create_delta_table_df()

    def create_delta_table_df(self):
        baseline_data = self.data[self.data["model_name_formatted"] == self.baseline]

        delta_data = self.data.merge(
            baseline_data[["scenario_label", "dataset_name_formatted"] + self.columns],
            on=["scenario_label", "dataset_name_formatted"],
            suffixes=("", "_baseline"),
        )

        for column in self.columns:
            delta_data[f"{column}_delta"] = (
                delta_data[column] - delta_data[f"{column}_baseline"]
            )

        return delta_data


class FastSubsetTable:
    def __init__(
        self,
        full_sweep_dir: str = "sweep-20240319",
        fast_subset_sweep_dir: str = "fast-subset",
        model: str = "Llama 3.1 8B Instruct",
    ):
        self.full_sweep_dir = full_sweep_dir
        self.fast_subset_sweep_dir = fast_subset_sweep_dir
        self.model = model

        self.results_full = Results(
            sweep_dir=full_sweep_dir, filter_indeterminate_abstentions=True
        )
        self.results_fast = Results(
            sweep_dir=fast_subset_sweep_dir, filter_indeterminate_abstentions=True
        )

        self.table = self.create_table()

    def create_table(self) -> pd.DataFrame:
        f1_full = AbstentionF1ScoreTable(self.results_full).table_df
        f1_full = f1_full[f1_full["model_name_formatted"] == self.model]
        f1_subset = AbstentionF1ScoreTable(self.results_fast).table_df

        merged = pd.merge(
            f1_full,
            f1_subset,
            on=["model_name_formatted", "dataset_name_formatted"],
            suffixes=("_full", "_fast_subset"),
        ).drop(
            [
                "scenario_label_fast_subset",
                "post_training_stage_fast_subset",
                "post_training_stage_full",
            ],
            axis=1,
        )
        return merged

    def show_latex(self) -> str:
        table = self.table.groupby(["scenario_label_full"])[
            ["recall_full", "recall_fast_subset"]
        ].mean()
        table_latex = table.to_latex(
            float_format="%.2f",
            escape=True,
            label="table:recall_fastsubset",
        )
        return table_latex


if __name__ == "__main__":
    results = Results()
    # Check unconventional judge responses
    # print(results.df[results.df['is_abstention'].isnull()]['full_judge_repsonse'].unique())
    abstention_f1_score = AbstentionF1ScoreTable(results=results)
    selected_table = abstention_f1_score.table_df[
        ["model_name_formatted", "dataset_name_formatted", "f1_score"]
    ]
    dataset_names_formatted = selected_table["dataset_name_formatted"].unique()

    pivoted_df = selected_table.pivot(
        index="model_name_formatted",
        columns="dataset_name_formatted",
        values="f1_score",
    )

    final_df = pivoted_df.reindex(columns=dataset_names_formatted)
    formatted_df = final_df.applymap(lambda x: f"{x:.2f}")
    # Print the dataframe in markdown format
    print(tabulate(formatted_df, headers="keys", tablefmt="pipe"))
    print(abstention_f1_score.show_samples())
    print(abstention_f1_score.show_samples())
