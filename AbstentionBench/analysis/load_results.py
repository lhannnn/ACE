"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import logging
import os
from pathlib import Path

import pandas as pd

from recipe.abstention import Responses
from recipe.job_manager import JobManager

logger = logging.getLogger(__name__)


class Results:
    """Loads results.
    By default uses the paths specified in
        analysis/results_path.py

    Attributes:
        df: pandas dataframe that turns saved results into a table
        final_file: json filename with the aggregated responses + evaluations
    """

    def __init__(
        self,
        base_results_dir: str = None,
        sweep_dir: str = "sweep-20240319",
        result_path_names: list[str] | None = None,
        final_file: str = "GroundTruthAbstentionEvaluator.json",
        filter_indeterminate_abstentions=False,
        filter_indeterminate_correctness=False,
        allow_missing_correctness_files=False,
        format=True,
    ):
        self.base_results_dir = base_results_dir
        self.sweep_dir = sweep_dir
        self.final_file = final_file
        self.result_path_names = (
            self.get_results_path_names()
            if result_path_names is None
            else result_path_names
        )

        self.result_file_paths = self.get_result_file_paths()
        self.raw_response_dicts = self.load(
            allow_missing_correctness_files=allow_missing_correctness_files
        )
        self.df = self.to_dataframe()

        self.df = self.filter_data(self.df)
        if format:
            self.df = self.format_data(self.df)

        if filter_indeterminate_abstentions:
            self.df = self.filter_indeterminate_abstentions(self.df)

        if filter_indeterminate_correctness:
            self.df = self.filter_indeterminate_correctness(self.df)

    def get_results_path_names(self) -> list[str]:
        job_manager = JobManager(
            sweep_dir=self.sweep_dir,
            base_results_dir=self.base_results_dir,
            final_result_file=self.final_file,
        )
        results_path_names = job_manager.show_complete(show_relative_dir=True)
        return results_path_names

    def get_result_file_paths(self) -> list[str]:
        result_files = []
        for result_path_name in self.result_path_names:
            result_file = os.path.join(
                self.base_results_dir, result_path_name, self.final_file
            )
            result_files.append(result_file)
        return result_files

    @property
    def model_names(self) -> list[str]:
        """Assumes folder format is DATASETNAME_MODELNAME/DATE if not in config.json"""
        _model_names = []
        for file_path in self.result_path_names:
            config_path = Path(self.base_results_dir) / Path(file_path) / "config.json"
            if Path(config_path).exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                model_name = config_dict["model_name"]
            else:
                model_name = file_path.split("_")[1].split("/20")[0]
            _model_names.append(model_name)
        return _model_names

    @property
    def dataset_names(self) -> list[str]:
        """Assumes folder format is DATASETNAME_MODELNAME/DATE"""
        _dataset_names = []
        for file_path in self.result_path_names:
            config_path = Path(self.base_results_dir) / Path(file_path) / "config.json"
            if Path(config_path).exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                dataset_name = config_dict["dataset_name"]
            else:
                dataset_name = file_path.split("_")[0].split("/20")[0]
            _dataset_names.append(dataset_name)
        return _dataset_names

    @property
    def abstention_detector_parameters(self) -> list[str]:
        _parameters = []
        for file_path in self.result_path_names:
            config_path = Path(self.base_results_dir) / Path(file_path) / "config.json"

            parameters = None

            if Path(config_path).exists():
                parameters = {}

                with open(config_path) as f:
                    config_dict = json.load(f)
                    abstention_detector_config = config_dict["abstention_detector"]

                    abstention_detector_name = abstention_detector_config[
                        "_target_"
                    ].split(".")[-1]

                    parameters["abstention_detector_name"] = abstention_detector_name

                    if abstention_detector_name == "LLMJudgeAbstentionDetector":
                        judge_name = (
                            abstention_detector_config.get("judge_model", {})
                            .get("_target_", None)
                            .split(".")[-1]
                        )

                        parameters["abstention_detector_judge_model_name"] = judge_name

            _parameters.append(parameters)

        return _parameters

    def load(self, allow_missing_correctness_files=False) -> list[dict]:
        response_flat_dicts = []
        for (
            model_name,
            dataset_name,
            abstention_detector_parameters,
            result_file_path,
        ) in zip(
            self.model_names,
            self.dataset_names,
            self.abstention_detector_parameters,
            self.result_file_paths,
        ):
            result_file_path = Path(result_file_path)

            if (
                result_file_path.stem == "LLMJudgeCorrectnessEvaluator"
                and not result_file_path.exists()
            ):
                message = f"Path {result_file_path} does not exist"
                if allow_missing_correctness_files:
                    logger.warning(message)
                    continue
                else:
                    raise ValueError(message)

            responses = Responses.load(result_file_path)
            for response in responses.responses:
                response_flat_dict = response.to_flat_dict()
                response_flat_dict["model_name"] = model_name
                response_flat_dict["dataset_name"] = dataset_name

                response_flat_dict["dataset_name_extended"] = self.extend_dataset_name(
                    dataset_name, response
                )

                response_flat_dict = response_flat_dict | abstention_detector_parameters
                response_flat_dicts.append(response_flat_dict)
        return response_flat_dicts

    def extend_dataset_name(self, dataset_name, response):
        extension = None

        if dataset_name == "KUQDataset":
            extension = response.prompt.metadata["KUQ_category"]
            if extension is None:
                extension = "missing_category"
        elif dataset_name == "CoCoNotDataset":
            extension = response.prompt.metadata["CoCoNot_AbstentionBench_category"]

        if extension is not None:
            return f"{dataset_name}_{extension.lower().replace(' ', '_')}"
        else:
            return dataset_name

    def to_dataframe(self) -> pd.DataFrame:
        """selects essential columns to form a pandas table
        response_or_abstention: str
        is_abstention: Optional[bool] = None
        is_abstention_correct: Optional[bool] = None
        prompt_should_abstain: bool
        response: str
        ...
        """
        df = pd.DataFrame(self.raw_response_dicts)
        df = df.convert_dtypes()
        return df

    def filter_data(self, data_df) -> pd.DataFrame:
        """Remove datasets or slices that we're excluding from our analysis."""
        data_df = data_df.query(
            "dataset_name_extended != 'KUQDataset_missing_category'"
        )
        data_df = data_df.query("dataset_name_extended != 'KUQDataset_counterfactual'")
        data_df = data_df.query("dataset_name_extended != 'CoCoNotDataset_safety'")
        data_df = data_df.query("dataset_name_extended != 'SelfAwareDataset'")
        data_df = data_df.query("dataset_name_extended != 'NQDataset'")
        data_df = data_df.query("dataset_name_extended != 'CoCoNotDataset_underspecification'")

        return data_df

    def format_data(self, data_df) -> pd.DataFrame:
        """Add extra columns with nicely formatted model/dataset names."""
        formatted_data_df = data_df.copy(deep=True)

        formatted_data_df["model_name_formatted"] = formatted_data_df[
            "model_name"
        ].apply(_format_model_name)
        formatted_data_df["dataset_name_formatted"] = formatted_data_df[
            "dataset_name_extended"
        ].apply(_format_dataset_name)
        formatted_data_df["uncertainty_source"] = formatted_data_df[
            "dataset_name_extended"
        ].apply(_uncertainty_source)
        formatted_data_df["scenario_label"] = formatted_data_df[
            "dataset_name_extended"
        ].apply(_scenario_label)
        formatted_data_df["post_training_stage"] = formatted_data_df[
            "model_name_formatted"
        ].apply(post_training_stage)

        return formatted_data_df

    def filter_indeterminate_abstentions(self, df) -> pd.DataFrame:
        logger.warning(
            f"Filtering out a total of {self.df['is_abstention'].isnull().sum()} "
            f"examples from the results table where judge couldn't determine "
            f"whether response contained abstention. See the breakdown of these examples below:\n"
            + self.df[self.df["is_abstention"].isnull()]
            .groupby(["model_name", "dataset_name"])
            .count()["prompt_question"]
            .to_string()
        )
        filtered_df = self.df[self.df["is_abstention"].notnull()]
        return filtered_df

    def filter_indeterminate_correctness(self, df) -> pd.DataFrame:
        # is_response_correct can be null, but only if reference answers are not provided,
        # i.e., the question is should_abstain=True
        is_invalid = (
            df["is_response_correct"].isnull()
            & df["prompt_reference_answers"].notnull()
        )

        samples_to_keep = df[~is_invalid]
        samples_to_remove = df[is_invalid]

        logger.warning(
            f"Filtering out a total of {len(samples_to_remove)} "
            f"examples from the results table where judge couldn't determine "
            f"whether the response was correct. See the breakdown of these examples below:\n"
            + samples_to_remove.groupby(["model_name", "dataset_name"])
            .count()["prompt_question"]
            .to_string()
        )

        return samples_to_keep


def _format_model_name(model_name):
    model_name_map = {
        "GPT4oAPI": "GPT-4o",
        "Mistral_7B_Instruct_v0_3": "Mistral 7B v0.3",
        "OLMo_7B_0724_Instruct": "OLMo 7B",
        "Gemini15ProAPI": "Gemini 1.5 Pro",
        "DeepSeek_R1_Distill_Llama_70B": "DeepSeek R1 Distill Llama 70B",
        "Llama3_1_8B_Instruct": "Llama 3.1 8B Instruct",
        "Llama3_1_70B_Instruct": "Llama 3.1 70B Instruct",
        "Llama3_1_405B_Instruct_FP8": "Llama 3.1 405B Instruct",
        "Llama3_3_70B_Instruct": "Llama 3.3 70B Instruct",
        "Llama3_1_8B_Base": "Llama 3.1 8B Base",
        "Llama3_1_8B_Tulu_3_SFT": "Llama 3.1 8B Tulu 3 SFT",
        "Llama3_1_8B_Tulu_3_DPO": "Llama 3.1 8B Tulu 3 DPO",
        "Llama3_1_8B_Tulu_3_PPO_RLVF": "Llama 3.1 8B Tulu 3 PPO RLVF",
        "Llama3_1_70B_Base": "Llama 3.1 70B Base",
        "Llama3_1_70B_Tulu_3_SFT": "Llama 3.1 70B Tulu 3 SFT",
        "Llama3_1_70B_Tulu_3_DPO": "Llama 3.1 70B Tulu 3 DPO",
        "Llama3_1_70B_Tulu_3_PPO_RLVF": "Llama 3.1 70B Tulu 3 PPO RLVF",
        "S1_1_32B": "S1.1 32B",
        "o1API": "o1",
        "Qwen2_5_32B_Instruct": "Qwen2.5 32B",
    }
    try:
        return model_name_map[model_name]
    except KeyError:
        # Unknown nice formatting for model name.
        return model_name


def _format_dataset_name(dataset_name_extended):
    dataset_name_map = {
        "ALCUNADataset": "ALCUNA",
        "AveritecDataset": "Averitec",
        "BBQDataset": "BBQ",
        "BigBenchDisambiguateDataset": "BB/Disambiguate",
        "BigBenchKnownUnknownsDataset": "BB/Known unknowns",
        # Begin CoCoNot
        "CoCoNotDataset_false_presumptions": "CoCoNot/False presumptions",
        "CoCoNotDataset_incomprehensible": "CoCoNot/Incomprehensible",
        "CoCoNotDataset_subjective": "CoCoNot/Subjective",
        "CoCoNotDataset_underspecification": "CoCoNot/Underspecification",
        "CoCoNotDataset_unknowns": "CoCoNot/Unknowns",
        "CoCoNotDataset_temporal": "CoCoNot/Temporal",
        "CoCoNotDataset_humanizing": "CoCoNot/Humanizing",
        "CoCoNotDataset_safety": "CoCoNot/Safety",
        "CoCoNotDataset_unsupported": "CoCoNot/Unsupported",
        # End CoCoNot
        "FalseQADataset": "FalseQA",
        "FreshQADataset": "FreshQA",
        # Begin KUQ
        "KUQDataset_ambiguous": "KUQ/Ambiguous",
        "KUQDataset_controversial": "KUQ/Controversial",
        "KUQDataset_false_assumption": "KUQ/False assumptions",
        "KUQDataset_future_unknown": "KUQ/Future unknowns",
        "KUQDataset_unsolved_problem": "KUQ/Unsolved problems",
        # End KUQ
        "MediQDataset": "MediQ",
        "MoralChoiceDataset": "MoralChoice",
        "MusiqueDataset": "Musique",
        "NQDataset": "Natural Questions",
        "QAQADataset": "QAQA",
        "QASPERDataset": "QASPER",
        "SelfAwareDataset": "SelfAware",
        "SituatedQAGeoDataset": "SituatedQA/Geo",
        "Squad2Dataset": "SQuAD 2.0",
        "WorldSenseDataset": "WorldSense",
        "GPQA": "GPQA-Diamond",
        "UMWP": "UMWP",
        "MMLUMath": "MMLU Math",
        "MMLUHistory": "MMLU History",
        "GSM8K": "GSM8K",
    }
    try:
        return dataset_name_map[dataset_name_extended]
    except KeyError:
        # Unknown nice formatting for dataset name.
        return dataset_name_extended


def _scenario_label(dataset_name_extended):
    dataset_to_scenario_map = {
        "ALCUNADataset": "underspecified context",
        "AveritecDataset": "underspecified context",
        "BBQDataset": "underspecified context",
        "BigBenchDisambiguateDataset": "underspecified context",
        "BigBenchKnownUnknownsDataset": "answer unknown",
        # Begin CoCoNot
        "CoCoNotDataset_false_presumptions": "false premise",
        "CoCoNotDataset_incomprehensible": "underspecified intent",
        "CoCoNotDataset_subjective": "subjective",
        "CoCoNotDataset_underspecified context": "underspecified context",
        "CoCoNotDataset_unknowns": "answer unknown",
        "CoCoNotDataset_temporal": "stale",
        "CoCoNotDataset_humanizing": "subjective",
        "CoCoNotDataset_safety": "safety",
        "CoCoNotDataset_unsupported": "answer unknown",
        # End CoCoNot
        "FalseQADataset": "false premise",
        "FreshQADataset": "stale",
        # Begin KUQ
        "KUQDataset_ambiguous": "underspecified intent",
        "KUQDataset_controversial": "subjective",
        "KUQDataset_false_assumption": "false premise",
        "KUQDataset_future_unknown": "answer unknown",
        "KUQDataset_unsolved_problem": "answer unknown",
        # End KUQ
        "MediQDataset": "underspecified context",
        "MoralChoiceDataset": "subjective",
        "MusiqueDataset": "underspecified context",
        "NQDataset": "underspecified context",
        "QAQADataset": "false premise",
        "QASPERDataset": "underspecified context",
        "SelfAwareDataset": "answer unknown",
        "SituatedQAGeoDataset": "underspecified intent",
        "Squad2Dataset": "underspecified context",
        "WorldSenseDataset": "underspecified context",
        "GPQA": "underspecified context",
        "UMWP": "underspecified context",
        "MMLUMath": "underspecified context",
        "MMLUHistory": "underspecified context",
        "GSM8K": "underspecified context",
    }
    try:
        return dataset_to_scenario_map[dataset_name_extended]
    except KeyError:
        # Unknown scenario for dataset, setting scenario to the name of the dataset.
        return dataset_name_extended


def _uncertainty_source(dataset_name_extended):
    dataset_to_uncertainty_source_map = {
        "ALCUNADataset": "Context",
        "AveritecDataset": "Context",
        "BBQDataset": "Context",
        "BigBenchDisambiguateDataset": "Question",
        "BigBenchKnownUnknownsDataset": "Data",
        # Begin CoCoNot
        "CoCoNotDataset_false_presumptions": "Question",
        "CoCoNotDataset_incomprehensible": "Question",
        "CoCoNotDataset_subjective": "Question",
        "CoCoNotDataset_underspecification": "Question",
        "CoCoNotDataset_unknowns": "Question",
        "CoCoNotDataset_temporal": "Question",
        "CoCoNotDataset_humanizing": "Question",
        "CoCoNotDataset_safety": "Question",
        "CoCoNotDataset_unsupported": "Question",
        # End CoCoNot
        "FalseQADataset": "Question",
        "FreshQADataset": "Data",
        # Begin KUQ
        "KUQDataset_ambiguous": "Question",
        "KUQDataset_controversial": "Question",
        "KUQDataset_false_assumption": "Question",
        "KUQDataset_future_unknown": "Data",
        "KUQDataset_unsolved_problem": "Data",
        # End KUQ
        "MediQDataset": "Context",
        "MoralChoiceDataset": "Question",
        "MusiqueDataset": "Context",
        "NQDataset": "Context",
        "QAQADataset": "Context",
        "QASPERDataset": "Context",
        "SelfAwareDataset": "Composite",
        "SituatedQAGeoDataset": "Question",
        "Squad2Dataset": "Context",
        "WorldSenseDataset": "Context",
    }
    try:
        return dataset_to_uncertainty_source_map[dataset_name_extended]
    except KeyError:
        # Unknown uncertainty source. setting scenario to the name of the dataset.
        return dataset_name_extended


def post_training_stage(model_name_formatted):
    model_name_to_post_training_stage = {
        "Llama 3.1 8B Base": "Base",
        "Llama 3.1 8B Tulu 3 SFT": "SFT",
        "Llama 3.1 8B Tulu 3 DPO": "DPO",
        "Llama 3.1 8B Tulu 3 PPO RLVF": "PPO RLVF",
        "Llama 3.1 8B Instruct": "Instruct",
        "Llama 3.1 70B Base": "Base",
        "Llama 3.1 70B Tulu 3 SFT": "SFT",
        "Llama 3.1 70B Tulu 3 DPO": "DPO",
        "Llama 3.1 70B Tulu 3 PPO RLVF": "PPO RLVF",
        "Llama 3.1 70B Instruct": "Instruct",
    }
    return model_name_to_post_training_stage.get(model_name_formatted, "NA")


if __name__ == "__main__":
    results = Results()
    print(results.df)
