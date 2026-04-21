import importlib
import json
import logging
import warnings
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich import print
from rich.live import Live
from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite


def _normalize_pipeline_name(model: ModelsEnum, defense: str | None) -> str:
    pipeline_name = f"{model}-{defense}" if defense is not None else f"{model}"
    return pipeline_name.replace("/", "_")


def _as_non_negative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value if value is not None else default))
    except (TypeError, ValueError):
        return max(0, default)


def _read_token_usage_from_file(path: Path) -> dict[str, int | bool]:
    result: dict[str, int | bool] = {
        "original_input": 0, "original_output": 0, "original_total": 0,
        "defense_input": 0, "defense_output": 0, "defense_total": 0,
        "total_input": 0, "total_output": 0, "total_total": 0,
        "is_approximate": False, "is_missing": False,
    }
    if not path.exists():
        result["is_missing"] = True
        return result

    with path.open() as f:
        data = json.load(f)

    token_usage = data.get("token_usage")
    if not isinstance(token_usage, dict):
        result["is_missing"] = True
        return result

    for scope in ("original", "defense", "total"):
        bucket = token_usage.get(scope, {})
        if not isinstance(bucket, dict):
            bucket = {}
        result[f"{scope}_input"] = _as_non_negative_int(bucket.get("input", 0))
        result[f"{scope}_output"] = _as_non_negative_int(bucket.get("output", 0))
        result[f"{scope}_total"] = _as_non_negative_int(bucket.get("total", 0))

    meta = token_usage.get("meta", {})
    approximate_calls = meta.get("approximate_calls", {}) if isinstance(meta, dict) else {}
    if isinstance(approximate_calls, dict):
        result["is_approximate"] = (
            _as_non_negative_int(approximate_calls.get("original", 0))
            + _as_non_negative_int(approximate_calls.get("defense", 0))
        ) > 0

    return result


def _collect_suite_token_summary(
    suite_name: str,
    results: SuiteResults,
    logdir: Path,
    pipeline_name: str,
    attack: str | None,
) -> dict[str, int]:
    task_keys = list(results["utility_results"].keys())
    n_tasks = len(task_keys)
    attack_name = attack if attack is not None else "none"

    accum_keys = [
        "original_input", "original_output", "original_total",
        "defense_input", "defense_output", "defense_total",
        "total_input", "total_output", "total_total",
    ]
    accum = {k: 0 for k in accum_keys}
    missing_tasks = 0
    approximate_tasks = 0

    for user_task_id, injection_task_id in task_keys:
        injection_name = injection_task_id if injection_task_id else "none"
        path = logdir / pipeline_name / suite_name / user_task_id / attack_name / f"{injection_name}.json"
        file_data = _read_token_usage_from_file(path)
        for k in accum_keys:
            accum[k] += int(file_data[k])
        if file_data["is_missing"]:
            missing_tasks += 1
        if file_data["is_approximate"]:
            approximate_tasks += 1

    return {
        "tasks": n_tasks,
        "sum_original": accum["original_total"],
        "sum_original_input": accum["original_input"],
        "sum_original_output": accum["original_output"],
        "sum_defense": accum["defense_total"],
        "sum_defense_input": accum["defense_input"],
        "sum_defense_output": accum["defense_output"],
        "sum_total": accum["total_total"],
        "sum_total_input": accum["total_input"],
        "sum_total_output": accum["total_output"],
        "missing": missing_tasks,
        "approximate": approximate_tasks,
    }


def _merge_token_summaries(summaries: list[dict[str, int]]) -> dict[str, int]:
    merged = {
        "tasks": 0,
        "sum_original": 0,
        "sum_original_input": 0,
        "sum_original_output": 0,
        "sum_defense": 0,
        "sum_defense_input": 0,
        "sum_defense_output": 0,
        "sum_total": 0,
        "sum_total_input": 0,
        "sum_total_output": 0,
        "missing": 0,
        "approximate": 0,
    }
    for summary in summaries:
        for key in merged:
            merged[key] += _as_non_negative_int(summary.get(key, 0))
    return merged


def benchmark_suite(
    suite: TaskSuite,
    model: ModelsEnum,
    logdir: Path,
    force_rerun: bool,
    benchmark_version: str,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    live: Live | None = None,
    tool_output_format: Literal["yaml", "json"] | None = None,
) -> SuiteResults:
    if not load_dotenv(".env"):
        warnings.warn("No .env file found")

    print(f"Running benchmark for suite: '{suite.name}'")
    print(f"Using model: '{model}'")
    if attack is not None:
        print(f"Using attack: '{attack}'")
    if defense is not None:
        print(f"Using defense: '{defense}'")
    if len(user_tasks) > 0:
        print(f"Using user tasks: {', '.join(user_tasks)}")

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=model,
            model_id=model_id,
            defense=defense,
            tool_delimiter=tool_delimiter,
            system_message_name=system_message_name,
            system_message=system_message,
            tool_output_format=tool_output_format,
        )
    )
    with OutputLogger(str(logdir), live=live):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
        else:
            attacker_ = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker_,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                injection_tasks=injection_tasks if len(injection_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
    print(f"Finished benchmark for suite: '{suite.name}'")

    return results


def show_results(
    suite_name: str,
    results: SuiteResults,
    show_security_results: bool,
    token_summary: dict[str, int] | None = None,
):
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results)

    print(f"Results for suite {suite_name}")
    print(f"Average utility: {avg_utility * 100:.2f}%")

    if show_security_results:
        passed_injection_tasks = sum(results["injection_tasks_utility_results"].values())
        total_injection_tasks = len(results["injection_tasks_utility_results"])
        print(f"\nPassed injection tasks as user tasks: {passed_injection_tasks}/{total_injection_tasks}")

        security_results = results["security_results"].values()
        avg_security = sum(security_results) / len(security_results)
        print(f"Average security: {avg_security * 100:.2f}%")

    if token_summary is not None:
        tasks = _as_non_negative_int(token_summary.get("tasks", 0))
        avg_original = token_summary.get("sum_original", 0) / tasks if tasks > 0 else 0.0
        avg_original_input = token_summary.get("sum_original_input", 0) / tasks if tasks > 0 else 0.0
        avg_original_output = token_summary.get("sum_original_output", 0) / tasks if tasks > 0 else 0.0
        avg_defense = token_summary.get("sum_defense", 0) / tasks if tasks > 0 else 0.0
        avg_defense_input = token_summary.get("sum_defense_input", 0) / tasks if tasks > 0 else 0.0
        avg_defense_output = token_summary.get("sum_defense_output", 0) / tasks if tasks > 0 else 0.0
        avg_total = token_summary.get("sum_total", 0) / tasks if tasks > 0 else 0.0
        avg_total_input = token_summary.get("sum_total_input", 0) / tasks if tasks > 0 else 0.0
        avg_total_output = token_summary.get("sum_total_output", 0) / tasks if tasks > 0 else 0.0
        print(
            "Token usage summary: "
            f"tasks={tasks}, "
            f"avg_original={avg_original:.2f} (input={avg_original_input:.2f}, output={avg_original_output:.2f}), "
            f"avg_defense={avg_defense:.2f} (input={avg_defense_input:.2f}, output={avg_defense_output:.2f}), "
            f"avg_total={avg_total:.2f} (input={avg_total_input:.2f}, output={avg_total_output:.2f})"
        )

        missing = _as_non_negative_int(token_summary.get("missing", 0))
        approximate = _as_non_negative_int(token_summary.get("approximate", 0))
        if missing > 0 or approximate > 0:
            print(
                "Token usage note: "
                f"missing={missing} task(s) treated as 0, "
                f"approximate={approximate} task(s)."
            )


@click.command()
@click.option(
    "--model",
    default="gpt-4o-2024-05-13",
    type=click.Choice([value for value in ModelsEnum]),
    help="The language model to benchmark. Defaults to `gpt-4o-2024-05-13`.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="The model id for local models.",
)
@click.option(
    "--benchmark-version",
    default="v1.2.1",
    type=str,
    help="The version of the benchmark to run. Defaults to `v1.2`.",
)
@click.option(
    "--tool-delimiter",
    type=str,
    default="tool",
    help="Which tool delimiter to use. Used for local models only.",
)
@click.option(
    "--logdir",
    default="./runs",
    type=Path,
    help="The directory to save logs. Defaults to `./runs`.",
)
@click.option(
    "--attack",
    type=str,
    default=None,
    help=f"The attack to use. `None` by default. It should be one of {ATTACKS}, or a custom attack defined and registered in of of the `--module-to-load`. If `None`, no attack is used.",
)
@click.option(
    "--defense",
    type=click.Choice(DEFENSES),
    default=None,
    help="The defense to use. `None` by default.",
)
@click.option(
    "--system-message-name",
    type=str,
    default=None,
    help="The name of the system message to use among the default ones in `data/system_messages.yaml`.",
)
@click.option(
    "--system-message",
    type=str,
    default=None,
    help="The system message to use (as a string). If provided, `--system-message-name` is ignored.",
)
@click.option(
    "--user-task",
    "-ut",
    "user_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--injection-task",
    "-it",
    "injection_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--suite",
    "-s",
    "suites",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help="How many suites can be benchmarked in parallel through multithreading. Nothing is parallelized by default.",
)
@click.option(
    "--force-rerun",
    "-f",
    is_flag=True,
    help="Whether to re-run tasks that have already been run.",
)
@click.option(
    "--module-to-load",
    "-ml",
    "modules_to_load",
    type=str,
    multiple=True,
    default=tuple(),
    help="The modules to load before running the benchmark. Useful for registering custom attacks, defenses, or suites.",
)
def main(
    suites: tuple[str, ...],
    model: ModelsEnum,
    benchmark_version: str = "v1.2.1",
    logdir: Path = Path("./runs"),
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    max_workers: int = 1,
    force_rerun: bool = False,
    modules_to_load: tuple[str, ...] = (),
):
    for module in modules_to_load:
        importlib.import_module(module)

    if len(suites) == 0:
        suites = tuple(get_suites(benchmark_version).keys())

    if len(suites) != 1:
        print(f"Benchmarking suites {', '.join(suites)}")

    if len(suites) != 1 and len(user_tasks) != 0:
        raise ValueError("A user task can be specified only when one suite is being executed")

    if max_workers == 1:
        results = {}
        suite_token_summaries: dict[str, dict[str, int]] = {}
        pipeline_name = _normalize_pipeline_name(model, defense)
        for suite_name in suites:
            suite = get_suite(benchmark_version, suite_name) # Suite object is returned
            results[suite_name] = benchmark_suite(
                suite,
                model,
                logdir,
                benchmark_version=benchmark_version,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                model_id=model_id,
                attack=attack,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                force_rerun=force_rerun,
            )
            suite_token_summaries[suite_name] = _collect_suite_token_summary(
                suite_name,
                results[suite_name],
                logdir,
                pipeline_name,
                attack,
            )
        combined_results = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})
        for suite_name, result in results.items():
            show_results(suite_name, result, attack is not None, token_summary=suite_token_summaries[suite_name])
            for user_task, injection_task in result["utility_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["utility_results"][(new_user_task, injection_task)] = result["utility_results"][
                    (user_task, injection_task)
                ]
            for user_task, injection_task in result["security_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["security_results"][(new_user_task, injection_task)] = result["security_results"][
                    (user_task, injection_task)
                ]
            for injection_task in result["injection_tasks_utility_results"]:
                new_injection_task = suite_name + "_" + injection_task
                combined_results["injection_tasks_utility_results"][new_injection_task] = result[
                    "injection_tasks_utility_results"
                ][injection_task]
        print(f"Combined security results: {combined_results['security_results']}")
        combined_token_summary = _merge_token_summaries(list(suite_token_summaries.values()))
        show_results("combined", combined_results, attack is not None, token_summary=combined_token_summary)
        exit(0)

    with Pool(max_workers) as p:
        arguments = zip(
            suites,
            repeat(model),
            repeat(logdir),
            repeat(user_tasks),
            repeat(injection_tasks),
            repeat(model_id),
            repeat(attack),
            repeat(defense),
            repeat(tool_delimiter),
            repeat(system_message_name),
            repeat(system_message),
            repeat(force_rerun),
        )
        results = p.starmap(benchmark_suite, arguments)

    for suite_name, result in zip(suites, results):
        pipeline_name = _normalize_pipeline_name(model, defense)
        token_summary = _collect_suite_token_summary(suite_name, result, logdir, pipeline_name, attack)
        show_results(suite_name, result, attack is not None, token_summary=token_summary)


if __name__ == "__main__":
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()
