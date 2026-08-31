import os
import json
from typing import List
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.evaluate_zero_shot import TASKS


# Script for additional analysis and plotting for the PsyNamic manuscript.

def _load_json(path: str):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def _get_metric_from_entry(entry: dict, metric: str):
    """
    Entry can be:
        {'metrics': {...}}
    or:
        {'f1-weighted': [mean, [ci_low, ci_high]], ...}
    """
    metrics = entry.get("metrics", entry)

    if metric in metrics:
        v = metrics[metric]

        if isinstance(v, list):
            return float(v[0])

        if isinstance(v, dict):
            return float(v.get("mean", float("nan")))

    return None


def collect_values_for_models(
    class_metric: str = "f1-weighted",
    ner_metric_candidates: List[str] = None
):
    class_metric = "f1-weighted"
    ner_metric = "f1_dosage"

    rows = []

    for task in TASKS:
        zs_path = os.path.join(
            "zero_shot",
            task.lower().replace(" ", "_"),
            "performance_reports.json"
        )

        data = _load_json(zs_path)

        if not data:
            continue

        for model, entry in data.items():
            mlow = model.lower()
            condition = "zero-shot"

            if "gpt-4o-2024-08-06" in mlow:
                model = "GPT-4o"

            elif "medgemma" in mlow:
                model = "MedGemma-27b-text-it"

            elif "bert" in mlow:
                model = "BERT"

            elif (
                "tuned" in mlow
                or "ift" in mlow
                or "instruction" in mlow
            ):
                model = "Llama-3.1-8B-Instruct"
                condition = "ift"

            else:
                continue

            val = _get_metric_from_entry(entry, class_metric)

            if val is not None:
                rows.append({
                    "model": model,
                    "condition": condition,
                    "task": task,
                    "value": val,
                    "is_ner": False
                })

    for task in TASKS:
        fs_path = os.path.join(
            "few_shot",
            task.lower().replace(" ", "_"),
            "performance_report.json"
        )

        data = _load_json(fs_path)

        if not data:
            continue

        for model, conditions in data.items():
            mlow = model.lower()

            if not (
                "gpt-4o-2024-08-06" in mlow
                or "medgemma" in mlow
            ):
                continue

            best_val = None

            for cond in [
                "selected_1shot",
                "selected_3shot",
                "selected_5shot"
            ]:
                if cond not in conditions:
                    continue

                val = _get_metric_from_entry(
                    conditions[cond],
                    class_metric
                )

                if val is not None:
                    if best_val is None or val > best_val:
                        best_val = val

            if "gpt-4o" in mlow:
                model = "GPT-4o"

            elif "medgemma" in mlow:
                model = "MedGemma-27b-text-it"

            if best_val is not None:
                rows.append({
                    "model": model,
                    "condition": "few-shot",
                    "task": task,
                    "value": best_val,
                    "is_ner": False
                })

    zs_ner = _load_json(
        os.path.join(
            "zero_shot",
            "ner",
            "performance_reports.json"
        )
    )

    if zs_ner:
        for model, entry in zs_ner.items():

            val = _get_metric_from_entry(
                entry,
                ner_metric
            )

            if val is None:
                continue

            mlow = model.lower()
            condition = "zero-shot"

            if "gpt-4o-2024-08-06" in mlow:
                model = "GPT-4o"

            elif "medgemma" in mlow:
                model = "MedGemma-27b-text-it"

            elif "bert" in mlow:
                model = "BERT"

            elif (
                "tuned" in mlow
                or "ift" in mlow
            ):
                model = "Llama-3.1-8B-Instruct"
                condition = "ift"

            else:
                continue

            rows.append({
                "model": model,
                "condition": condition,
                "task": "NER-dosage",
                "value": val,
                "is_ner": True
            })

    fs_ner = _load_json(
        os.path.join(
            "few_shot",
            "ner",
            "ner_performance_report.json"
        )
    )

    if fs_ner:
        for model, conditions in fs_ner.items():
            mlow = model.lower()

            if not (
                "gpt-4o-2024-08-06" in mlow
                or "medgemma" in mlow
            ):
                continue

            best_val = None

            for cond, entry in conditions.items():
                val = _get_metric_from_entry(
                    entry,
                    ner_metric
                )

                if val is not None:
                    if best_val is None or val > best_val:
                        best_val = val

            if "medgemma" in mlow:
                model = "MedGemma-27b-text-it"

            elif "gpt-4o" in mlow:
                model = "GPT-4o"

            if best_val is not None:
                rows.append({
                    "model": model,
                    "condition": "few-shot",
                    "task": "NER-dosage",
                    "value": best_val,
                    "is_ner": True
                })

    df = pd.DataFrame(rows)

    # Save the combined data
    os.makedirs("evaluation", exist_ok=True)

    df.to_csv(
        "evaluation/custom_box_plot_data.csv",
        index=False
    )

    return df


def compare_models(df: pd.DataFrame):
    """
    Performs the comparisons from Script 1 using the DataFrame
    generated by collect_values_for_models().
    """

    # Clean text fields
    df = df.copy()

    df["task"] = df["task"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    bert_zs = df[
        (df["model"] == "BERT") &
        (df["condition"] == "zero-shot")
    ]

    gpt_zs = df[
        (df["model"] == "GPT-4o") &
        (df["condition"] == "zero-shot")
    ]

    zs = pd.merge(
        bert_zs,
        gpt_zs,
        on="task",
        suffixes=("_bert", "_gpt")
    )

    zs["gap"] = (
        zs["value_bert"] -
        zs["value_gpt"]
    )

    print("\n" + "=" * 60)
    print("ZERO-SHOT: BERT - GPT-4o")
    print("=" * 60)

    if zs.empty:
        print("No matching BERT/GPT-4o zero-shot tasks found.")

    else:
        print("Min gap:", zs["gap"].min())
        print("Max gap:", zs["gap"].max())
        print("Mean gap:", zs["gap"].mean())

    bert_zs = df[
        (df["model"] == "BERT") &
        (df["condition"] == "zero-shot")
    ]

    gpt_fs = df[
        (df["model"] == "GPT-4o") &
        (df["condition"] == "few-shot")
    ]

    cross = pd.merge(
        bert_zs,
        gpt_fs,
        on="task",
        suffixes=("_bert", "_gpt")
    )

    cross["gap"] = (
        cross["value_bert"] -
        cross["value_gpt"]
    )

    print("\n" + "=" * 60)
    print("CROSS SETTING: BERT ZERO-SHOT - GPT-4o FEW-SHOT")
    print("=" * 60)

    if cross.empty:
        print("No matching BERT/GPT-4o cross-setting tasks found.")

    else:
        print("Min gap:", cross["gap"].min())
        print("Max gap:", cross["gap"].max())
        print("Mean gap:", cross["gap"].mean())

        print("\nGPT few-shot wins (cross setting):")

        gpt_wins = cross[
            cross["gap"] < 0
        ][["task", "gap"]]

        if gpt_wins.empty:
            print("None")

        else:
            print(
                gpt_wins.to_string(index=False)
            )


        print("\nBERT zero-shot wins (cross setting):")

        bert_wins = cross[
            cross["gap"] > 0
        ][["task", "gap"]]

        if bert_wins.empty:
            print("None")

        else:
            print(
                bert_wins.to_string(index=False)
            )


        ties = cross[
            cross["gap"] == 0
        ][["task", "gap"]]

        print("\nTies:")

        if ties.empty:
            print("None")

        else:
            print(
                ties.to_string(index=False)
            )

    return zs, cross


def make_custom_boxplot(
    df: pd.DataFrame,
    save_path: str = "evaluation/custom_box_plot.png",
    show_stats: bool = True
):
    if df.empty:
        print(
            "No data found for requested models/tasks. "
            "Check the prediction report files."
        )
        return

    df = df.copy()

    df["model_cond"] = (
        df["model"] +
        " - " +
        df["condition"]
    )

    # Display labels for plot
    df["model_cond_display"] = df[
        "model_cond"
    ].replace({
        "BERT - zero-shot": "BERT"
    })

    order = [
        "GPT-4o - zero-shot",
        "MedGemma-27b-text-it - zero-shot",
        "GPT-4o - few-shot",
        "MedGemma-27b-text-it - few-shot",
        "Llama-3.1-8B-Instruct - ift",
        "BERT - zero-shot"
    ]

    present_order = [
        o for o in order
        if o in df["model_cond"].unique()
    ]

    present_order_display = [
        "BERT" if o == "BERT - zero-shot" else o
        for o in present_order
    ]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    ax = sns.boxplot(
        x="model_cond_display",
        y="value",
        data=df,
        order=present_order_display,
        showcaps=True,
        boxprops={"alpha": 0.6}
    )

    unique = present_order_display
    x_map = {
        g: i
        for i, g in enumerate(unique)
    }

    rng = 0.12

    for _, row in df.iterrows():

        x = x_map.get(
            row["model_cond_display"],
            None
        )

        if x is None:
            continue

        jitter = random.uniform(
            -rng / 2,
            rng / 2
        )

        c = (
            "tab:orange"
            if row["is_ner"]
            else "tab:blue"
        )

        ax.scatter(
            x + jitter,
            row["value"],
            color=c,
            edgecolor="k",
            s=50,
            alpha=0.9
        )

    grouped = df.groupby(
        "model_cond_display"
    )["value"]

    stats = {}

    for name, series in grouped:

        q1 = series.quantile(0.25)
        median = series.median()
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_whisker = series[
            series >= (q1 - 1.5 * iqr)
        ].min()

        upper_whisker = series[
            series <= (q3 + 1.5 * iqr)
        ].max()

        stats[name] = {
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "iqr": float(iqr),
            "lw": float(lower_whisker),
            "uw": float(upper_whisker)
        }

    with open(
        "evaluation/custom_box_plot_stats.json",
        "w"
    ) as f:
        json.dump(
            stats,
            f,
            indent=2
        )

    max_uw = max(
        v["uw"]
        for v in stats.values()
    )

    top_ylim = max(
        1.0,
        max_uw +
        (0.12 if show_stats else 0.04)
    )

    ax.set_ylim(0, top_ylim)

    if show_stats:

        for i, g in enumerate(unique):

            s = stats.get(g)

            if not s:
                continue

            y = s["uw"] + 0.02

            text = (
                f"median: {s['median']:.3f}\n"
                f"Q1: {s['q1']:.3f}  "
                f"Q3: {s['q3']:.3f}\n"
                f"IQR: {s['iqr']:.3f}\n"
                f"lower whisker: {s['lw']:.3f}  "
                f"upper whisker: {s['uw']:.3f}"
            )

            ax.text(
                i,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    alpha=0.6
                )
            )

    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Model and Condition")
    ax.set_title(
        "Comparison: Zero-shot, Few-shot (best), IFT, and BERT"
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            markeredgecolor="k",
            markersize=8,
            label="Abstract-level parameters"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:orange",
            markeredgecolor="k",
            markersize=8,
            label="Token-level parameter"
        )
    ]

    ax.legend(
        handles=handles,
        loc="lower right"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600
    )

    plt.close()

    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":

    # 1. Collect data from JSON reports
    df = collect_values_for_models()

    # 2. Run BERT vs GPT comparisons
    zs, cross = compare_models(df)

    # 3. Generate boxplot
    make_custom_boxplot(
        df,
        show_stats=False
    )