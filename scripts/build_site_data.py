#!/usr/bin/env python3
"""Build compact site datasets from the benchmark repository artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
COLLECTED_DIR = ROOT / "cuda-profiling" / "collected-data"
HECBENCH_DIR = ROOT / "HeCBench"
AGENTIC_DIR = ROOT / "gpuFLOPBench-agentic"

PERF_CSV = COLLECTED_DIR / "all-NCU-GPU-Data.csv"
FEATURE_CSV = COLLECTED_DIR / "all-NCU-GPU-Data-with-IMIX.csv"
IMIX_CSV = COLLECTED_DIR / "all-IMIX-Data.csv"
BENCHMARKS_YAML = HECBENCH_DIR / "benchmarks.yaml"

DEVICE_MAP = {
    "NVIDIA GeForce RTX 3080": {"short": "3080", "family": "Ampere", "cc": "sm_86"},
    "NVIDIA A10": {"short": "A10", "family": "Ampere", "cc": "sm_86"},
    "NVIDIA A100-SXM4-40GB": {"short": "A100", "family": "Ampere", "cc": "sm_80"},
    "NVIDIA H100 80GB HBM3": {"short": "H100", "family": "Hopper", "cc": "sm_90"},
}

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_MISSING = "missing"


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "unknown"


def compact_num(value: float | int) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def maybe_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_records(frame: pd.DataFrame) -> list[dict]:
    records = json.loads(frame.to_json(orient="records"))
    return records


def normalize_device_name(name: str) -> str:
    return DEVICE_MAP.get(name, {}).get("short", name)


def normalize_omp_symbol(symbol: str) -> str:
    if not symbol:
        return "unknown"
    match = re.search(r"([A-Za-z0-9]+_l\d+)$", symbol)
    if match:
        return match.group(1)
    return symbol


def simplify_demangled(name: str) -> str:
    if not name:
        return "unknown"
    simple = name.strip()
    if "(" in simple:
        simple = simple.split("(", 1)[0].strip()
    if " " in simple:
        simple = simple.split()[-1]
    if "<" in simple:
        simple = simple.split("<", 1)[0]
    if "::" in simple:
        simple = simple.split("::")[-1]
    return simple or "unknown"


def normalize_display_kernel(row: pd.Series) -> str:
    if row["model_type"] == "omp":
        return normalize_omp_symbol(str(row["Kernel Name"]))
    return simplify_demangled(str(row.get("Demangled Name", "")))


def average_rank(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return maybe_float(series.mean())


def top_entries(mapping: dict[str, float], limit: int = 3) -> list[dict]:
    sorted_items = sorted(mapping.items(), key=lambda item: item[1], reverse=True)
    return [
        {"name": name, "value": maybe_float(value)}
        for name, value in sorted_items[:limit]
        if maybe_float(value) is not None and value > 0
    ]


def kmeans(points: np.ndarray, k: int, seed: int = 17, iterations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=min(k, len(points)), replace=False)
    centroids = points[indices].astype(float)

    for _ in range(iterations):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster in range(len(centroids)):
            cluster_points = points[labels == cluster]
            if len(cluster_points) == 0:
                new_centroids[cluster] = points[rng.integers(0, len(points))]
            else:
                new_centroids[cluster] = cluster_points.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, centroids


def dominant_label(row: pd.Series, columns: list[str]) -> str:
    values = {column.replace("OpType_", ""): maybe_float(row[column]) or 0.0 for column in columns}
    if not values:
        return "unknown"
    return max(values.items(), key=lambda item: item[1])[0]


def load_metadata() -> dict:
    with BENCHMARKS_YAML.open() as handle:
        return yaml.safe_load(handle)


def build_inventory(metadata: dict, perf_sources: set[str], feature_sources: set[str], unmapped: list[str]) -> dict:
    model_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    with_tests = 0

    for _, details in metadata.items():
        for model in details.get("models", []):
            model_counts[model] = model_counts.get(model, 0) + 1
        for category in details.get("categories", []) or ["uncategorized"]:
            category_counts[category] = category_counts.get(category, 0) + 1
        if details.get("test"):
            with_tests += 1

    total_benchmarks = len(metadata)
    profiled_benchmarks = {source.rsplit("-", 1)[0] for source in perf_sources}
    feature_benchmarks = {source.rsplit("-", 1)[0] for source in feature_sources}

    return {
        "totals": {
            "benchmarks_yaml": total_benchmarks,
            "profiled_benchmarks": len(profiled_benchmarks),
            "feature_benchmarks": len(feature_benchmarks),
            "profiled_sources": len(perf_sources),
            "feature_sources": len(feature_sources),
            "with_tests": with_tests,
        },
        "models_available": [{"model": model, "count": count} for model, count in sorted(model_counts.items())],
        "categories_available": [{"category": category, "count": count} for category, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)],
        "coverage": {
            "profiled_benchmark_ratio": round(len(profiled_benchmarks) / total_benchmarks, 4),
            "feature_benchmark_ratio": round(len(feature_benchmarks) / total_benchmarks, 4),
            "feature_source_gap": len(perf_sources - feature_sources),
            "unmapped_profiled_benchmarks": unmapped,
        },
    }


def build_audit(metadata: dict, perf_sources: set[str], feature_sources: set[str], missing_feature_sources: list[str], agentic_stub_files: list[str]) -> dict:
    build_dir = ROOT / "build" / "bin"
    perf_available = PERF_CSV.exists() and PERF_CSV.stat().st_size > 1024
    feature_available = FEATURE_CSV.exists() and FEATURE_CSV.stat().st_size > 1024
    imix_available = IMIX_CSV.exists() and IMIX_CSV.stat().st_size > 1024

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "checks": [
            {
                "id": "hecbench-metadata",
                "label": "HeCBench metadata base",
                "status": STATUS_COMPLETE if metadata else STATUS_MISSING,
                "detail": f"{len(metadata)} benchmark entries parsed from benchmarks.yaml.",
            },
            {
                "id": "aggregated-performance",
                "label": "Aggregated performance corpus",
                "status": STATUS_COMPLETE if perf_available else STATUS_MISSING,
                "detail": f"{compact_num(PERF_CSV.stat().st_size)} CSV available at {PERF_CSV.relative_to(ROOT)}." if perf_available else "Full performance CSV missing.",
            },
            {
                "id": "feature-enriched-corpus",
                "label": "IMIX-enriched feature corpus",
                "status": STATUS_COMPLETE if feature_available and imix_available and not missing_feature_sources else STATUS_PARTIAL,
                "detail": f"{len(feature_sources)} feature-enriched sources available; {len(missing_feature_sources)} raw performance sources do not appear in the IMIX merge.",
            },
            {
                "id": "build-artifacts",
                "label": "Local build artifacts",
                "status": STATUS_COMPLETE if build_dir.exists() else STATUS_PARTIAL,
                "detail": "build/bin is present." if build_dir.exists() else "This checkout does not include build/bin outputs; runBuild.sh must be executed in the Docker/NVIDIA toolchain before artifact tests can pass.",
            },
            {
                "id": "agentic-tools",
                "label": "Agentic analysis layer",
                "status": STATUS_PARTIAL if agentic_stub_files else STATUS_COMPLETE,
                "detail": f"{len(agentic_stub_files)} helper modules still expose NotImplemented TODO stubs." if agentic_stub_files else "No helper stubs detected in treesitter support modules.",
            },
        ],
        "feature_gap_sources": missing_feature_sources[:72],
        "agentic_stub_files": agentic_stub_files,
    }


def build_perf_data(metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = [
        "source",
        "codename",
        "model_type",
        "device",
        "Kernel Name",
        "Demangled Name",
        "exeArgs",
        "Block Size",
        "Grid Size",
        "sample",
        "SP_FLOP",
        "DP_FLOP",
        "HP_FLOP",
        "INTOP",
        "traffic",
        "bytesTotal",
        "dpAI",
        "spAI",
        "hpAI",
        "intAI",
        "dpPerf",
        "spPerf",
        "hpPerf",
        "intPerf",
        "xtime",
    ]
    perf = pd.read_csv(PERF_CSV, usecols=usecols, low_memory=False)
    perf["device_label"] = perf["device"]
    perf["device"] = perf["device"].map(normalize_device_name)
    perf["benchmark"] = perf["source"].str.replace(r"-(cuda|omp)$", "", regex=True)
    perf["category"] = perf["benchmark"].map(
        {name: (details.get("categories") or ["uncategorized"])[0] for name, details in metadata.items()}
    ).fillna("unmapped")
    perf["display_kernel"] = perf.apply(normalize_display_kernel, axis=1)

    key_cols = [
        "source",
        "benchmark",
        "category",
        "codename",
        "model_type",
        "device",
        "device_label",
        "Kernel Name",
        "Demangled Name",
        "display_kernel",
        "exeArgs",
        "Block Size",
        "Grid Size",
    ]
    metric_cols = [
        "SP_FLOP",
        "DP_FLOP",
        "HP_FLOP",
        "INTOP",
        "traffic",
        "bytesTotal",
        "dpAI",
        "spAI",
        "hpAI",
        "intAI",
        "dpPerf",
        "spPerf",
        "hpPerf",
        "intPerf",
        "xtime",
    ]
    perf_kernel = perf.groupby(key_cols, dropna=False)[metric_cols].mean().reset_index()
    perf_kernel["total_ops"] = perf_kernel[["SP_FLOP", "DP_FLOP", "HP_FLOP", "INTOP"]].fillna(0).sum(axis=1)
    perf_kernel["total_perf"] = perf_kernel[["dpPerf", "spPerf", "hpPerf", "intPerf"]].fillna(0).sum(axis=1)
    perf_kernel["total_ai"] = perf_kernel.apply(
        lambda row: (row["total_ops"] / row["traffic"]) if maybe_float(row["traffic"]) and row["traffic"] else np.nan,
        axis=1,
    )
    perf_kernel["dominant_perf_type"] = perf_kernel[["intPerf", "spPerf", "dpPerf", "hpPerf"]].fillna(0).idxmax(axis=1).str.replace("Perf", "", regex=False)

    source_group_cols = ["source", "benchmark", "category", "codename", "model_type", "device", "device_label"]
    perf_source = (
        perf_kernel.groupby(source_group_cols, dropna=False)
        .agg(
            kernel_count=("display_kernel", "nunique"),
            peak_total_perf=("total_perf", "max"),
            median_total_perf=("total_perf", "median"),
            median_total_ai=("total_ai", "median"),
            median_xtime=("xtime", "median"),
            mean_bytes_total=("bytesTotal", "mean"),
            mean_traffic=("traffic", "mean"),
        )
        .reset_index()
    )
    perf_source["coverage_rank"] = perf_source.groupby(["device", "model_type"])["peak_total_perf"].rank(ascending=False, method="dense")

    return perf, perf_kernel, perf_source


def build_feature_data(metadata: dict) -> tuple[pd.DataFrame, dict]:
    feature = pd.read_csv(FEATURE_CSV, low_memory=False)
    feature["benchmark"] = feature["source"].str.replace(r"-(cuda|omp)$", "", regex=True)
    feature["category"] = feature["benchmark"].map(
        {name: (details.get("categories") or ["uncategorized"])[0] for name, details in metadata.items()}
    ).fillna("unmapped")
    feature["display_kernel"] = feature["Kernel Name"]

    op_columns = [column for column in feature.columns if column.startswith("OpType_")]
    op_matrix = feature[op_columns].replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    op_totals = op_matrix.sum(axis=1).replace(0, 1.0)
    op_shares = op_matrix.div(op_totals, axis=0)
    feature["dominant_op_type"] = op_shares.idxmax(axis=1).str.replace("OpType_", "", regex=False)

    centered = op_shares.replace([np.inf, -np.inf], 0).fillna(0)
    centered = centered - centered.mean(axis=0)
    centered_matrix = centered.to_numpy(dtype=float, copy=True)
    _, _, vt = np.linalg.svd(centered_matrix, full_matrices=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coords = centered_matrix @ vt[:2].T
    feature["pc1"] = coords[:, 0]
    feature["pc2"] = coords[:, 1]

    cluster_labels, centroids = kmeans(coords, k=min(8, len(feature)))
    feature["cluster"] = cluster_labels.astype(int)

    cluster_summary = []
    for cluster_id in sorted(feature["cluster"].unique()):
        subset = feature[feature["cluster"] == cluster_id]
        mean_op_shares = op_shares.loc[subset.index].mean().to_dict()
        examples = (
            subset.groupby(["source", "display_kernel"])
            .size()
            .sort_values(ascending=False)
            .head(3)
            .reset_index()[["source", "display_kernel"]]
            .to_dict(orient="records")
        )
        cluster_summary.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(subset)),
                "top_op_types": top_entries({key.replace("OpType_", ""): value for key, value in mean_op_shares.items()}, limit=4),
                "median_total_ai": maybe_float(subset["total_AI"].median()),
                "median_ipc": maybe_float(subset["ipc_active"].median()),
                "median_issue_to_exec": maybe_float(subset["issue_to_exec"].median()),
                "examples": examples,
                "centroid": {"pc1": maybe_float(centroids[cluster_id][0]), "pc2": maybe_float(centroids[cluster_id][1])},
            }
        )

    return feature, {"clusters": cluster_summary, "op_columns": [column.replace("OpType_", "") for column in op_columns]}


def build_device_summary(perf_kernel: pd.DataFrame, perf_source: pd.DataFrame, feature: pd.DataFrame) -> list[dict]:
    result = []
    feature_summary = feature.groupby("device").agg(
        median_ipc=("ipc_active", "median"),
        median_issue_to_exec=("issue_to_exec", "median"),
        median_occupancy=("theoretical_occupancy_pct", "median"),
    )

    for device, subset in perf_kernel.groupby("device"):
        source_subset = perf_source[perf_source["device"] == device]
        full_name = subset["device_label"].iloc[0]
        info = DEVICE_MAP.get(full_name, {"short": device, "family": "unknown", "cc": "unknown"})
        models = (
            source_subset.groupby("model_type")["source"]
            .nunique()
            .sort_values(ascending=False)
            .to_dict()
        )
        result.append(
            {
                "device": info["short"],
                "label": full_name,
                "family": info["family"],
                "compute_capability": info["cc"],
                "rows": int(len(subset)),
                "benchmarks": int(source_subset["benchmark"].nunique()),
                "sources": int(source_subset["source"].nunique()),
                "kernels": int(subset["display_kernel"].nunique()),
                "models": [{"model": model, "sources": int(count)} for model, count in models.items()],
                "median_total_perf": maybe_float(subset["total_perf"].median()),
                "p95_total_perf": maybe_float(subset["total_perf"].quantile(0.95)),
                "median_total_ai": maybe_float(subset["total_ai"].median()),
                "median_xtime": maybe_float(subset["xtime"].median()),
                "median_ipc": maybe_float(feature_summary.loc[device, "median_ipc"]) if device in feature_summary.index else None,
                "median_issue_to_exec": maybe_float(feature_summary.loc[device, "median_issue_to_exec"]) if device in feature_summary.index else None,
                "median_occupancy": maybe_float(feature_summary.loc[device, "median_occupancy"]) if device in feature_summary.index else None,
            }
        )

    return sorted(result, key=lambda item: item["device"])


def build_model_matrix(metadata: dict, perf_source: pd.DataFrame, feature_sources: set[str]) -> list[dict]:
    available_by_model: dict[str, int] = {}
    for _, details in metadata.items():
        for model in details.get("models", []):
            if model in {"cuda", "omp"}:
                available_by_model[model] = available_by_model.get(model, 0) + 1

    result = []
    profiled = perf_source.groupby("model_type")["source"].nunique().to_dict()
    feature_profiled = (
        pd.Series(sorted(feature_sources))
        .str.rsplit("-", n=1)
        .str[-1]
        .value_counts()
        .to_dict()
    )

    for model in sorted(available_by_model):
        result.append(
            {
                "model": model,
                "available": int(available_by_model[model]),
                "profiled": int(profiled.get(model, 0)),
                "feature_enriched": int(feature_profiled.get(model, 0)),
                "profiled_ratio": round(profiled.get(model, 0) / available_by_model[model], 4),
                "feature_ratio": round(feature_profiled.get(model, 0) / available_by_model[model], 4),
            }
        )
    return result


def find_agentic_stubs() -> list[str]:
    tree = AGENTIC_DIR / "langchain-tools" / "treesitter-tools"
    stub_files = []
    if not tree.exists():
        return stub_files
    for path in sorted(tree.glob("*.py")):
        text = path.read_text()
        if "NotImplementedError" in text and "TODO" in text:
            stub_files.append(str(path.relative_to(ROOT)))
    return stub_files


def build_top_lists(perf_source: pd.DataFrame) -> dict:
    by_perf = perf_source.sort_values("peak_total_perf", ascending=False).head(20)
    by_ai = perf_source.sort_values("median_total_ai", ascending=False).head(20)
    return {
        "peak_perf_sources": clean_records(
            by_perf[
                ["source", "benchmark", "category", "model_type", "device", "kernel_count", "peak_total_perf", "median_total_ai", "median_xtime"]
            ].reset_index(drop=True)
        ),
        "ai_dense_sources": clean_records(
            by_ai[
                ["source", "benchmark", "category", "model_type", "device", "kernel_count", "peak_total_perf", "median_total_ai", "median_xtime"]
            ].reset_index(drop=True)
        ),
    }


def write_json(name: str, payload: dict | list) -> None:
    path = DATA_DIR / name
    path.write_text(json.dumps(payload, indent=2))


def write_js(name: str, variable_name: str, payload: dict | list) -> None:
    path = DATA_DIR / name
    compact = json.dumps(payload, separators=(",", ":"))
    path.write_text(f"window.{variable_name} = {compact};\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    perf_raw, perf_kernel, perf_source = build_perf_data(metadata)
    feature, cluster_summary = build_feature_data(metadata)

    perf_sources = set(perf_raw["source"].unique())
    feature_sources = set(feature["source"].unique())
    missing_feature_sources = sorted(perf_sources - feature_sources)
    unmapped = sorted(set(perf_source["benchmark"].unique()) - set(metadata.keys()))
    agentic_stub_files = find_agentic_stubs()

    inventory = build_inventory(metadata, perf_sources, feature_sources, unmapped)
    audit = build_audit(metadata, perf_sources, feature_sources, missing_feature_sources, agentic_stub_files)
    device_summary = build_device_summary(perf_kernel, perf_source, feature)
    model_matrix = build_model_matrix(metadata, perf_source, feature_sources)
    top_lists = build_top_lists(perf_source)

    category_profiled = (
        perf_source.groupby(["category", "model_type"])["source"]
        .nunique()
        .reset_index()
        .rename(columns={"source": "profiled_sources"})
    )
    category_profiled["category_slug"] = category_profiled["category"].map(slug)

    hero = {
        "headline_metrics": [
            {"label": "benchmark entries", "value": inventory["totals"]["benchmarks_yaml"]},
            {"label": "GPUs covered", "value": len(device_summary)},
            {"label": "profiled binaries", "value": inventory["totals"]["profiled_sources"]},
            {"label": "kernel-device rows", "value": int(len(perf_kernel))},
        ],
        "subhead": "gpuFLOPBench is a multi-GPU benchmark atlas for comparing source binaries, kernel behavior, and instruction-mix structure across the profiled corpus.",
    }

    perf_kernel_export = perf_kernel[
        [
            "source",
            "benchmark",
            "category",
            "model_type",
            "device",
            "display_kernel",
            "total_ai",
            "total_perf",
            "traffic",
            "bytesTotal",
            "xtime",
            "dominant_perf_type",
        ]
    ].rename(columns={"display_kernel": "kernel"}).copy()
    perf_kernel_export["total_ai"] = perf_kernel_export["total_ai"].round(6)
    perf_kernel_export["total_perf"] = perf_kernel_export["total_perf"].round(2)
    perf_kernel_export["traffic"] = perf_kernel_export["traffic"].round(2)
    perf_kernel_export["bytesTotal"] = perf_kernel_export["bytesTotal"].round(2)
    perf_kernel_export["xtime"] = perf_kernel_export["xtime"].round(2)

    feature_export = feature[
        [
            "source",
            "benchmark",
            "category",
            "model_type",
            "device",
            "display_kernel",
            "total_AI",
            "ipc_active",
            "issue_to_exec",
            "theoretical_occupancy_pct",
            "dominant_op_type",
            "cluster",
            "pc1",
            "pc2",
        ]
    ].rename(
        columns={
            "display_kernel": "kernel",
            "total_AI": "total_ai",
            "theoretical_occupancy_pct": "occupancy_pct",
        }
    )
    feature_export["total_ai"] = feature_export["total_ai"].round(6)
    feature_export["ipc_active"] = feature_export["ipc_active"].round(4)
    feature_export["issue_to_exec"] = feature_export["issue_to_exec"].round(4)
    feature_export["occupancy_pct"] = feature_export["occupancy_pct"].round(2)
    feature_export["pc1"] = feature_export["pc1"].round(4)
    feature_export["pc2"] = feature_export["pc2"].round(4)

    source_export = perf_source[
        [
            "source",
            "benchmark",
            "category",
            "model_type",
            "device",
            "kernel_count",
            "peak_total_perf",
            "median_total_perf",
            "median_total_ai",
            "median_xtime",
            "coverage_rank",
        ]
    ].copy()
    for column in ["peak_total_perf", "median_total_perf", "median_total_ai", "median_xtime", "coverage_rank"]:
        source_export[column] = source_export[column].round(4)

    source_export.to_csv(DATA_DIR / "source-performance.csv", index=False)
    perf_kernel_export.to_csv(DATA_DIR / "kernel-performance.csv", index=False)
    feature_export.to_csv(DATA_DIR / "feature-space.csv", index=False)

    write_json("kernel-performance.json", clean_records(perf_kernel_export))
    write_json("source-performance.json", clean_records(source_export))
    write_json("feature-space.json", clean_records(feature_export))
    metadata_payload = {
        "hero": hero,
        "inventory": inventory,
        "audit": audit,
        "device_summary": device_summary,
        "model_matrix": model_matrix,
        "category_profiled": clean_records(category_profiled),
        "top_lists": top_lists,
        "cluster_summary": cluster_summary,
        "downloads": [
            {
                "label": "Compact source summary CSV",
                "path": "docs/data/source-performance.csv",
                "size_bytes": (DATA_DIR / "source-performance.csv").stat().st_size,
            },
            {
                "label": "Compact kernel summary CSV",
                "path": "docs/data/kernel-performance.csv",
                "size_bytes": (DATA_DIR / "kernel-performance.csv").stat().st_size,
            },
            {
                "label": "Full performance CSV",
                "path": "cuda-profiling/collected-data/all-NCU-GPU-Data.csv",
                "size_bytes": PERF_CSV.stat().st_size,
            },
            {
                "label": "Feature-enriched CSV",
                "path": "cuda-profiling/collected-data/all-NCU-GPU-Data-with-IMIX.csv",
                "size_bytes": FEATURE_CSV.stat().st_size,
            },
            {
                "label": "IMIX CSV",
                "path": "cuda-profiling/collected-data/all-IMIX-Data.csv",
                "size_bytes": IMIX_CSV.stat().st_size if IMIX_CSV.exists() else None,
            },
        ],
    }
    write_json("site-metadata.json", metadata_payload)
    write_js(
        "site-data.js",
        "GPU_FLOWBENCH_DATA",
        {
            "meta": metadata_payload,
            "kernelRows": clean_records(perf_kernel_export),
            "featureRows": clean_records(feature_export),
            "sourceRows": clean_records(source_export),
        },
    )

    print(f"Wrote site datasets to {DATA_DIR}")


if __name__ == "__main__":
    main()
