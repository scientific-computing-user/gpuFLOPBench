#!/usr/bin/env python3
"""Build public site datasets from the benchmark repository artifacts."""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
DOWNLOADS_DIR = DOCS_DIR / "downloads"
COLLECTED_DIR = ROOT / "cuda-profiling" / "collected-data"
HECBENCH_DIR = ROOT / "HeCBench"

PERF_CSV = COLLECTED_DIR / "all-NCU-GPU-Data.csv"
FEATURE_CSV = COLLECTED_DIR / "all-NCU-GPU-Data-with-IMIX.csv"
IMIX_CSV = COLLECTED_DIR / "all-IMIX-Data.csv"
BENCHMARKS_YAML = HECBENCH_DIR / "benchmarks.yaml"
HECBENCH_README = HECBENCH_DIR / "README.md"

DEVICE_NAME_MAP = {
    "NVIDIA GeForce RTX 3080": "3080",
    "NVIDIA A10": "A10",
    "NVIDIA A100-SXM4-40GB": "A100",
    "NVIDIA H100 80GB HBM3": "H100",
}

GPU_SPECS = {
    "3080": {
        "label": "RTX 3080 (PCIe)",
        "architecture": "Ampere (GA102)",
        "compute_capability": "8.6",
        "base_mem_mhz": 9500,
        "base_core_mhz": 1440,
        "sm_count": 68,
        "memory_type": "GDDR6X",
        "tdp_w": 320,
        "global_memory_gb": 10,
        "memory_bandwidth_gbps": 760,
        "l2_cache_mb": 5,
        "l1_cache_kb": 128,
        "cuda_cores_per_sm": 128,
        "shared_memory": "up to 100 KB",
        "peak_tflops": {"fp16": 30.55, "fp32": 30.55, "fp64": 0.477},
    },
    "A10": {
        "label": "A10 (PCIe)",
        "architecture": "Ampere (GA102)",
        "compute_capability": "8.6",
        "base_mem_mhz": 6251,
        "base_core_mhz": 885,
        "sm_count": 72,
        "memory_type": "GDDR6",
        "tdp_w": 150,
        "global_memory_gb": 24,
        "memory_bandwidth_gbps": 600,
        "l2_cache_mb": 6,
        "l1_cache_kb": 128,
        "cuda_cores_per_sm": 128,
        "shared_memory": "up to 100 KB",
        "peak_tflops": {"fp16": 15.62, "fp32": 15.62, "fp64": 0.244},
    },
    "A100": {
        "label": "A100 (SXM4)",
        "architecture": "Ampere (GA100)",
        "compute_capability": "8.0",
        "base_mem_mhz": 1215,
        "base_core_mhz": 1095,
        "sm_count": 108,
        "memory_type": "HBM2e",
        "tdp_w": 400,
        "global_memory_gb": 40,
        "memory_bandwidth_gbps": 1555,
        "l2_cache_mb": 40,
        "l1_cache_kb": 192,
        "cuda_cores_per_sm": 64,
        "shared_memory": "up to 164 KB",
        "peak_tflops": {"fp16": 77.97, "fp32": 19.49, "fp64": 9.75},
    },
    "H100": {
        "label": "H100 (SXM5)",
        "architecture": "Hopper (GH100)",
        "compute_capability": "9.0",
        "base_mem_mhz": 1313,
        "base_core_mhz": 1590,
        "sm_count": 132,
        "memory_type": "HBM3",
        "tdp_w": 700,
        "global_memory_gb": 80,
        "memory_bandwidth_gbps": 3360,
        "l2_cache_mb": 50,
        "l1_cache_kb": 256,
        "cuda_cores_per_sm": 128,
        "shared_memory": "up to 228 KB",
        "peak_tflops": {"fp16": 133.82, "fp32": 66.91, "fp64": 33.45},
    },
}

PAPER_BIBTEX = """@inproceedings{boletCanLargeLanguage2025a,
  title = {Can {{Large Language Models Predict Parallel Code Performance}}?},
  booktitle = {Proceedings of the 34th {{International Symposium}} on {{High-Performance Parallel}} and {{Distributed Computing}}},
  author = {Bolet, Gregory and Georgakoudis, Giorgis and Menon, Harshitha and Parasyris, Konstantinos and Hasabnis, Niranjan and Estes, Hayden and Cameron, Kirk and Oren, Gal},
  date = {2025-09-09},
  series = {{{HPDC}} '25},
  pages = {1--6},
  publisher = {Association for Computing Machinery},
  location = {New York, NY, USA},
  doi = {10.1145/3731545.3743645},
  url = {https://dl.acm.org/doi/10.1145/3731545.3743645},
  urldate = {2026-02-21},
  isbn = {979-8-4007-1869-4},
  keywords = {llms4PerfPrediction}
}"""

TEAM_MEMBERS = [
    {
        "name": "Gregory Bolet",
        "affiliation": "Virginia Tech",
        "profile_url": "https://people.cs.vt.edu/gbolet/",
        "image_path": "./assets/team/gregory-bolet.jpg",
    },
    {
        "name": "Giorgis Georgakoudis",
        "affiliation": "Lawrence Livermore National Laboratory",
        "profile_url": "https://people.llnl.gov/georgakoudis1",
        "image_path": "./assets/team/giorgis-georgakoudis.png",
    },
    {
        "name": "Harshitha Menon",
        "affiliation": "Lawrence Livermore National Laboratory",
        "profile_url": "https://www.harshithamenon.com/",
        "image_path": "./assets/team/harshitha-menon.jpg",
    },
    {
        "name": "Konstantinos Parasyris",
        "affiliation": "Lawrence Livermore National Laboratory",
        "profile_url": "https://www.ashes-hpc.org/2025/program.html",
        "image_path": "./assets/team/konstantinos-parasyris.png",
    },
    {
        "name": "Niranjan Hasabnis",
        "affiliation": "Code Metal",
        "profile_url": "https://www.codemetal.ai/about",
        "image_path": "./assets/team/niranjan-hasabnis.jpg",
    },
    {
        "name": "Hayden Estes",
        "affiliation": "Virginia Tech",
        "profile_url": "https://www.linkedin.com/in/haydenvestes/",
        "image_path": "./assets/team/hayden-estes.svg",
    },
    {
        "name": "Kirk Cameron",
        "affiliation": "Virginia Tech",
        "profile_url": "https://website.cs.vt.edu/people/faculty/kirk-cameron.html",
        "image_path": "./assets/team/kirk-cameron.jpg",
    },
    {
        "name": "Gal Oren",
        "affiliation": "Stanford University",
        "profile_url": "https://profiles.stanford.edu/galoren",
        "image_path": "./assets/team/gal-oren.jpg",
    },
]


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "unknown"


def maybe_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def clean_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def normalize_device_name(name: str) -> str:
    return DEVICE_NAME_MAP.get(name, name)


def normalize_omp_symbol(symbol: str) -> str:
    match = re.search(r"([A-Za-z0-9]+_l\d+)$", symbol or "")
    return match.group(1) if match else (symbol or "unknown")


def simplify_demangled(name: str) -> str:
    simple = (name or "").strip()
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


def parse_hecbench_categories() -> tuple[dict[str, str], list[str]]:
    text = HECBENCH_README.read_text()
    category_map: dict[str, str] = {}
    category_order: list[str] = []
    in_section = False
    current_category: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# Benchmark categories"):
            in_section = True
            continue
        if in_section and line.startswith("# Run a benchmark"):
            break
        if not in_section:
            continue
        if line.startswith("### "):
            current_category = line.replace("### ", "", 1).strip()
            category_order.append(current_category)
            continue
        if current_category and line.startswith("    "):
            for benchmark in [item.strip() for item in line.strip().split(",")]:
                if benchmark:
                    category_map[benchmark] = current_category

    return category_map, category_order


def load_metadata() -> dict:
    with BENCHMARKS_YAML.open() as handle:
        return yaml.safe_load(handle)


def benchmark_name_from_source(source: str) -> str:
    return re.sub(r"-(cuda|hip|omp|sycl)$", "", source)


def category_for_benchmark(benchmark: str, metadata: dict, readme_categories: dict[str, str]) -> str:
    if benchmark in readme_categories:
        return readme_categories[benchmark]
    details = metadata.get(benchmark, {})
    return (details.get("categories") or ["uncategorized"])[0]


def dominant_precision(row: pd.Series) -> str:
    counts = {
        "fp64": maybe_float(row["DP_FLOP"]) or 0.0,
        "fp32": maybe_float(row["SP_FLOP"]) or 0.0,
        "fp16": maybe_float(row["HP_FLOP"]) or 0.0,
    }
    if max(counts.values()) <= 0:
        return "int-only"
    return max(counts.items(), key=lambda item: item[1])[0]


def build_inventory(metadata: dict, perf_sources: set[str], category_map: dict[str, str]) -> dict:
    available_by_model: dict[str, int] = {}
    available_by_category: dict[str, int] = {}

    for benchmark, details in metadata.items():
        for model in details.get("models", []):
            available_by_model[model] = available_by_model.get(model, 0) + 1
        category = category_for_benchmark(benchmark, metadata, category_map)
        available_by_category[category] = available_by_category.get(category, 0) + 1

    profiled_benchmarks = {benchmark_name_from_source(source) for source in perf_sources}
    categorized_profiled = {name for name in profiled_benchmarks if name in category_map}

    return {
        "totals": {
            "benchmarks_yaml": len(metadata),
            "profiled_benchmarks": len(profiled_benchmarks),
            "profiled_sources": len(perf_sources),
            "categorized_profiled_benchmarks": len(categorized_profiled),
        },
        "models_available": [{"model": model, "count": count} for model, count in sorted(available_by_model.items())],
        "categories_available": [{"category": name, "count": count} for name, count in sorted(available_by_category.items(), key=lambda item: item[1], reverse=True)],
        "category_source": "HeCBench README.md",
    }


def build_audit(category_map: dict[str, str]) -> dict:
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "category_source": str(HECBENCH_README.relative_to(ROOT)),
        "categorized_benchmarks": len(category_map),
    }


def build_perf_data(metadata: dict, category_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        "bytesTotal",
        "xtime",
    ]
    perf = pd.read_csv(PERF_CSV, usecols=usecols, low_memory=False)
    perf["device_label"] = perf["device"]
    perf["device"] = perf["device"].map(normalize_device_name)
    perf["benchmark"] = perf["source"].map(benchmark_name_from_source)
    perf["category"] = perf["benchmark"].map(lambda name: category_for_benchmark(name, metadata, category_map))
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
    metric_cols = ["SP_FLOP", "DP_FLOP", "HP_FLOP", "INTOP", "bytesTotal", "xtime"]
    perf_kernel = perf.groupby(key_cols, dropna=False)[metric_cols].mean().reset_index()
    perf_kernel["float_flops"] = perf_kernel[["SP_FLOP", "DP_FLOP", "HP_FLOP"]].fillna(0).sum(axis=1)
    perf_kernel["int_ops"] = perf_kernel["INTOP"].fillna(0)
    perf_kernel["performance_tflops"] = np.where(
        perf_kernel["xtime"] > 0,
        perf_kernel["float_flops"] / (perf_kernel["xtime"] * 1_000.0),
        np.nan,
    )
    perf_kernel["arithmetic_intensity"] = np.where(
        perf_kernel["bytesTotal"] > 0,
        perf_kernel["float_flops"] / perf_kernel["bytesTotal"],
        np.nan,
    )
    perf_kernel["dominant_precision"] = perf_kernel.apply(dominant_precision, axis=1)
    perf_kernel["kernel_row_id"] = perf_kernel.apply(
        lambda row: slug(
            f"{row['device']}-{row['model_type']}-{row['source']}-{row['display_kernel']}-{row['Block Size']}-{row['Grid Size']}-{row['exeArgs']}"
        ),
        axis=1,
    )

    source_group_cols = ["source", "benchmark", "category", "codename", "model_type", "device", "device_label"]
    perf_source = (
        perf_kernel.groupby(source_group_cols, dropna=False)
        .agg(
            kernel_count=("display_kernel", "nunique"),
            peak_performance_tflops=("performance_tflops", "max"),
            median_performance_tflops=("performance_tflops", "median"),
            median_arithmetic_intensity=("arithmetic_intensity", "median"),
            median_xtime_ns=("xtime", "median"),
            max_float_flops=("float_flops", "max"),
        )
        .reset_index()
    )
    perf_source["coverage_rank"] = (
        perf_source.groupby(["device", "model_type"])["peak_performance_tflops"]
        .rank(ascending=False, method="dense")
    )

    return perf, perf_kernel, perf_source


def build_device_summary(perf_kernel: pd.DataFrame, perf_source: pd.DataFrame) -> list[dict]:
    device_summary = []
    for device, subset in perf_kernel.groupby("device"):
        source_subset = perf_source[perf_source["device"] == device]
        full_name = subset["device_label"].iloc[0]
        specs = GPU_SPECS.get(device, {})
        positive_perf = subset.loc[subset["performance_tflops"] > 0, "performance_tflops"]
        positive_ai = subset.loc[subset["arithmetic_intensity"] > 0, "arithmetic_intensity"]
        models = source_subset.groupby("model_type")["source"].nunique().sort_values(ascending=False).to_dict()
        device_summary.append(
            {
                "device": device,
                "label": specs.get("label", full_name),
                "full_name": full_name,
                "architecture": specs.get("architecture", "unknown"),
                "compute_capability": specs.get("compute_capability", "unknown"),
                "memory_bandwidth_gbps": specs.get("memory_bandwidth_gbps"),
                "peak_fp16_tflops": specs.get("peak_tflops", {}).get("fp16"),
                "peak_fp32_tflops": specs.get("peak_tflops", {}).get("fp32"),
                "peak_fp64_tflops": specs.get("peak_tflops", {}).get("fp64"),
                "rows": int(len(subset)),
                "benchmarks": int(source_subset["benchmark"].nunique()),
                "sources": int(source_subset["source"].nunique()),
                "kernels": int(subset["display_kernel"].nunique()),
                "models": [{"model": model, "sources": int(count)} for model, count in models.items()],
                "median_performance_tflops": maybe_float(positive_perf.median()),
                "p95_performance_tflops": maybe_float(positive_perf.quantile(0.95)) if not positive_perf.empty else None,
                "median_arithmetic_intensity": maybe_float(positive_ai.median()),
                "median_xtime_ns": maybe_float(subset["xtime"].median()),
            }
        )
    return sorted(device_summary, key=lambda item: item["device"])


def build_model_matrix(metadata: dict, perf_source: pd.DataFrame) -> list[dict]:
    available_by_model: dict[str, int] = {}
    for details in metadata.values():
        for model in details.get("models", []):
            available_by_model[model] = available_by_model.get(model, 0) + 1

    profiled_by_model = perf_source.groupby("model_type")["source"].nunique().to_dict()
    result = []
    for model, available in sorted(available_by_model.items()):
        profiled = int(profiled_by_model.get(model, 0))
        result.append(
            {
                "model": model,
                "available": int(available),
                "profiled": profiled,
                "profiled_ratio": round(profiled / available, 4) if available else 0.0,
            }
        )
    return result


def build_top_lists(perf_source: pd.DataFrame) -> dict:
    perf_positive = perf_source[perf_source["peak_performance_tflops"] > 0].sort_values("peak_performance_tflops", ascending=False)
    ai_positive = perf_source[perf_source["median_arithmetic_intensity"] > 0].sort_values("median_arithmetic_intensity", ascending=False)

    fields = [
        "source",
        "benchmark",
        "category",
        "model_type",
        "device",
        "kernel_count",
        "peak_performance_tflops",
        "median_arithmetic_intensity",
        "median_xtime_ns",
    ]
    return {
        "performance_sources": clean_records(perf_positive.head(20)[fields].reset_index(drop=True)),
        "ai_dense_sources": clean_records(ai_positive.head(20)[fields].reset_index(drop=True)),
    }


def write_json(name: str, payload: dict | list) -> None:
    (DATA_DIR / name).write_text(json.dumps(payload, indent=2))


def write_js(name: str, variable_name: str, payload: dict | list) -> None:
    compact = json.dumps(payload, separators=(",", ":"))
    (DATA_DIR / name).write_text(f"window.{variable_name} = {compact};\n")


def write_gzip_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as src, gzip.open(destination, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    for obsolete in ["feature-space.csv", "feature-space.json"]:
        path = DATA_DIR / obsolete
        if path.exists():
            path.unlink()

    metadata = load_metadata()
    readme_categories, category_order = parse_hecbench_categories()
    perf_raw, perf_kernel, perf_source = build_perf_data(metadata, readme_categories)

    perf_sources = set(perf_raw["source"].unique())
    inventory = build_inventory(metadata, perf_sources, readme_categories)
    audit = build_audit(readme_categories)
    device_summary = build_device_summary(perf_kernel, perf_source)
    model_matrix = build_model_matrix(metadata, perf_source)
    top_lists = build_top_lists(perf_source)

    category_profiled = (
        perf_source.groupby(["category", "model_type"])["source"]
        .nunique()
        .reset_index()
        .rename(columns={"source": "profiled_sources"})
    )
    totals = category_profiled.groupby("category")["profiled_sources"].sum().sort_values(ascending=False)
    category_profiled["category_sort"] = category_profiled["category"].map(totals.to_dict())
    category_profiled["category_order"] = category_profiled["category"].map({name: index for index, name in enumerate(category_order)})
    category_profiled = category_profiled.sort_values(["category_sort", "category_order", "category", "model_type"], ascending=[False, True, True, True])

    hero = {
        "headline_metrics": [
            {"label": "benchmark entries", "value": inventory["totals"]["benchmarks_yaml"]},
            {"label": "GPUs covered", "value": len(device_summary)},
            {"label": "profiled binaries", "value": inventory["totals"]["profiled_sources"]},
            {"label": "kernel-device rows", "value": int(len(perf_kernel))},
        ],
        "subhead": "gpuFLOPBench is a multi-GPU benchmark atlas for floating-point rooflines, source-level coverage, and exact kernel exploration across the profiled corpus.",
    }

    perf_kernel_export = perf_kernel[
        [
            "kernel_row_id",
            "source",
            "benchmark",
            "category",
            "model_type",
            "device",
            "display_kernel",
            "Block Size",
            "Grid Size",
            "exeArgs",
            "SP_FLOP",
            "DP_FLOP",
            "HP_FLOP",
            "INTOP",
            "float_flops",
            "bytesTotal",
            "xtime",
            "arithmetic_intensity",
            "performance_tflops",
            "dominant_precision",
        ]
    ].rename(
        columns={
            "display_kernel": "kernel",
            "Block Size": "block_size",
            "Grid Size": "grid_size",
            "exeArgs": "exe_args",
            "INTOP": "int_ops",
            "bytesTotal": "bytes_total",
            "xtime": "xtime_ns",
        }
    ).copy()
    for column in ["SP_FLOP", "DP_FLOP", "HP_FLOP", "int_ops", "float_flops", "bytes_total", "xtime_ns"]:
        perf_kernel_export[column] = perf_kernel_export[column].round(2)
    perf_kernel_export["arithmetic_intensity"] = perf_kernel_export["arithmetic_intensity"].round(6)
    perf_kernel_export["performance_tflops"] = perf_kernel_export["performance_tflops"].round(6)

    source_export = perf_source[
        [
            "source",
            "benchmark",
            "category",
            "model_type",
            "device",
            "kernel_count",
            "peak_performance_tflops",
            "median_performance_tflops",
            "median_arithmetic_intensity",
            "median_xtime_ns",
            "coverage_rank",
        ]
    ].copy()
    for column in [
        "peak_performance_tflops",
        "median_performance_tflops",
        "median_arithmetic_intensity",
        "median_xtime_ns",
        "coverage_rank",
    ]:
        source_export[column] = source_export[column].round(6)

    source_export.to_csv(DATA_DIR / "source-performance.csv", index=False)
    perf_kernel_export.to_csv(DATA_DIR / "kernel-performance.csv", index=False)
    write_json("kernel-performance.json", clean_records(perf_kernel_export))
    write_json("source-performance.json", clean_records(source_export))

    perf_gz = DOWNLOADS_DIR / "all-NCU-GPU-Data.csv.gz"
    feature_gz = DOWNLOADS_DIR / "all-NCU-GPU-Data-with-IMIX.csv.gz"
    imix_gz = DOWNLOADS_DIR / "all-IMIX-Data.csv.gz"
    write_gzip_copy(PERF_CSV, perf_gz)
    write_gzip_copy(FEATURE_CSV, feature_gz)
    if IMIX_CSV.exists():
        write_gzip_copy(IMIX_CSV, imix_gz)

    metadata_payload = {
        "hero": hero,
        "inventory": inventory,
        "audit": audit,
        "device_summary": device_summary,
        "model_matrix": model_matrix,
        "category_profiled": clean_records(category_profiled.drop(columns=["category_sort", "category_order"])),
        "top_lists": top_lists,
        "roofline_specs": [
            {
                "device": device,
                "label": specs["label"],
                "architecture": specs["architecture"],
                "compute_capability": specs["compute_capability"],
                "memory_bandwidth_gbps": specs["memory_bandwidth_gbps"],
                "peak_fp16_tflops": specs["peak_tflops"]["fp16"],
                "peak_fp32_tflops": specs["peak_tflops"]["fp32"],
                "peak_fp64_tflops": specs["peak_tflops"]["fp64"],
            }
            for device, specs in GPU_SPECS.items()
        ],
        "paper": {
            "title": "Can Large Language Models Predict Parallel Code Performance?",
            "venue": "HPDC '25",
            "doi_url": "https://dl.acm.org/doi/10.1145/3731545.3743645",
            "pdf_url": "https://arxiv.org/pdf/2505.03988",
            "pdf_label": "arXiv PDF",
            "bibtex": PAPER_BIBTEX,
        },
        "team": TEAM_MEMBERS,
        "downloads": [
            {
                "label": "Compact source summary CSV",
                "path": "data/source-performance.csv",
                "href": "./data/source-performance.csv",
                "size_bytes": (DATA_DIR / "source-performance.csv").stat().st_size,
            },
            {
                "label": "Compact kernel summary CSV",
                "path": "data/kernel-performance.csv",
                "href": "./data/kernel-performance.csv",
                "size_bytes": (DATA_DIR / "kernel-performance.csv").stat().st_size,
            },
            {
                "label": "Full performance CSV.gz",
                "path": "downloads/all-NCU-GPU-Data.csv.gz",
                "href": "./downloads/all-NCU-GPU-Data.csv.gz",
                "size_bytes": perf_gz.stat().st_size,
            },
            {
                "label": "IMIX-enriched CSV.gz",
                "path": "downloads/all-NCU-GPU-Data-with-IMIX.csv.gz",
                "href": "./downloads/all-NCU-GPU-Data-with-IMIX.csv.gz",
                "size_bytes": feature_gz.stat().st_size,
            },
            {
                "label": "Raw IMIX CSV.gz",
                "path": "downloads/all-IMIX-Data.csv.gz",
                "href": "./downloads/all-IMIX-Data.csv.gz",
                "size_bytes": imix_gz.stat().st_size if imix_gz.exists() else None,
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
            "sourceRows": clean_records(source_export),
        },
    )

    print(f"Wrote site datasets to {DATA_DIR}")


if __name__ == "__main__":
    main()
