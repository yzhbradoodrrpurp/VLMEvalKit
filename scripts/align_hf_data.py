#!/usr/bin/env python3
"""Align locally downloaded HF/ModelScope data to VLMEvalKit TSV inputs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_NAME = "yzh_qwen35_27b_4b_global_step_60"
MODEL_PATH = "/root/autodl-fs/models/20260602_yzh_qwen35_27b_4b/global_step_60"


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    class_name: str
    dataset: str
    source: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-data", type=Path, default=Path("/root/autodl-fs/hf_data"))
    parser.add_argument("--lmu-data", type=Path, default=Path("/root/autodl-fs/LMUData"))
    parser.add_argument(
        "--config-out",
        type=Path,
        default=Path("configs/yzh_qwen35_27b_4b.json"),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite generated files and symlinks.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def replace_path(path: Path, *, force: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite it")
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"Refusing to remove directory: {path}")
    path.unlink()


def symlink_tsv(src: Path, dst: Path, *, force: bool) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return True
        replace_path(dst, force=force)
    dst.symlink_to(src)
    return True


def split_vstar_question(text: str) -> tuple[str, dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    option_re = re.compile(r"^\(([A-Z])\)\s*(.+)$")
    question_lines: list[str] = []
    options: dict[str, str] = {}
    for line in lines:
        match = option_re.match(line)
        if match:
            options[match.group(1)] = match.group(2).strip()
            continue
        if line.lower().startswith("answer with"):
            continue
        if not options:
            question_lines.append(line)
    question = "\n".join(question_lines).strip()
    if not question or not options:
        raise ValueError(f"Unable to parse VStar question: {text[:120]}")
    return question, options


def write_tsv(rows: list[dict[str, Any]], path: Path, *, force: bool) -> bool:
    if path.exists() and not force:
        return True
    ensure_dir(path.parent)
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return True


def convert_vstar(hf_data: Path, lmu_data: Path, *, force: bool) -> bool:
    src = hf_data / "vstar_bench" / "test_questions.jsonl"
    if not src.exists():
        return False

    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as handle:
        for row_num, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            question, options = split_vstar_question(str(item["text"]))
            out: dict[str, Any] = {
                "index": item.get("question_id", row_num),
                "question": question,
                "answer": item["label"],
                "category": item.get("category", ""),
                "image_path": str((hf_data / "vstar_bench" / item["image"]).resolve()),
            }
            out.update(options)
            rows.append(out)

    return write_tsv(rows, lmu_data / "VStarBench_LOCAL.tsv", force=force)


def normalize_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
            if isinstance(parsed, list):
                options = parsed
        except json.JSONDecodeError:
            return [options]
    if not isinstance(options, list):
        return []
    return [str(option) for option in options]


def option_text(option: str, label: str) -> str:
    cleaned = option.strip()
    prefix_patterns = (f"{label}.", f"{label})", f"({label})")
    for prefix in prefix_patterns:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


def image_bytes_from_payload(payload: Any) -> bytes | None:
    if payload is None:
        return None
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        if text.startswith("data:image") and "," in text:
            text = text.split(",", 1)[1]
        try:
            return base64.b64decode(text, validate=False)
        except Exception:
            return text.encode()
    return None


def suffix_from_path(path_value: Any, default: str = ".jpg") -> str:
    if not path_value:
        return default
    suffix = Path(str(path_value)).suffix
    return suffix if suffix else default


def write_image(payload: Any, path_value: Any, dst: Path, *, force: bool) -> Path | None:
    image_bytes = image_bytes_from_payload(payload)
    if image_bytes is None:
        candidate = Path(str(path_value)) if path_value else None
        if candidate and candidate.exists():
            return candidate.resolve()
        return None
    if dst.exists():
        return dst.resolve()
    ensure_dir(dst.parent)
    dst.write_bytes(image_bytes)
    return dst.resolve()


def iter_parquet_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        if not path.exists():
            continue
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                yield row


def convert_mme_realworld_lite(hf_data: Path, lmu_data: Path, *, force: bool) -> bool:
    src_dir = hf_data / "MME-RealWorld-lite-lmms-eval" / "data"
    parquet_paths = sorted(src_dir.glob("*.parquet"))
    if not parquet_paths:
        return False

    image_dir = lmu_data / "images" / "MME-RealWorld-Lite_LOCAL"
    rows: list[dict[str, Any]] = []
    for row_num, item in enumerate(iter_parquet_rows(parquet_paths)):
        index = item.get("index", row_num)
        suffix = suffix_from_path(item.get("path"))
        image_path = write_image(
            item.get("bytes"),
            item.get("path"),
            image_dir / f"{index}{suffix}",
            force=force,
        )
        options = normalize_options(item.get("multi-choice options"))
        out: dict[str, Any] = {
            "index": index,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "category": item.get("category", ""),
            "l2-category": item.get("l2-category", ""),
            "image_path": str(image_path) if image_path else "",
            "multi-choice options": json.dumps(options, ensure_ascii=False),
        }
        for label, option in zip("ABCDE", options):
            out[label] = option_text(option, label)
        rows.append(out)

    return write_tsv(rows, lmu_data / "MME-RealWorld-Lite_LOCAL.tsv", force=force)


def split_zoombench_query(text: Any) -> tuple[str, dict[str, str]]:
    normalized = re.sub(r"\s*\|\s*", "\n", str(text).strip())
    parts = [part.strip() for part in normalized.splitlines() if part.strip()]
    question_lines: list[str] = []
    options: dict[str, str] = {}
    instructions: list[str] = []
    option_re = re.compile(r"^([A-D])\.\s*(.+)$", re.DOTALL)
    for part in parts:
        match = option_re.match(part)
        if match:
            option = match.group(2).strip()
            # Pandas treats the literal "None" as NA when VLMEvalKit loads TSVs.
            if option.lower() == "none":
                option = "None of the above"
            options[match.group(1)] = option
        elif options:
            instructions.append(part)
        else:
            question_lines.append(part)
    question = "\n".join(question_lines).strip()
    if instructions:
        question = "\n".join([question, *instructions]).strip()
    return question, options


def convert_zoombench(hf_data: Path, lmu_data: Path, *, force: bool) -> bool:
    src = hf_data / "zoom_bench" / "data" / "test.parquet"
    if not src.exists():
        return False

    image_dir = lmu_data / "images" / "ZoomBench_LOCAL"
    rows: list[dict[str, Any]] = []
    for row_num, item in enumerate(iter_parquet_rows([src])):
        index = item.get("id", row_num)
        payload = item.get("image") or {}
        image_path = None
        if isinstance(payload, dict):
            suffix = suffix_from_path(payload.get("path"))
            image_path = write_image(
                payload.get("bytes"),
                payload.get("path"),
                image_dir / f"{index}_image{suffix}",
                force=force,
            )
        question, options = split_zoombench_query(item.get("query", ""))
        out: dict[str, Any] = {
            "index": index,
            "question": question,
            "answer": item.get("response", ""),
            "answer_type": item.get("question_type", ""),
            "category": item.get("question_type", ""),
            "bbox": json.dumps(item.get("bbox", []), ensure_ascii=False),
            "image_path": str(image_path) if image_path else "",
        }
        out.update(options)
        rows.append(out)

    return write_tsv(rows, lmu_data / "ZoomBench_LOCAL.tsv", force=force)


def build_config(entries: list[DatasetEntry], config_out: Path) -> None:
    ensure_dir(config_out.parent)
    config = {
        "model": {
            MODEL_NAME: {
                "class": "Qwen3VLChat",
                "model_path": MODEL_PATH,
                "use_vllm": True,
                "use_custom_prompt": False,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 1.5,
                "max_new_tokens": 32768,
            }
        },
        "data": {
            entry.name: {
                "class": entry.class_name,
                "dataset": entry.dataset,
            }
            for entry in entries
        },
    }
    config_out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dir(args.lmu_data)

    entries: list[DatasetEntry] = []
    link_specs = [
        DatasetEntry("MMStar", "ImageMCQDataset", "MMStar", args.hf_data / "MMStar" / "MMStar.tsv"),
        DatasetEntry("HRBench4K", "HRBenchDataset", "HRBench4K", args.hf_data / "hr_bench" / "hr_bench_4k.tsv"),
        DatasetEntry("HRBench8K", "HRBenchDataset", "HRBench8K", args.hf_data / "hr_bench" / "hr_bench_8k.tsv"),
    ]
    for entry in link_specs:
        assert entry.source is not None
        if symlink_tsv(entry.source, args.lmu_data / f"{entry.dataset}.tsv", force=args.force):
            entries.append(entry)
            print(f"linked {entry.dataset}: {entry.source}")
        else:
            print(f"missing {entry.dataset}: {entry.source}")

    converters = [
        (DatasetEntry("VStarBench_LOCAL", "CustomMCQDataset", "VStarBench_LOCAL"), convert_vstar),
        (DatasetEntry("MME-RealWorld-Lite_LOCAL", "CustomMCQDataset", "MME-RealWorld-Lite_LOCAL"),
         convert_mme_realworld_lite),
        (DatasetEntry("ZoomBench_LOCAL", "ZoomBenchDataset", "ZoomBench_LOCAL"), convert_zoombench),
    ]
    for entry, converter in converters:
        if converter(args.hf_data, args.lmu_data, force=args.force):
            entries.append(entry)
            print(f"generated {entry.dataset}: {args.lmu_data / (entry.dataset + '.tsv')}")
        else:
            print(f"missing source for {entry.dataset}")

    build_config(entries, args.config_out)
    print(f"wrote config: {args.config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
