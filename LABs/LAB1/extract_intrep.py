"""Extract Qwen3-4B-Base decoder-block activations from a sentence CSV.

Provided features use revision 906bfd4b4dc7f14ee4320094d8b41684abff8539,
zero-based block outputs BEFORE final RMSNorm, and no chat template.
CUDA is required. Output: <input-stem>.layer<layer>.npy, float16, CSV row order.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_rows(path: Path, selection: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sentence", "country"} if selection == "country" else {"sentence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows or any(not (row.get("sentence") or "").strip() for row in rows):
        raise ValueError("CSV must contain nonempty sentences")
    if selection == "country":
        for row in rows:
            country_span(row)
    return rows


def country_span(row: dict[str, str]) -> tuple[int, int]:
    sentence, country = row["sentence"], row.get("country") or ""
    start = sentence.rfind(country) if country else -1
    end = start + len(country)
    if start < 0 or sentence[end:].strip() != "?":
        raise ValueError("Country must occur at the end of the sentence before ?")
    return start, end


def select_token(row: dict[str, str], offsets: list[list[int]], selection: str) -> int:
    # Padding has offset (0, 0); sentence-final punctuation counts as a token.
    if selection == "last":
        candidates = [i for i, (start, end) in enumerate(offsets) if end > start]
    else:
        start, end = country_span(row)
        candidates = [
            i
            for i, (left, right) in enumerate(offsets)
            if right > left and left < end and right > start
        ]
    if not candidates:
        raise ValueError("No token found for the requested selection")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Local Qwen3-4B-Base model directory",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with a sentence column (and country for world)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        choices=range(36),
        required=True,
        help="Zero-based decoder block index, 0 through 35",
    )
    parser.add_argument(
        "--token-selection",
        choices=("last", "country"),
        required=True,
        help="Last sentence token or last country token",
    )
    return parser.parse_args()


def load_model(model_dir: Path):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; run this script in a GPU job")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for token offsets")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # AutoModel omits the language-model head; only sentence text is provided.
    model = (
        AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype)
        .to("cuda")
        .eval()
    )
    if (
        model.config.model_type != "qwen3"
        or model.config.hidden_size != 2560
        or len(model.layers) != 36
    ):
        raise ValueError("Expected Qwen3-4B-Base with 36 blocks and width 2560")
    return tokenizer, model


def extract_features(
    rows, tokenizer, model, *, layer: int, selection: str
) -> np.ndarray:
    import torch

    result = np.empty((len(rows), 2560), dtype=np.float16)
    token_indices = []

    def capture(_module, _inputs, outputs):
        # A block hook avoids confusing the final block with final RMSNorm.
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        batch_indices = torch.arange(len(token_indices), device=hidden.device)
        selected = hidden[batch_indices, token_indices, :]
        selected = selected.detach().float().cpu().numpy()
        result[offset : offset + len(token_indices)] = selected

    handle = model.layers[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                batch = rows[offset : offset + 8]
                tokens = tokenizer(
                    [row["sentence"] for row in batch],
                    padding=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=False,
                )
                spans = tokens.pop("offset_mapping").tolist()
                token_indices = [
                    select_token(row, span, selection)
                    for row, span in zip(batch, spans, strict=True)
                ]
                if tokens.input_ids.shape[1] > model.config.max_position_embeddings:
                    raise ValueError("Input exceeds the model context length")
                model(**tokens.to("cuda"), use_cache=False)
                print(f"Extracted {offset + len(batch)}/{len(rows)}", flush=True)
    finally:
        handle.remove()
    if not np.isfinite(result).all():
        raise ValueError("Extracted features contain non-finite values")
    return result


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.token_selection)
    output = args.input.with_name(f"{args.input.stem}.layer{args.layer}.npy")
    if output.exists():
        raise FileExistsError(f"Already exists: {output}; copy the CSV elsewhere")
    tokenizer, model = load_model(args.model_dir)
    features = extract_features(
        rows, tokenizer, model, layer=args.layer, selection=args.token_selection
    )
    # Exclusive creation also protects against another job writing this path.
    with output.open("xb") as stream:
        np.save(stream, features, allow_pickle=False)
    print(f"Saved {output}: {features.shape}, {features.dtype}")


if __name__ == "__main__":
    main()
