from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def polygon_mask(
    segmentation: list,
    height: int,
    width: int,
) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=np.uint8)
    valid_polygon = False
    for polygon in segmentation:
        coordinates = np.asarray(polygon, dtype=np.float32)
        if coordinates.size < 6 or coordinates.size % 2 != 0:
            continue
        points = np.rint(coordinates.reshape(-1, 2)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 255)
        valid_polygon = True
    return mask if valid_polygon and mask.any() else None


def prepare_coco(args: argparse.Namespace) -> list[dict]:
    images_root = Path(args.images).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    mask_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {item["id"]: item for item in coco["images"]}
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        if not isinstance(annotation.get("segmentation"), list):
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_ids = list(annotations_by_image)
    random.Random(args.seed).shuffle(image_ids)
    records: list[dict] = []

    for image_id in tqdm(image_ids, desc="COCO masks"):
        image_info = images[image_id]
        image_area = image_info["height"] * image_info["width"]
        candidates = []
        for annotation in annotations_by_image[image_id]:
            ratio = float(annotation.get("area", 0.0)) / max(image_area, 1)
            if args.min_area <= ratio <= args.max_area:
                candidates.append(annotation)
        candidates.sort(key=lambda item: float(item.get("area", 0.0)), reverse=True)

        selected = None
        selected_mask = None
        for candidate in candidates:
            mask = polygon_mask(
                candidate["segmentation"],
                image_info["height"],
                image_info["width"],
            )
            if mask is not None:
                selected = candidate
                selected_mask = mask
                break
        if selected is None or selected_mask is None:
            continue

        image_path = (images_root / image_info["file_name"]).resolve()
        if not image_path.is_file():
            continue

        mask_path = (mask_dir / f"{image_id:012d}_{selected['id']}.png").resolve()
        cv2.imwrite(str(mask_path), selected_mask)
        category_id = int(selected["category_id"])
        records.append(
            {
                "id": f"coco_{image_id}_{selected['id']}",
                "dataset": "COCO-val2017",
                "image": str(image_path),
                "mask": str(mask_path),
                "label": categories.get(category_id, str(category_id)),
                "category_id": category_id,
            }
        )
        if len(records) >= args.limit:
            break
    return records


def find_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_ade20k_labels(annotations_root: Path) -> dict[int, str]:
    info_path = annotations_root.parent.parent / "objectInfo150.txt"
    if not info_path.is_file():
        return {}

    labels: dict[int, str] = {}
    with info_path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 5:
                labels[int(fields[0])] = fields[4]
    return labels


def prepare_ade20k(args: argparse.Namespace) -> list[dict]:
    images_root = Path(args.images).expanduser().resolve()
    annotations_root = Path(args.annotations).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    mask_dir.mkdir(parents=True, exist_ok=True)
    labels = read_ade20k_labels(annotations_root)

    image_paths = find_images(images_root)
    random.Random(args.seed).shuffle(image_paths)
    records: list[dict] = []

    for image_path in tqdm(image_paths, desc="ADE20K masks"):
        relative = image_path.relative_to(images_root).with_suffix(".png")
        annotation_path = annotations_root / relative
        annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            continue

        height, width = annotation.shape
        total = height * width
        counts = Counter(annotation.reshape(-1).tolist())
        candidates = [
            (class_id, count)
            for class_id, count in counts.items()
            if class_id not in (0, 255)
            and args.min_area <= count / max(total, 1) <= args.max_area
        ]
        if not candidates:
            continue
        class_id, _ = max(candidates, key=lambda item: item[1])
        binary_mask = (annotation == class_id).astype(np.uint8) * 255

        sample_stem = "_".join(relative.with_suffix("").parts)
        mask_path = (mask_dir / f"{sample_stem}_class_{class_id}.png").resolve()
        cv2.imwrite(str(mask_path), binary_mask)
        records.append(
            {
                "id": f"ade20k_{sample_stem}_{class_id}",
                "dataset": "ADE20K-validation",
                "image": str(image_path.resolve()),
                "mask": str(mask_path),
                "label": labels.get(class_id, f"class_{class_id}"),
                "category_id": int(class_id),
            }
        )
        if len(records) >= args.limit:
            break
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build XAI experiment manifests")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    for dataset in ("coco", "ade20k"):
        subparser = subparsers.add_parser(dataset)
        subparser.add_argument("--images", required=True)
        subparser.add_argument("--annotations", required=True)
        subparser.add_argument("--output", required=True)
        subparser.add_argument("--mask-dir", required=True)
        subparser.add_argument("--limit", type=int, default=250)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--min-area", type=float, default=0.05)
        subparser.add_argument("--max-area", type=float, default=0.70)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.min_area < args.max_area <= 1.0:
        parser.error("Area limits must satisfy 0 < min-area < max-area <= 1")

    if args.dataset == "coco":
        records = prepare_coco(args)
    else:
        records = prepare_ade20k(args)

    write_jsonl(records, Path(args.output).expanduser().resolve())
    print(f"Wrote {len(records)} samples to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
