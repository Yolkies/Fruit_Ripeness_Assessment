import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ["apple", "banana", "mango", "orange", "strawberry", "tomato"]


@dataclass
class TrainParams:
    source_dir: str = "data"
    annotations: str = "fruit_merged.json"
    images_subdir: str = "images"
    dataset_dir: str = "dataset"
    val_ratio: float = 0.2
    seed: int = 0
    include_negatives: bool = True
    base_model: str = "yolo26n-seg.pt"
    epochs: int = 50
    patience: int = 20
    imgsz: int = 640
    batch: int = 8
    device: str = "0"
    workers: int = 2
    runs_dir: str = "runs/segment/Fruit_Project"
    run_name: str = "segmentation_run"
    run_name_no_negatives: str = "segmentation_run-no-false"


PARAMS = TrainParams()


def run_name_for(params=None):
    p = params if params is not None else PARAMS
    return p.run_name if p.include_negatives else p.run_name_no_negatives


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _log(message, callback):
    if callback is not None:
        callback(message)


def _polygon_line(annotation, width, height):
    polygon = annotation["segmentation"][0]
    if len(polygon) < 6:
        return None
    coords = []
    for i in range(0, len(polygon) - 1, 2):
        x = min(max(polygon[i] / width, 0.0), 1.0)
        y = min(max(polygon[i + 1] / height, 0.0), 1.0)
        coords.append(f"{x:.6f} {y:.6f}")
    return f"{annotation['category_id']} " + " ".join(coords)


def build_dataset(params=None, log=None):
    p = params if params is not None else PARAMS
    source = resolve_path(p.source_dir)
    images_dir = source / p.images_subdir
    manifest = source / p.annotations

    if not manifest.exists():
        raise FileNotFoundError(f"Annotations not found: {manifest}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    _log(f"Reading {manifest}", log)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    grouped = {}
    for annotation in data["annotations"]:
        grouped.setdefault(annotation["image_id"], []).append(annotation)

    records = []
    missing = 0
    collided = 0
    seen = set()

    for image in data["images"]:
        name = os.path.basename(image["file_name"].replace("\\", "/"))
        stem = os.path.splitext(name)[0]
        if stem in seen:
            collided += 1
            continue
        path = images_dir / name
        if not path.exists():
            missing += 1
            continue

        annotations = grouped.get(image["id"], [])
        if not annotations and not p.include_negatives:
            continue

        lines = []
        for annotation in annotations:
            line = _polygon_line(annotation, image["width"], image["height"])
            if line is not None:
                lines.append(line)

        seen.add(stem)
        records.append((path, lines))

    if not records:
        raise RuntimeError("No usable images found.")

    labelled = sum(1 for _, lines in records if lines)
    _log(f"Usable images: {len(records)} ({labelled} labelled, "
         f"{len(records) - labelled} negatives)", log)
    if missing:
        _log(f"Skipped {missing} entries whose image file was not found.", log)
    if collided:
        _log(f"Skipped {collided} entries sharing a filename stem.", log)

    random.Random(p.seed).shuffle(records)
    split = int(len(records) * (1.0 - p.val_ratio))
    parts = {"train": records[:split], "val": records[split:]}

    root = resolve_path(p.dataset_dir)
    if root.exists():
        shutil.rmtree(root)

    for part, rows in parts.items():
        (root / part / "images").mkdir(parents=True, exist_ok=True)
        (root / part / "labels").mkdir(parents=True, exist_ok=True)
        for path, lines in rows:
            shutil.copy2(path, root / part / "images" / path.name)
            label = (root / part / "labels" / (path.stem + ".txt"))
            label.write_text("\n".join(lines), encoding="utf-8")
        _log(f"{part}: {len(rows)} images", log)

    yaml_path = root / "data.yaml"
    lines = [
        "train: train/images",
        "val: val/images",
        "",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Wrote {yaml_path}", log)

    return yaml_path


def train(data_yaml, params=None, log=None):
    from ultralytics import YOLO

    p = params if params is not None else PARAMS
    name = run_name_for(p)
    _log(f"Training {p.base_model} for {p.epochs} epochs "
         f"(patience {p.patience}) into {p.runs_dir}/{name}", log)

    model = YOLO(p.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=p.epochs,
        patience=p.patience,
        imgsz=p.imgsz,
        batch=p.batch,
        device=p.device,
        workers=p.workers,
        project=str(resolve_path(p.runs_dir)),
        name=name,
        exist_ok=True,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", resolve_path(p.runs_dir) / name))
    weights = save_dir / "weights" / "best.pt"
    try:
        weights = weights.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    _log(f"Training finished. Best weights: {weights}", log)
    return weights, results


def run(params=None, log=None):
    p = params if params is not None else PARAMS
    data_yaml = build_dataset(p, log)
    return train(data_yaml, p, log)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", type=int, choices=(0, 1))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    if args.negatives is not None:
        PARAMS.include_negatives = bool(args.negatives)
    for field in ("epochs", "patience", "batch", "workers", "imgsz", "device"):
        value = getattr(args, field)
        if value is not None:
            setattr(PARAMS, field, value)

    run(log=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
