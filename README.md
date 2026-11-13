# DDColor XAI

Region-based explainability study of the [DDColor](https://github.com/piddnad/DDColor) image colorization model, using Region-Grad-CAM, Region-RISE, query attention and semantic occlusion on COCO and ADE20K. Developed with Mustafa Anakök. See [report.pdf](report.pdf) for details.

## Setup

```bash
git clone https://github.com/piddnad/DDColor.git third_party/DDColor
git -C third_party/DDColor checkout 2adb63f2656ac41cbdf7b894cddd94121a3faf13
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python scripts/prepare_manifest.py coco --images data/coco/val2017 \
  --annotations data/coco/annotations/instances_val2017.json \
  --output data/manifests/coco_val150.jsonl --mask-dir data/masks/coco --limit 150

python scripts/run_experiment.py --manifest data/manifests/coco_val150.jsonl \
  --ddcolor-root third_party/DDColor --output runs/coco
```

The image IDs used in the experiments are listed in `subset_lists/`.
