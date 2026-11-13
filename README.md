# DDColor XAI

Which parts of a grayscale image decide the color of a given region? This project explores that question for [DDColor](https://github.com/piddnad/DDColor) (ICCV 2023). The two main methods are Region-Grad-CAM and Region-RISE. Query-attention maps and semantic occlusion are extra analyses. Everything is evaluated on 150-image subsets of COCO val2017 and ADE20K.

Joint work with Mustafa Anakök. The full write-up is in [report.pdf](report.pdf).

## Findings

- Grad-CAM has the best localization (mean energy inside the mask 0.49, vs 0.26 for RISE) and is about 13x faster.
- RISE gets the best deletion score on ADE20K, but it is slow and noisy.
- In 15.3% of ADE20K images, a neighboring region changes the target's color more than the target region itself.

## Setup

```bash
git clone https://github.com/piddnad/DDColor.git third_party/DDColor
git -C third_party/DDColor checkout 2adb63f2656ac41cbdf7b894cddd94121a3faf13
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
```

The DDColor weights are downloaded from Hugging Face on the first run.

## Running

Build a manifest with `scripts/prepare_manifest.py` (`coco` or `ade20k`), then run:

```bash
python scripts/run_experiment.py \
  --manifest data/manifests/coco_val150.jsonl \
  --ddcolor-root third_party/DDColor \
  --output runs/coco
```

The extra analyses run with `sanity_check.py`, `run_attention_analysis.py` and `run_semantic_occlusion.py`. The image IDs used in the experiments are listed in `subset_lists/`.
