# DDColor XAI

Which parts of a grayscale image decide the color of a given region? This project explores that question for [DDColor](https://github.com/piddnad/DDColor) (ICCV 2023), an image colorization model. It adapts Grad-CAM and RISE from classification to dense color output, and adds query-attention maps and semantic occlusion as supporting analyses. All experiments use 150-image subsets of COCO and ADE20K.

Developed with Mustafa Anakök. The full write-up is in [report.pdf](report.pdf).

## Findings

- Region-Grad-CAM gives the best localization (mean energy inside the mask 0.49, vs 0.26 for RISE) and is about 13 times faster.
- Region-RISE gets the best deletion score on ADE20K, but it is slower and noisier.
- In 15.3% of ADE20K images, a neighboring region changes the target's color more than the target region itself.
