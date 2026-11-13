# 🎨 DDColor XAI

How does an AI decide what color the sky should be? 🤔

This project looks inside [DDColor](https://github.com/piddnad/DDColor), a state-of-the-art image colorization model, to see which parts of a black-and-white photo shape the colors it chooses. We adapted explainability methods such as Grad-CAM and RISE to colorization and tested them on the COCO and ADE20K datasets.

## ✨ Highlights

- 🔍 Grad-CAM pointed to the right regions most reliably, and it was also the fastest method.
- 🌳 The model mostly relies on the region itself, but in about 15% of scenes a neighbor matters more, such as trees next to a building.

📄 [Full report](report.pdf)
