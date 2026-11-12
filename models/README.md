---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: image-segmentation
tags:
  - ultralytics
  - yolo26
  - instance-segmentation
  - reinforced-concrete
  - damage-detection
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - map50
---

# RC damage segmenter: outlines cracks, spalling, exposed and corroded rebar and crushing on photos of concrete structures

This model finds and outlines five types of damage on photos of reinforced-concrete structures: **crack, spalling, rebar exposure, rebar corrosion and concrete crushing**. For every damage region it returns a mask, a box, a class and a confidence. It is meant for facility managers and inspectors who need to **screen many photos and decide which elements an engineer should look at first**. It is a YOLO26n-seg (… parameters) fine-tuned from `yolo26n-seg.pt` (pretrained on COCO) on 2,119 annotated photos. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/Group4-day4-assignment
- **Team:** Group 4

## Model at a glance

![Predictions next to the ground truth](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/predictions.png)

*Test photos: ground truth (middle) and prediction (right), from the best (top) to the worst (bottom). TODO: one sentence on the mistakes in the bottom rows.*

| Model | mask mAP50 (test) | mask mAP50-95 (test) | crack pixel IoU (test) |
|---|---|---|---|
| Baseline: pretrained YOLO26n-seg, no fine-tuning | 0.000 | 0.000 | – |
| Baseline: rule-based crack detector | – | – | … |
| **This model** | **…** | **…** | **…** |

![Per-class AP](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/per_class_ap.png)

*TODO: the strongest and the weakest class.*

## Intended use

- **Intended users and use:** facility managers, inspectors and researchers who **screen** photos of RC structures (buildings, parking garages, bridges) to find and rank the photos that show damage, before an engineer looks at them.
- **Out-of-scope uses:** structural safety or load-capacity decisions on its own; measuring crack widths or damage areas in m² (there is no scale); other materials (masonry, steel, timber, asphalt); aerial or thermal images; any decision about individual people.

## Model details

| | |
|---|---|
| **Task** | Instance segmentation |
| **Architecture** | YOLO26n-seg |
| **Parameters** | … (all trainable) |
| **Base model** | `yolo26n-seg.pt`, pretrained on COCO |
| **Input** | One RGB photo, resized by Ultralytics to … px (the training image size) |
| **Output** | Per damage region: mask (polygon), box, class, confidence |
| **Recommended confidence threshold** | … (see *Choosing a confidence threshold*) |
| **Files** | `model.pt` (weights + class names), `config.json` (image size, threshold, versions), `training_log.csv` (per-epoch metrics), figures |
| **Library versions** | ultralytics …, torch …, Python … |

### Classes

| ID | Class | Definition (our reading of the dataset's classes; colour in the source masks) |
|---|---|---|
| 0 | crack | Linear fracture in the concrete surface (red in the source masks) |
| 1 | spalling | Concrete broken off the surface, without visible reinforcement (green) |
| 2 | rebar_exposure | Reinforcing steel visible because the cover is lost (magenta) |
| 3 | rebar_corrosion | Visible rust on reinforcement or rust staining (blue) |
| 4 | crushing | Concrete crushed under compression, typically at column ends or joints (yellow) |

Each instance is a **connected region** of one class in the source masks: a crack with a gap is two instances, and a closed crack network is one polygon.

## Training data

| | |
|---|---|
| **Dataset** | Reinforced concrete structure segmentation dataset 2119 (Wang, 2026) |
| **Source** | [Mendeley Data](https://data.mendeley.com/datasets/2vkm6k4cfg/1), DOI 10.17632/2vkm6k4cfg.1 |
| **Licence** | CC BY 4.0 |
| **Size** | … images (train … / validation … / test …); instances: crack …, spalling …, rebar exposure …, rebar corrosion …, crushing … |
| **Coverage** | Photos of damaged RC structures collected by the dataset authors from public datasets; mostly close-ups of damage, various structures, cameras and lighting |
| **Preparation** | Colour masks converted to polygons; photos resized to at most 1,280 px; near-duplicates (perceptual hash) kept in the same split; split 70/15/15, stratified by class |

![Dataset overview](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/dataset_overview.png)

## Training procedure

- **Settings:** fine-tuned from `yolo26n-seg.pt`, up to 50 epochs with early stopping (patience 15), batch …, image size …, Ultralytics default optimiser and learning-rate schedule, seed 0.
- **Augmentation:** Ultralytics defaults (mosaic, horizontal flips, scaling, translation, HSV colour changes).
- **Experiment:** image size 640 vs. 960, compared on validation: TODO result.
- **Hardware and time:** Google Colab, NVIDIA T4 GPU, … minutes.
- **Model selection:** the epoch with the best validation fitness (`best.pt`); the test set was used once, at the end.

![Training curves](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/training_curves.png)

*TODO: still learning? overfitting? best epoch?*

## Evaluation

Test set: … photos never seen during training or model selection.

| Metric (test) | Pretrained YOLO26n-seg | Rule-based crack detector | This model |
|---|---|---|---|
| mask mAP50 | 0.000 | – | … |
| mask mAP50-95 | 0.000 | – | … |
| box mAP50 | 0.000 | – | … |
| mask precision / recall at the recommended threshold | – | – | … / … |
| crack pixel IoU | – | … | … |

**Per class (test):**

| Class | Instances | mask precision | mask recall | mask AP50 | mask AP50-95 |
|---|---|---|---|---|---|
| crack | … | … | … | … | … |
| spalling | … | … | … | … | … |
| rebar_exposure | … | … | … | … | … |
| rebar_corrosion | … | … | … | … | … |
| crushing | … | … | … | … | … |

![Confusion matrix](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/confusion_matrix.png)

**Where it fails:** TODO (your own reading of the failure gallery): e.g. confusion between rebar exposure and corrosion, missed hairline cracks, joints or stains taken for cracks, dark or wet surfaces.

![Failure cases](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/failures.png)

### Choosing a confidence threshold

A missed rebar or crushed zone costs more than a photo an engineer has to dismiss, so the threshold maximises the **F2 score** (recall counts twice) of the mean precision and recall curves on the **validation** set: **conf = …** (validation precision …, recall …).

![Threshold sweep](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/threshold_sweep.png)

### Application: an inspection list

![Inspection list](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter/resolve/main/application.png)

*New photos (not in the dataset), ranked by the most severe damage type found and then by the damaged share of the photo. TODO: one sentence.*

## Limitations and biases

- **Data:** photos collected from public datasets, mostly close-ups of damage; few photos of intact concrete, so false alarms on healthy surfaces are probably underestimated. Unknown regions and structure types.
- **Imbalance:** rebar exposure and crushing have fewer than 100 instances each; their scores are uncertain and probably lower than for cracks and corrosion.
- **Split:** near-duplicates are grouped, but different photos of the same structure may be in different splits, so the test score may be optimistic.
- **Labels:** instances are connected regions of semantic masks; the line between rebar exposure and corrosion is a judgement call of the annotator.
- **Image conditions:** sensitive to scale (hairline cracks), lighting, wet surfaces, paint, graffiti, formwork marks and joints.
- **No scale:** damaged areas are shares of the photo, not of the element.

## Ethics and privacy

The photos show structures, not people. Do not use the model to make safety decisions without an engineer: a photo without detected damage is **not** proof that a structure is safe.

## How to use

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("iaac-maai/day4-group4-rc-damage-segmenter", "model.pt"))
result = model.predict("wall.jpg", conf=0.25, imgsz=640)[0]     # TODO: recommended conf and imgsz (config.json)
print(len(result.boxes), "damage regions:", [model.names[int(c)] for c in result.boxes.cls])
result.save("wall_damage.jpg")                                    # photo with the masks drawn
```

⚠️ `.pt` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/Group4-day4-assignment. Run `notebook/object_detection.ipynb` with the packages in `requirements.txt` (on Colab with a T4 GPU for training; with `TRAIN = False` on a CPU).

## Authors and licence

Group 4. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under **AGPL-3.0** (fine-tuned from Ultralytics YOLO26, which is AGPL-3.0). The data is subject to its own licence: **CC BY 4.0**, Wang, J. (2026). *Reinforced concrete structure segmentation dataset 2119*. Mendeley Data, V1. DOI 10.17632/2vkm6k4cfg.1.
