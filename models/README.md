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
  - map
---

# RC damage segmentation: cracks, spalling, exposed and corroded rebar, crushing

This model outlines **five types of damage on photos of reinforced concrete**: crack, spalling, exposed rebar, corroded rebar and crushing. It is meant to **sort inspection photos**, so that an engineer looks at the worst ones first. It is a YOLO26n-seg model (3.1 M parameters, pretrained on COCO), fine-tuned on the *Reinforced concrete structure segmentation dataset 2119* (Wang, 2026).

**Team:** Group 4 – Luis, Fawad, Hetashri, Daria · AI Methods I, IAAC MaAI 2026/27 · **Code:** https://github.com/iaac-maai/Group4-day4-assignment

## Model at a glance

![Test images: truth vs prediction](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/09_test_predictions.png)

*Test images, annotation on the left, prediction on the right. Top: 4 random images, segmented well. Bottom: the 4 worst. Typical mistakes are a right outline with the wrong class and cracks cut into different pieces than the annotation.*

![Per-class AP on the test set](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/08_test_per_class.png)

*Good on the frequent classes (crack, corrosion: AP50 about 0.8), weak on the rare ones (about 0.2–0.3), which are also the most severe. Right: much better than simple rules (pixel IoU 0.77 vs 0.20 for cracks).*

| Test set (303 images) | mask mAP50 | mask mAP50-95 |
|---|---|---|
| Pretrained YOLO26n-seg, no fine-tuning | 0.00 | 0.00 |
| **This model** | **0.478** | **0.301** |

At the recommended threshold **0.30**: precision **0.75**, recall **0.69**.

## Intended use

- **Intended users and use:** facility managers, inspectors and researchers who **screen** photos of RC structures (buildings, parking garages, bridges) to find and rank the photos that show damage, before an engineer looks at them. Always with a human looking at every photo.
- **Out-of-scope uses:** structural safety or load-capacity decisions on its own, or declaring an element safe ("no damage found" is not "no damage"); measuring crack widths or damaged areas in m² (there is no scale); other materials (masonry, steel, timber, asphalt); aerial, thermal or night images; any decision about individual people.

## Model details

| | |
|---|---|
| **Architecture** | YOLO26n-seg (Ultralytics), 3.1 M parameters |
| **Base model** | `yolo26n-seg.pt`, pretrained on COCO |
| **Input** | RGB image, resized to 640 px (longest side) |
| **Output** | per damage: outline (mask), box, class, confidence |
| **Versions** | ultralytics 8.4.210, torch 2.15.0, Python 3.12 |

| ID | Class | Definition |
|---|---|---|
| 0 | `crack` | a crack line, hairline to wide |
| 1 | `spalling` | concrete broken off, no visible rebar |
| 2 | `rebar_exposure` | visible rebar, not (or barely) rusted |
| 3 | `rebar_corrosion` | visible rusted rebar |
| 4 | `crushing` | concrete crushed under compression |

## Training data

Wang, J. (2026), *Reinforced concrete structure segmentation dataset 2119*, Mendeley Data, V1, [doi:10.17632/2vkm6k4cfg.1](https://doi.org/10.17632/2vkm6k4cfg.1), CC BY 4.0. 2,119 photos of damaged RC structures collected by the authors from public datasets, with 3,124 polygons (crack 1,833, spalling 282, rebar exposure 89, rebar corrosion 832, crushing 88). Every photo shows damage; mostly close-ups, various structures, cameras and lighting.

**Split by duplicate group:** many photos are flipped, rotated or cropped copies of each other (77 % of the images are in a group). We grouped copies (perceptual hash + ResNet18 features) and kept every group in one split: train 1,513 / validation 303 / test 303 images.

## Training procedure

50 epochs (patience 15), batch 16, image size 640, Ultralytics default augmentation, seed 42, on an Apple M4 laptop GPU (90 min). Best epoch: 47. An experiment with image size 416 was 0.024 mask mAP50 worse on validation, so we kept 640.

![Training curves](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/04_training_curves.png)

*Validation mAP rises fast for 20 epochs, then slowly to 0.56. The validation loss stays flat while the training loss keeps falling: beginning overfitting.*

## Evaluation

| Class (test polygons) | mask AP50 | mask AP50-95 |
|---|---|---|
| crack (283) | **0.83** | 0.50 |
| spalling (34) | 0.28 | 0.16 |
| rebar_exposure (12) | 0.28 | 0.20 |
| rebar_corrosion (98) | **0.78** | 0.52 |
| crushing (11) | 0.22 | 0.12 |

![Confusion matrix](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/05_confusion_matrix_val.png)

*Validation set; columns = true class, rows = prediction. Last row = missed damage (29 % of cracks, 40 % of spalling). Last column = false alarms (outlines where no damage is annotated): 57 % of them are cracks, because joints, shadows and edges look like cracks. Exposed rebar is never recognised: 58 % is called corrosion, since the classes differ only by rust and corrosion has ten times more training examples.*

![Failures](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/06_failures_val.png)

*The worst validation images: cracks cut into different pieces, exposed bars called corrosion, several bars drawn as one spalling area.*

![Threshold sweep](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/07_threshold_sweep.png)

*Threshold = the lowest one with at least 70 % precision on validation: **0.30**. Exposed rebar has recall 0 at every threshold.*

![Application: bridge photos ranked](https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmentation/resolve/main/10_application_bridge.png)

*44 new photos of a damaged bridge (Achim Hering, Wikimedia Commons, CC BY 3.0), ranked by the most severe damage found. Ranks 2 and 4 are right, rank 1 is a false alarm (shadowed pier edge), and one photo with clearly corroded bars was missed.*

## Limitations and biases

- **Rare classes are weak** (AP50 0.22–0.28), and they are the severe ones: rebar exposure and crushing have fewer than 100 polygons each.
- **No intact concrete in the data:** every photo shows damage, so false alarms on healthy surfaces are probably underestimated.
- **Optimistic for new buildings:** many test crack close-ups still resemble training photos.
- **Unknown sources:** the photos come from unnamed public datasets; two of our "new" bridge photos were in it.
- **Labels:** inconsistent granularity (one crack vs several pieces); the line between rebar exposure and corrosion is a judgement call of the annotator.
- **Image conditions:** shadows, joints, edges and rust stains from steel parts caused false alarms on the bridge photos; wet surfaces, paint or graffiti are probably also a problem.
- **No scale:** damaged areas are shares of the photo, not of the element.
- Rare-class scores rest on 11–34 test polygons; one training run per setting.

## Ethics and privacy

The photos show structures, not people. Do not use the model to make safety decisions without an engineer: a photo without detected damage is **not** proof that a structure is safe.

## How to use

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("iaac-maai/day4-group4-rc-damage-segmentation", "model.pt"))
result = model.predict("column.jpg", conf=0.30, imgsz=640)[0]
for cls, conf in zip(result.boxes.cls, result.boxes.conf):
    print(model.names[int(cls)], round(float(conf), 2))
result.save("column_damage.jpg")   # image with the outlines drawn
```

`.pt` files can run code when loaded: only load models you trust.

## Reproducibility

Run `notebook/object_detection.ipynb` from the GitHub repository with `requirements.txt`. With `TRAIN = False` it downloads this model instead of training.

## Authors and licence

Luis, Fawad, Hetashri, Daria (Group 4), IAAC MaAI 2026/27. Model: **AGPL-3.0** (fine-tuned from Ultralytics YOLO). Data: Wang, J. (2026), Mendeley Data, V1, CC BY 4.0; please cite Wang & Ueda (2025), *Structural Concrete* 26(5), 5511–5534. Bridge photos: Achim Hering, CC BY 3.0.
