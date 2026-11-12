# Reinforced-concrete damage segmentation with YOLO26-seg

**Team:** Group 4  <!-- TODO -->
**Course:** AI Methods I – Day 4: Neural networks and computer vision (IAAC MaAI 2026/27)
**Track:** B (detection / segmentation)
**Model on Hugging Face:** https://huggingface.co/iaac-maai/day4-group4-rc-damage-segmenter  <!-- TODO: replace group4 everywhere -->

![What the model does](figures/09_test_predictions.png)
*Test photos with the ground truth (middle) and the model's masks (right): two of the best, two typical and two of the worst predictions. TODO: one sentence on what the reader should notice, e.g. which class it confuses in the bottom rows.*

## Use case

A facility manager or structural inspector has hundreds of photos of a building, parking garage or bridge and must decide which elements an engineer should visit first. The model outlines five types of damage on every photo (crack, spalling, rebar exposure, rebar corrosion, crushing) and the notebook turns the result into a ranked inspection list. **A missed exposed or corroded rebar, or crushed concrete, is the expensive mistake** (a structural problem stays unseen); a false alarm costs an engineer a minute to dismiss a photo. We therefore choose the confidence threshold with the F2 score, which favours recall.

## Dataset

| | |
|---|---|
| **Name** | Reinforced concrete structure segmentation dataset 2119 (Wang, 2026) |
| **Source** | [Mendeley Data](https://data.mendeley.com/datasets/2vkm6k4cfg/1), DOI 10.17632/2vkm6k4cfg.1 |
| **Content** | 2,119 photos of damaged RC structures collected from public datasets, with pixel-level colour masks for 5 damage types |
| **Size after conversion** | TODO from `figures/results_summary.md`: … images (train … / val … / test …); instances per class: crack …, spalling …, rebar exposure …, rebar corrosion …, crushing … |
| **Licence** | CC BY 4.0 |

The colour masks are converted to YOLO polygons (one polygon per connected region of a class). Near-duplicate photos (same scene resized, mirrored or cropped; found with a perceptual hash) are kept in the same split, and the split is stratified so that every split gets its share of the rare classes.

![Dataset overview](figures/01_dataset_overview.png)
*One example per class and the number of instances per split: the dataset is strongly imbalanced, with fewer than 100 instances of rebar exposure and crushing.*

## Repository structure

```
├── data/            # downloaded and converted data (not committed)
├── figures/         # exported plots, experiment table, results summary
├── models/          # model card (README.md) + its figures + config.json; model.pt lives on Hugging Face
├── notebook/
│   └── object_detection.ipynb   # the full pipeline, from download to published model
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `notebook/object_detection.ipynb`, select the `.venv` kernel and run all cells. The notebook downloads the dataset itself into `data/`.

**Where the training ran:** Google Colab, free T4 GPU (tier "free cloud"). Run A (image size 640) took … min, run B (image size 960) … min. TODO. Without a GPU, set `TRAIN = False` and `RUN_EXPERIMENT = False` in the first code cell: the notebook then downloads the published weights from Hugging Face and runs everything else on a CPU.

## Approach

- **Model:** YOLO26n-seg (… parameters), pretrained on COCO, **fine-tuned** (not trained from scratch) on our five damage classes.
- **Training:** up to 50 epochs, early stopping (patience 15), batch 16, image size 640, Ultralytics default augmentation and optimiser, seed 0. Best epoch selected on validation.
- **Baselines:** (1) the pretrained YOLO26n-seg without fine-tuning: it has no damage classes, so mAP = 0; (2) a rule-based crack detector (black-hat transform + Otsu threshold, notebook 02), compared on crack pixels.
- **Experiment:** image size 640 vs. 960 (same model, same settings), compared on the validation set.
- **Threshold:** chosen on validation by maximising the F2 score (recall weighs twice as much as precision).

## Results

TODO: paste the tables from `figures/results_summary.md`.

| Model (test set) | mask mAP50 | mask mAP50-95 | crack pixel IoU |
|---|---|---|---|
| Pretrained YOLO26n-seg, no fine-tuning | 0.000 | 0.000 | – |
| Rule-based crack detector | – | – | … |
| **This model** | **…** | **…** | **…** |

| Experiment (validation) | mask mAP50 | mask mAP50-95 | training time |
|---|---|---|---|
| Image size 640 | … | … | … min |
| Image size 960 | … | … | … min |

![Training curves](figures/03_training_curves.png)
*TODO: is the model still learning at the end, does the validation loss rise (overfitting), at which epoch is the best validation mAP?*

![Per-class AP](figures/07_per_class_ap.png)
*TODO: strongest and weakest class, and whether the weak ones are the rare ones.*

![Confusion matrix](figures/08_confusion_matrix.png)
*TODO: which classes are confused with each other (e.g. rebar exposure vs. corrosion), and how much damage is missed (the "background" row).*

![Threshold sweep](figures/06_threshold_sweep.png)
*The recommended threshold (…) maximises F2 on validation: TODO precision … / recall ….*

![Failure cases](figures/09b_test_failures.png)
*TODO: what the worst test photos have in common.*

![Application](figures/10_application_inspection_list.png)
*New photos, not in the dataset, ranked by the most severe damage type found and the damaged share of the photo. TODO: does the ranking make sense?*

## Limitations

- The photos come from public datasets and are mostly close-ups of damage: few photos of intact concrete, so false alarms on healthy surfaces are probably underestimated.
- Rebar exposure and crushing have fewer than 100 instances each, so their scores are uncertain.
- Near-duplicates are grouped, but different photos of the same structure may still be in different splits: the test score may be optimistic.
- Instances are connected regions of the semantic masks, not instances drawn by an annotator; hairline cracks are thinner than one pixel at 640 px.
- A screening tool only: not for structural safety decisions, crack-width measurement or other materials.

## AI usage disclosure

| | |
|---|---|
| **Tools** | TODO, e.g. Claude (agent in claude.ai) |
| **Why** | TODO |
| **Where** | TODO, e.g. choice of dataset, notebook code (mask-to-polygon conversion, near-duplicate grouping, figures), README and model card templates |
| **How** | TODO: what you asked, how you checked the output (e.g. ran the notebook end to end, checked the colour-to-class mapping printed in section 1.3, inspected the figures), what you corrected |

**Lessons learned:** TODO

## License

Code released under the [MIT License](LICENSE). The model is released under AGPL-3.0, because it is fine-tuned from Ultralytics YOLO26 (see the model card). The data is subject to its own licence: CC BY 4.0, Wang (2026), DOI 10.17632/2vkm6k4cfg.1.
