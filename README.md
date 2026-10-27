# MiLa: multimodal landmark model for liver-related progression in MASLD

Code, trained model checkpoints and a synthetic example dataset accompanying the manuscript
*Multimodal Deep Learning Integrating Digital Pathology, Clinical Features, and Routine Laboratory Results for
Landmark-Based Prediction of Liver-Related Progression in MASLD* (World Journal of Hepatology, manuscript No. 126362).

MiLa (Multimodal integrated Landmark analysis) combines a slide-level GigaPath embedding (768-d, mean of the H&E and
Sirius Red embeddings), 30 baseline clinical and histological features, and routine laboratory results from the first
two years after biopsy (annual grid, last observation carried forward, with observation masks). Predictions are the
average of five fold-specific models. M3 (WSI only) and M4 (WSI + baseline features) are the comparison models.

## What is and is not in this repository

| Included | Not included |
|---|---|
| Model definitions, training and validation code (`src/`, `run_validation.py`) | Patient-level data (privacy obligations and IRB data-sharing terms; available to qualified investigators on request, see the manuscript's Data sharing statement) |
| Trained 5-fold checkpoints for M3, M4 and MiLa (`weights/`, plain PyTorch `state_dict` files) | Whole-slide images and the slide embeddings derived from them |
| Revision analyses (`analysis/`) and the retraining code used for the sensitivity analyses (`retrain/`) | Reinhard normalization, tiling and GigaPath feature-extraction code (see below) |
| A synthetic example dataset and generator (`examples/`) | |

**Slide preprocessing.** Slides were Reinhard-normalized, tiled and embedded with the public Prov-GigaPath tile encoder
(Xu et al., *Nature* 2024); the tile embeddings were mean-pooled to one 768-d vector per stain. The scripts used for
these steps are not part of this release. The models in this repository start from the pooled 768-d embeddings
(one `*-HE_embed.pt` and one `*-SR_embed.pt` tensor of shape `[1, 768]` per patient).

## Installation

```bash
conda create -n mila python=3.11 -y && conda activate mila
pip install -r requirements.txt
```

The code was run on CPU; `--device mps` or `--device cuda` also work for inference.

## Quick start with the synthetic data

```bash
python examples/make_synthetic_data.py --out examples/synthetic --n 40
python run_validation.py \
    --data-dir examples/synthetic/dataset \
    --embeddings-dir examples/synthetic/embeddings \
    --out-dir examples/synthetic/output
```

The synthetic cohort is random: identifiers (`SYN_0001`, ...), clinical values, laboratory series, embeddings and
outcomes are all generated, and the outcomes are unrelated to the inputs. The run therefore only demonstrates the file
formats and that the pipeline executes; the AUROCs it prints (around 0.5) have no meaning.

## Data layout expected by `run_validation.py`

```
<data-dir>/val/clinical_data_clean.csv       patient_id + 30 baseline features (see src/dataset.py)
<data-dir>/val/patient_labels.csv            patient_id, baseline_diagnosis, liver_progression_occurred, event_type,
                                             time_to_event_years, censored, follow_up_years
<data-dir>/follow_up_labs_clean/<id>_follow_up_labs.csv     days_from_biopsy + 11 laboratory columns
<embeddings-dir>/val_HE/<id with - for _>-HE_embed.pt        tensor [1, 768]
<embeddings-dir>/val_SR/<id with - for _>-SR_embed.pt        tensor [1, 768]
```

## Reproducing the reported external validation

With access to the study data placed in the same layout:

```bash
python run_validation.py --data-dir <data> --embeddings-dir <embeddings>
```

This loads the five fold checkpoints of each model from `weights/`, averages the softmax outputs, and writes
per-patient predictions and AUROC, AUPRC and Brier score with bootstrap confidence intervals (seed 42, 2000
resamples) to `outputs/validation_reproduced/`. On the study cohort (n = 94) the shipped checkpoints give MiLa
AUROC 0.8446 (3-year) and 0.9020 (8-year); M4 0.7440 / 0.8132; M3 0.6458 / 0.7147.

## Training

`src/train.py` trains one model type with 5-fold cross-validation (batch 16, 60 epochs, early stopping with patience
10, unweighted cross-entropy; no resampling or class weighting). Seeds are set explicitly. `retrain/` contains the
multi-seed and ablation runner used for the sensitivity analyses (`experiments.json` lists the experiments).

## Notes on the other folders

`analysis/` contains the analysis scripts used for the supplementary tables and figures. They read the
original study data layout and prediction files, which are not distributed, so they are provided for transparency
and will not run without access to those files.

## Checkpoints

`weights/` holds plain `state_dict` files (load with `torch.load(path, weights_only=True)`); they contain no patient
data. Architectures and hyperparameters are defined in `run_validation.py::build_model`.

## Citation

Please cite the manuscript above. A Zenodo DOI for the archived release will be added here.

## License

MIT (see `LICENSE`). The trained weights are released under the same license. The software and weights are research
tools, not validated medical devices, and must not be used for clinical decisions.
