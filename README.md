# Chest X-Ray Classification — Research Prototype

A TensorFlow/Keras notebook that trains an EfficientNetV2S classifier on public chest X-ray datasets and predicts one of four labels: **normal**, **other**, **pneumonia**, or **tuberculosis**. The workflow covers data collection, exact-file deduplication, image preprocessing, transfer learning, validation metrics, model export, and an optional single-image demo.

> **Medical disclaimer:** This repository is an educational research prototype only. It is not a medical device and has not been clinically validated. Do not use it for diagnosis, triage, treatment, or any healthcare decision.

## Repository contents

- `chest_xray_classifier.ipynb` — cleaned Google Colab notebook. Stale execution outputs were removed; previously recorded validation metrics are documented in the notebook and identified as historical.
- `requirements.txt` — core training and notebook dependencies.
- `requirements-lung-segmentation.txt` — optional dependencies for the lung-cropping experiment.

No datasets or trained weights are included.

## Run the notebook

The notebook targets **Google Colab with a GPU** and Google Drive. A full run downloads several external datasets and can take substantial time, disk space, and compute. Upstream datasets are not version-pinned, so results can change over time.

1. Open `chest_xray_classifier.ipynb` in Colab.
2. Run the notebook's first setup cell to install the core packages (it works even when Colab opened only the notebook file). For a local Jupyter environment, install with `pip install -r requirements.txt` from the repository folder.
3. Run the Drive-mount cell and approve access so model artifacts can be saved under `MyDrive/chest_xray_project`.
4. Run the dataset download and training cells. The notebook uses public KaggleHub dataset identifiers listed below; Kaggle access or account setup may be required by Kaggle's current policies.
5. The LIME explanation and single-image prediction cells are optional and interactive. The final demo asks you to upload an image.

To enable the optional lung segmentation step, install `requirements-lung-segmentation.txt`, then set `USE_SEG = True` in the configuration cell. The default is `False`.

## Data sources

The notebook references these public Kaggle datasets. Review each source's terms, citation requirements, and redistribution rules before use; dataset licenses are independent of this repository's code.

### Tuberculosis and normal

- [TB Chest X-ray Database — tawsifurrahman](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- [Shenzhen Chest X-rays — raddar](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen)
- [Montgomery Chest X-rays — raddar](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery)
- [Pulmonary Chest X-ray Abnormalities — kmader](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities)
- [TBX11K simplified — vbookshelf](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified)
- [DA and DB TB Chest X-ray datasets — vbookshelf](https://www.kaggle.com/datasets/vbookshelf/da-and-db-tb-chest-x-ray-datasets)
- [Drug-resistant Tuberculosis X-rays — raddar](https://www.kaggle.com/datasets/raddar/drug-resistant-tuberculosis-xrays)
- [Chest X-rays Tuberculosis from India — raddar](https://www.kaggle.com/datasets/raddar/chest-xrays-tuberculosis-from-india)

### Pneumonia and other

- [Chest X-ray Pneumonia Balanced Dataset — yusufmurtaza01](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset)
- [Chest X-ray dataset — alifrahman](https://www.kaggle.com/datasets/alifrahman/chestxraydataset)
- [X-ray body images — ibombonato](https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion)
- [Intel Image Classification — puneet6060](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

The `other` label is a broad catch-all assembled from non-target X-ray sources and generic imagery; it is not a coherent medical diagnosis category.

## Previously recorded validation results

The cleaned notebook preserves the latest saved run's results for a **2,144-image validation split**:

| Class | Precision | Recall / sensitivity | F1 | Specificity | Support |
|---|---:|---:|---:|---:|---:|
| Normal | 0.9620 | 0.9453 | 0.9536 | 0.9799 | 750 |
| Other | 0.9927 | 0.9838 | 0.9882 | 0.9975 | 555 |
| Pneumonia | 0.9564 | 0.9669 | 0.9616 | 0.9814 | 635 |
| Tuberculosis | 0.9256 | 0.9755 | 0.9499 | 0.9918 | 204 |

- **Accuracy:** 0.9646
- **Macro one-vs-rest AUC:** 0.9979

These values are historical outputs from the original notebook, not a fresh reproduction or an independent test result. They should be treated as preliminary.

## Limitations and interpretation

- The split is seeded but **image-level**, not patient-level or source-level. Exact duplicate files are removed before splitting, but near-duplicates, patient overlap, and dataset/source leakage may remain.
- Several heterogeneous datasets are combined, and dataset versions are not pinned. Acquisition and source artifacts may influence predictions.
- The validation split is not an external-site test set. There is no clinical validation, calibration study, prospective study, or comparison to a clinical standard.
- The image-color screening function is a simple heuristic, not a reliable chest-X-ray detector. The model's softmax scores are not calibrated probabilities.
- Horizontal flipping and the broad `other` label may not be appropriate for all clinical contexts.
- The notebook's historical metrics have not been independently reproduced during cleanup.

Do not upload identifiable patient images or other sensitive health information to public services or repositories.

## Reproducibility notes

- Random seeds are set to `42`; GPU kernels and upstream data changes may still introduce nondeterminism.
- The notebook saves model artifacts to Google Drive and does not commit them to Git.
- Before public release, choose and add a code license if you intend to grant reuse rights. Dataset licenses and citation requirements must be checked separately.
