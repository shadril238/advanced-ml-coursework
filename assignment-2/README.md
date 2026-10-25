# Assignment 2: Comparative Evaluation of ML and Deep Learning Models for Chest X-ray Classification

This assignment evaluates six models for chest X-ray classification in two settings: (1) individual-dataset evaluation and (2) evaluation on a combined CheXpert–VinDr-CXR dataset. The individual experiments use CheXpert, NIH ChestX-ray14, or RSNA Pneumonia, as appropriate for each model. The combined experiment brings together CheXpert and VinDr-CXR images for four shared findings. Model experiments were run on Kaggle with T4 GPUs. The notebooks document data preparation, model configurations, and evaluation procedures. They also present the resulting metrics and figures.

## 1. Dataset-based evaluation

| Notebook | Year | Venue | Paper | Official dataset |
| --- | ---: | --- | --- | --- |
| [clear-chexpert.ipynb](./clear-chexpert.ipynb) | 2026 | Nature Biomedical Engineering | [Paper](https://www.nature.com/articles/s41551-026-01741-4) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| [clear-nih-chestxray14.ipynb](./clear-nih-chestxray14.ipynb) | 2026 | Nature Biomedical Engineering | [Paper](https://www.nature.com/articles/s41551-026-01741-4) | [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| [alpharad-chexpert.ipynb](./alpharad-chexpert.ipynb) | 2026 | ECCV | [Paper](https://arxiv.org/abs/2609.01757) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| [alpharad-rsna.ipynb](./alpharad-rsna.ipynb) | 2026 | ECCV | [Paper](https://arxiv.org/abs/2609.01757) | [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) |
| [eva-x-s-chexpert.ipynb](./eva-x-s-chexpert.ipynb) | 2025 | npj Digital Medicine | [Paper](https://www.nature.com/articles/s41746-025-02032-z) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| [eva-x-s-nih-chestxray14.ipynb](./eva-x-s-nih-chestxray14.ipynb) | 2025 | npj Digital Medicine | [Paper](https://www.nature.com/articles/s41746-025-02032-z) | [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| [chexworld-chexpert.ipynb](./chexworld-chexpert.ipynb) | 2025 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| [chexworld-nih-chestxray14.ipynb](./chexworld-nih-chestxray14.ipynb) | 2025 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html) | [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| [radzero-chexpert.ipynb](./radzero-chexpert.ipynb) | 2025 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/510e5c2fd5ad326f75594563f4ad5e0d-Abstract-Conference.html) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| [radzero-nih-chestxray14.ipynb](./radzero-nih-chestxray14.ipynb) | 2025 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/510e5c2fd5ad326f75594563f4ad5e0d-Abstract-Conference.html) | [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| [gloria-chexpert.ipynb](./gloria-chexpert.ipynb) | 2021 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| [gloria-rsna.ipynb](./gloria-rsna.ipynb) | 2021 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html) | [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) |

Official implementation repositories: [GLoRIA](https://github.com/marshuang80/gloria), [CLEAR](https://github.com/peterhan91/CLEAR), [AlphaRAD](https://github.com/jz5426/ECCV-2026-AlphaRAD), [EVA-X-S](https://github.com/hustvl/EVA-X), [CheXWorld](https://github.com/LeapLabTHU/CheXWorld), and [RadZero](https://github.com/deepnoid-ai/RadZero).

### Reproduce the individual-dataset experiments

1. Import a notebook into Kaggle and attach one of the three datasets listed above: CheXpert, NIH ChestX-ray14, or RSNA Pneumonia.
2. **Exception:** AlphaRAD-CheXpert and RadZero-CheXpert use the separate 668-image CheXpert expert test set.
3. Enable a GPU if the notebook needs one, and enable internet access to load model weights. Run all cells, save the Kaggle version, and download results from `/kaggle/working/`.

## 2. Combined-dataset-based evaluation

The combined experiment brings together [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) and [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) chest X-rays.

**Dataset preparation:** [chexpert-vindr-combined-image-dataset.ipynb](./chexpert-vindr-combined-image-dataset.ipynb) builds the shared train, validation, and test dataset used by the model notebooks below.

| Notebook | Year | Venue | Paper | Official datasets |
| --- | ---: | --- | --- | --- |
| [alpharad-combined-chexpert-vindr.ipynb](./alpharad-combined-chexpert-vindr.ipynb) | 2026 | ECCV | [Paper](https://arxiv.org/abs/2609.01757) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) + [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) |
| [radzero-combined-chexpert-vindr.ipynb](./radzero-combined-chexpert-vindr.ipynb) | 2025 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/510e5c2fd5ad326f75594563f4ad5e0d-Abstract-Conference.html) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) + [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) |
| [eva-x-s-combined-chexpert-vindr.ipynb](./eva-x-s-combined-chexpert-vindr.ipynb) | 2025 | npj Digital Medicine | [Paper](https://www.nature.com/articles/s41746-025-02032-z) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) + [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) |
| [gloria-combined-chexpert-vindr.ipynb](./gloria-combined-chexpert-vindr.ipynb) | 2021 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) + [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) |

**Harmonization and split:** The preparation notebook selects 10,000 frontal CheXpert images and 15,000 VinDr-CXR images. It aligns four shared findings: Atelectasis, Cardiomegaly, Consolidation, and Pleural Effusion. CheXpert is split by patient and VinDr-CXR by image, with approximately 80% training, 10% validation, and 10% test data from each source. The combined ZIP contains the images and labels for each split.

### Reproduce the combined-dataset experiments

1. Run [the preparation notebook](./chexpert-vindr-combined-image-dataset.ipynb) in Kaggle with the three inputs linked above, then save its version containing `chexpert_vindr_combined.zip`.
2. In each model notebook, attach that saved version through **Add Input → Notebooks**. Add any model weights requested by the notebook, and enable internet or a GPU if needed.
3. Run all cells, save each Kaggle version, and download metrics and figures from `/kaggle/working/`. Use the same ZIP for all four models and compare zero-shot and trained results separately.

A detailed [technical report](./Technical%20Report%20-%20Assignment%202_Shifat_Shadril_Hassan_22646203.docx) is included separately with the assignment submission. It covers the experimental setup, results, comparisons, figures, and discussion for both the individual-dataset and combined-dataset evaluations.
