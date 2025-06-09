# ProtoDC-Net
This paper will be disclosed when the submission is accepted (paper에 사용된 그림들 적극 활용)

## Dataset

1. USB-MD

<div align="center">
    <img src="images/USB_MD.jpg" alt="USB_MD" width="600"/>
</div>

2. USB-SD

<div align="center">
    <img src="images/USB_SD.jpg" alt="USB_SD" width="600"/>
</div>

3. DAMG2007

<div align="center">
    <img src="images/DAGM2007.jpg" alt="DAGM2007" width="600"/>
</div>


USB-MD 다운로드: https://drive.google.com/drive/folders/1NvQ5vZvZMdpJN8s1ttp9ZaMZ13OgQbAa?usp=sharing

USB-SD 다운로드: https://github.com/Xavierman/A-deep-learning-based-surface-defect-inspection-system-using-multi-scale-and-channel-compressed-feat?tab=readme-ov-file

DAGM2007 다운로드: https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html


## Classification Accuracy


### 📊 Classification Accuracies on USB-MD Dataset

| Method            | BrightLine | Deformation | Dent   | Scratch | Normal | Total  |
|-------------------|------------|-------------|--------|---------|--------|--------|
| ETE               | 68.47      | 52.08       | 77.78  | 74.49   | 73.94  | 71.16  |
| DECAF+MLR         | 50.45      | 52.60       | 92.07  | 78.59   | 80.03  | 73.38  |
| SN_MRF_CC         | 75.23      | 69.27       | 96.83  | **80.44** | 78.51  | 79.02  |
| FOMI              | 72.52      | 66.67       | **97.62** | 78.40   | 81.73  | 78.90  |
| MHAF              | 74.33      | **85.94**   | **97.62** | 77.47   | 85.96  | 82.55  |
| ProtoDC-Net (ours)| **85.14** | 84.90       | 96.83  | 79.14   | **90.69** | **86.03** |

> **Note**: Bold values indicate the highest score per class.  
> All results are averaged over three runs with different random seeds (42, 43, and 44).

### 📊 Classification Accuracies on USB-SD Dataset

| Method        | BrightLine | Deformation | Dent   | Scratch | Spot   | Squalidity | Normal | Total  |
|---------------|------------|-------------|--------|---------|--------|-------------|--------|--------|
| ETE           | 88.00      | 78.67       | 77.78  | 80.68   | 70.00  | 79.89       | 92.44  | 81.07  |
| DECAF+MLR     | 91.89      | 88.00       | 87.33  | 79.89   | 66.22  | 83.11       | 96.33  | 84.68  |
| SN_MRF_CC     | 94.56      | 91.67       | 96.34  | 93.78   | 91.33  | 94.66       | 97.44  | 94.25  |
| SN_MRF_CC *   | 97.89      | 97.67       | 98.67  | 96.67   | 92.00  | 95.22       | 98.44  | 96.65  |
| **ProtoDC-Net** | **100.00** | **100.00**   | **100.00** | **99.72** | **99.72** | **95.28**     | **99.65** | **99.20** |

> **Note**: * indicates that input images were augmented using rotation and cropping.  
> All results are averaged over three trials.


### 📊 Classification Accuracies on DAGM2007 Dataset

| Method        | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Total  |
|---------------|---------|---------|---------|---------|---------|---------|--------|
| DECAF+MLR     | 36.67   | 83.33   | 5.67    | 80.00   | 6.67    | 70.00   | 47.06  |
| SN_MRF_CC     | 100.00  | 100.00  | 93.33   | 100.00  | 30.00   | 100.00  | 87.22  |
| **ProtoDC-Net** | **100.00** | **100.00** | **100.00** | **100.00** | **93.33** | **100.00** | **98.89** |

> **Note**: Classification accuracies (%) on six classes (Class 1–6) of the **DAGM2007** dataset.  
> Each value indicates the classification accuracy on the defect class within each binary classification task.  
> All results are averaged over three trials.


## 🔍 Anomaly Detection using ProtoDC-Net (not included in the paper)

---

### 📊 1. Dataset Configuration (MVTec-AD)

- **Train**: 1 class → `normal`
- **Validation**: 1 class → `normal`
- **Test**: 2 classes → `normal`, `abnormal`

---


### ⚙️ 2. Classification by Thresholding

We classify samples based on the similarity between their global embeddings and the learned prototype.

- **Similarity**  
  similarity = cos(e_G^t, prototype)  
  where e_G^t is the global vector of a test sample.

- **Threshold**  
  threshold = min{ cos(e_G^v, prototype) | e_G^v ∈ D_val^normal }  
  where e_G^v is the global vector of a validation sample.

- **Prediction Rule**
  if similarity < (threshold × margin):
      prediction = abnormal
  else:
      prediction = normal

---

### 📌 Notation

- e_G^t: Global vector of a **test** sample  
- e_G^v: Global vector of a **validation** sample  
- prototype: Mean embedding of normal training samples


---


### 📈 3. Evaluation Metrics

We evaluate the method using:
- Accuracy (Normal/Abnormal)
- F1-score (binary classification)

The following table shows results for 5 selected classes:

| **Class**   | Bottle | Pill   | Toothbrush | Transistor | Wood   |
|------------|--------|--------|------------|------------|--------|
| **Margin** | 1.000  | 1.0005 | 1.001      | 1.0005     | 1.0005 |
| **Threshold** | 0.9927 | 0.9932 | 0.9886     | 0.9942     | 0.9917 |
| **Accuracy (normal)**   | 95.00  | 100.00 | 85.00      | 85.00      | 85.00  |
| **Accuracy (abnormal)** | 73.02  | 79.43  | 76.67      | 52.50      | 85.00  |
| **F1 Score**            | 83.64  | 84.21  | 77.97      | 53.16      | 87.18  |

> *Note: Only 5 out of 15 classes are shown above.*


