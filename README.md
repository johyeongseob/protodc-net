# Unknown
This paper will be disclosed when the submission is accepted (paper에 사용된 그림들 적극 활용)

Dataset

1. USB-MD

<div align="center">
    <img src="images/USB_MD.jpg" alt="USB_MD" width="600"/>
</div>

다운로드: https://drive.google.com/drive/folders/1NvQ5vZvZMdpJN8s1ttp9ZaMZ13OgQbAa?usp=sharing

2. USB-SD

<div align="center">
    <img src="images/USB_SD.jpg" alt="USB_SD" width="600"/>
</div>

다운로드: https://github.com/Xavierman/A-deep-learning-based-surface-defect-inspection-system-using-multi-scale-and-channel-compressed-feat?tab=readme-ov-file

6. DAMG2007

<div align="center">
    <img src="images/DAGM2007.jpg" alt="DAGM2007" width="600"/>
</div>

다운로드: https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html


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
