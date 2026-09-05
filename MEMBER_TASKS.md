# Group Member Task Allocations & System Architecture

**Course**: BMDS2133 Image Processing  
**Project**: Automated Banana Ripeness and Surface Quality Assessment System  
**Mode**: Mode A (Comparative & Enhancement Study)  

---

## 👥 Task Separation Matrix

Each member developed an independent, contemporary image-processing pipeline that performs feature extraction and ripeness grading. All four algorithms are unified under the common `BaseBananaPipeline` interface and evaluated side-by-side in the Streamlit prototype.

| Member | Assigned Algorithm / Task | Module Directory | Report Section | Key Techniques |
| :--- | :--- | :--- | :---: | :--- |
| **(a) Ronaldo Loh Yu Fei** | **HSV Colour Segmentation + Brown Spot Ratio + SVM** | [`src/pipelines/member_a_hsv_svm/`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/pipelines/member_a_hsv_svm/) | **Section 2.4** (pp. 22–33) | • RGB to HSV conversion<br>• Green, yellow, brown, dark surface ratios<br>• Brown spot defect ratio (%)<br>• Support Vector Machine (RBF kernel) |
| **(b) Lim Tze Chuen** | **K-Means Clustering with L\*a\*b\* Colour Space + KNN** | [`src/pipelines/member_b_lab_knn/`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/pipelines/member_b_lab_knn/) | **Section 2.2** (pp. 14–17) | • CIE L\*a\*b\* colour space transformation<br>• K-Means++ clustering ($k=4$ modes)<br>• Cluster centre chromaticity + weights<br>• Distance-weighted K-Nearest Neighbours ($k=7$) |
| **(c) Chen Kai Jian** | **HSV Colour Histogram + Random Forest Classifier** | [`src/pipelines/member_c_hsv_rf/`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/pipelines/member_c_hsv_rf/) | **Section 2.1** (pp. 7–13) | • 16-bin marginal histograms on H, S, V<br>• 48-dimensional normalised global colour vector<br>• 200-tree Random Forest Ensemble<br>• Gini Impurity feature subspace selection |
| **(d) Lee Siu Yee** | **GLCM Texture Feature Extraction + Morphology + Random Forest** | [`src/pipelines/member_d_glcm_rf/`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/pipelines/member_d_glcm_rf/) | **Section 2.3** (pp. 19–21) | • Morphological closing/opening mask refinement<br>• Gray-Level Co-occurrence Matrix (GLCM)<br>• Contrast, correlation, energy, homogeneity<br>• Geometric contour area, perimeter, aspect ratio |

---

## 🏗️ Shared Core Modules

In addition to individual algorithms, the shared system architecture provides the foundational image processing and evaluation framework:

- **Image Preprocessing & Calibration**: [`src/preprocessing.py`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/preprocessing.py)  
  *Gaussian/Median/Bilateral noise reduction, CLAHE contrast enhancement, spatial calibration ($\text{px}/\text{cm}$), and commercial quality grading.*
- **Video & Real-Time Stream Engine**: [`src/video.py`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/video.py)  
  *Conveyor belt video frame ingestion, temporal ripeness trend line chart, and live camera snapshot handling.*
- **Streamlit Prototype UI**: [`app.py`](file:///c:/Users/User/Downloads/Banana_Ripeness/app.py)  
  *3-page streamlined dashboard presenting all four algorithms, camera snapshots, and Model Evaluation leaderboard.*
- **Report & Audit Exports**: [`src/exports.py`](file:///c:/Users/User/Downloads/Banana_Ripeness/src/exports.py)  
  *Automated multi-page PDF audit reports and CSV results export.*
