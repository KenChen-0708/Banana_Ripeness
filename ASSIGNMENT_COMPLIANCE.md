# Assignment Compliance Review

This review maps the implemented fruit ripeness assessment system to the supplied BMDS2133 specification. It is an engineering checklist for the prototype and report; it does not replace the required group documentation, presentation slides, Group Contract Form, or AI disclosure.

## Selected mode and project scope

The system implements Mode A, Comparative and Enhancement Study, for Fruit Ripeness Assessment. Four classical pipelines are benchmarked on fixed train, validation, and held-out test sets. The application classifies unripe, ripe, overripe, and rotten bananas and quantifies blemished area relative to the detected fruit surface.

## Functional requirement evidence

| Specification requirement | Implementation evidence | Status |
|---|---|---|
| Noise removal and enhancement | Deterministic 5 x 5 median filtering and luminance-only CLAHE are shown in the assessment evidence panel. | Implemented |
| Image calibration and rectification | Object crops are rectified to a fixed canvas without aspect-ratio distortion. An optional known reference length derives pixels per centimetre, projected surface area, and contour perimeter. | Implemented |
| Object detection | Largest connected fruit region, external contour, and bounding box are computed and visualised. | Implemented |
| Surface quality and damage quantification | Brown and dark masks are combined; blemish pixels are divided by detected fruit pixels. Green/yellow and fruit-area ratios are also reported. | Implemented |
| Data analysis dashboard | Batch predictions, per-pipeline probabilities, masks, object measurements, timing, agreement, confusion matrices, class metrics, and leaderboard are available. | Implemented |
| Automated PDF reporting | Summary and per-pipeline PDF downloads contain prediction and surface-quality results. | Extra effort implemented |
| Video processing | MP4, AVI, MOV, and MKV uploads are sampled at a configurable rate and analysed frame-by-frame with live annotated output, temporal charts, safety bounds, and CSV/PDF exports. | Extra effort implemented |
| Bulk GUI ingestion | Multiple JPG, JPEG, and PNG files are validated and processed by every model. Duplicate filenames are preserved safely. | Extra effort implemented |
| Algorithm comparison | HSV segmentation plus SVM, L*a*b* K-means plus KNN, masked HSV histograms plus Random Forest, and masked GLCM/morphology plus Random Forest are compared quantitatively. | Implemented |
| Big-data evidence | The recorded full experiment contains 13,478 images across immutable train, validation, and test folders. | Implemented |

## Algorithm correctness controls

- GLCM co-occurrences include only fruit-to-fruit pixel pairs, preventing black padded background from dominating texture measurements.
- HSV histograms are calculated through the fruit mask rather than from the full scene.
- K-means uses a fixed OpenCV random seed, deterministic sampling, fixed criteria, and sorted cluster centres.
- SVM and KNN features are standardised inside fitted scikit-learn pipelines.
- Random Forest and SVM randomness is fixed by the shared random state.
- Feature caches store and verify the feature version, image size, and ordered paths before reuse.
- Model bundles verify schema, class order, pipeline key, feature version, and image size on load.
- Train, validation, and test source identifiers are checked for overlap before training.

## Evidence needed in the final report for the top rubric band

The prototype can support the top prototype band, but the documentation marks depend on the submitted report. The final report still needs all of the following:

1. An exceptional background and explicit SMART objectives, including numeric target metrics and a completion date.
2. A deep literature review with a comprehensive comparison table and complete matching APA in-text citations and references.
3. A detailed logical flowchart and step-by-step explanation covering acquisition, validation, preprocessing, segmentation, rectification, feature extraction, classification, evaluation, and export.
4. Robust experimental results for every pipeline, class-level precision/recall/F1, confusion matrices, timing, generalisation gaps, and critical interpretation of failure cases.
5. A conclusion that addresses every SMART objective individually.
6. Objective/passive UK English with no first-person pronouns.
7. Editable documentation plus PDF, source code plus presentation slides, the required forms, and one final ZIP submitted by the group leader.

## Presentation and live coding preparation

Every team member should be able to explain the colour spaces, threshold masks, morphology, GLCM equations/properties, K-means ordering, classifier choices, data leakage controls, metrics, confidence limitations, and cache/model compatibility checks. Rehearse a small live change such as adjusting a threshold, adding a metric, or changing a chart while preserving the tests.
