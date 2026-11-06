# Spatiotemporal Vehicle Light Detection & Tracking

## Overview
This project is a custom computer vision pipeline built from scratch in Python (NumPy/OpenCV) designed to detect, extract, and track vehicle taillights, brake lights, and turn signals from 30Hz driving videos. 

Instead of relying on black-box machine learning models, this pipeline mathematically processes raw video feeds into **Spatiotemporal Motion Profiles**, isolating the temporal lifespan and trajectory of active vehicles. It utilizes custom RGB linear transformations, multi-scale image pyramids, Laplacian edge-detection convolutions, and Sobel gradient-directed line tracking.

## Pipeline Walkthrough

### Phase 1: Spatiotemporal Image Generation
The system reads a driving video frame-by-frame, immediately cropping to the horizon region of interest to eliminate sky and vehicle hood glare.
1. **Color Isolation:** Linear transformations `(R - 0.5G - 0.5B)` and `(0.5R + 0.5G - B)` are applied with zero-clipping to mathematically isolate intense red and yellow lights while erasing neutral white headlights and reflections.
2. **Temporal Compression:** Vertical max-pooling compresses the isolated frame data into a single 1D scan line.
3. **Matrix Stacking:** The 1D lines are stacked continuously over time. The resulting 2D matrix (`RT` for Red, `YT` for Yellow) represents physical space on the X-axis and time flowing upward on the Y-axis. Moving vehicles appear as continuous solid or dashed lines.

![alt text](<passing (13)_YT.png>) ![alt text](<passing (13)_RT.png>)
> *Caption: Moving red/yellow vehicles traced over time. A solid continuous line represents steady taillights, while vertical dashed lines indicate blinking turn signals.*

### Phase 2: Sizing for Depth & Laplacian Filtering
Under perspective projection, close vehicles appear large and distant vehicles appear small.
1. **Multi-Scale Processing:** The temporal matrix is geometrically scaled horizontally (e.g., 259px and 518px widths) to compress close-up and mid-range vehicle lights into uniform 2-to-3-pixel targets without distorting the temporal Y-axis.
2. **Edge Convolution:** A custom 1D horizontal Laplacian kernel `[-2, -1, 6, -1, -2]` slides across every temporal row. 
3. **Thresholding:** Applied at `DELTA = 125`, the filter mathematically highlights localized bright peaks while actively erasing broad, weak-yellow structural noise (e.g., painted lane lines, autumn foliage, yellow vehicle bodies).

![alt text](<passing_changeline (1)_YLT2.png>) ![alt text](<passing_changeline (1)_RLT2.png>)
> *Caption: The Laplacian filtered output (`RLT`). Structural noise is completely suppressed, leaving only crisp, mathematically isolated 2-to-3-pixel traces of actual signal lights.*

### Phase 3: Sobel Gradient Direction & Temporal Tracking
To confirm a light trace belongs to a physical, continuously moving vehicle and not a temporary camera glitch, a temporal length tracker validates object lifespans.
1. **Peak Detection:** The algorithm scans row-by-row (frame-by-frame) locating the isolated single-pixel core of a light.
2. **Directional Prediction:** $3 \times 3$ horizontal and temporal Sobel Operators (`Dx`, `Dt`) compute the trajectory gradient `(-Dt/Dx)`. The calculated tangent predicts the vehicle's exact velocity and expected physical position in the previous frame, ensuring crossing vehicles do not hijack each other's tracks.
3. **Template Matching:** A 7-pixel localized template of the peak is checked against the predicted location in the prior frame utilizing Sum of Squared Differences (SSD).
4. **Length Scoring:** If a match is found, the physical lifespan of the trace is incremented. Background noise is successfully bypassed by purging any track that fails to survive for fewer than 5 consecutive frames.

![alt text](<passing (13)_YLEN2.png>) ![alt text](<passing (13)_RLEN2.png>)
> *Caption: The final length tracking matrix (`RLEN`). Brightness correlates to temporal track length, proving continuous physical existence and filtering out single-frame artifacts.*

## Technologies Used
* **Python 3.x**
* **NumPy:** Matrix math, horizontal max-pooling, Sobel edge/gradient calculations.
* **OpenCV (cv2):** Video ingestion, geometric resizing (`INTER_AREA`), 2D convolutions (`filter2D`), and local I/O handling.
