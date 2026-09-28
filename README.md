<a id="top"></a>

<div align="center">

🩺 NeuroOcular AI

Physics-Informed Retinal Hemodynamics Platform

<p align="center">
  <b>Retinal Computer Vision × Hemodynamics × Physics-Informed AI × Edge Inference</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" />
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B" />
  <img src="https://img.shields.io/badge/Vision-OpenCV-5C3EE8" />
  <img src="https://img.shields.io/badge/Inference-ONNX%20Runtime-005CED" />
  <img src="https://img.shields.io/badge/Model%20Size-10.28%20KB-success" />
  <img src="https://img.shields.io/badge/Target%20Latency-%3C5ms-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

</div>

NeuroOcular AI is a research-oriented medical computer vision platform that analyzes retinal microvascular video to extract flow-related biomarkers, estimate physiological variables, and detect localized pre-occlusive flow abnormalities using a combination of dense optical flow, hemodynamic modeling, spatio-temporal anomaly detection, and lightweight ONNX inference.

<br>

<h2 align="center">⚠️ Medical & Research Disclaimer</h2>
<p align="center"><b>Research Use Only • Not a Clinically Validated Medical Device</b></p>

NeuroOcular AI is an experimental research prototype intended for academic research, engineering validation, and technical demonstration.

It must not be used as a substitute for:

Laboratory blood glucose measurement

Certified continuous glucose monitoring

Vascular imaging

Physician diagnosis

Emergency medical assessment

Anticoagulation decisions

Any other treatment or clinical decision

Clinical Translation Requirement:
A deployable clinical system would require prospective human studies, synchronized ground-truth measurements, independent external validation, safety testing, demographic robustness evaluation, and appropriate regulatory approval.

<br>

<h2 align="center">📑 Table of Contents</h2>
<p align="center"><b>Follow the project from concept → algorithms → AI → deployment → validation.</b></p>

🧭 Reading Map

Chapter

Focus

Sections

I. System Foundation

What the project is, why it exists, and how it works

1–6

II. Vision, Flow & Physics

Algorithms that extract and interpret retinal blood-flow dynamics

7–12

III. AI & Inference

NeuroOcularNet, features, training, ONNX, and application layer

13–17

IV. Engineering & Evaluation

Stack, repository, scenarios, results, and metrics

18–22

V. Run & Use the Platform

Installation, execution, deployment, inputs, and outputs

23–27

VI. Validation & Next Steps

Limitations, clinical roadmap, future work, security, and project metadata

28–35

<br>

<details open>
<summary><b>📌 Chapter I — System Foundation</b></summary>

1. Project Summary

2. Clinical & Engineering Motivation

3. Research Hypothesis

4. Project Objectives

5. What the System Does

6. End-to-End Architecture

</details>

<details>
<summary><b>👁️ Chapter II — Vision, Flow & Physics</b></summary>

7. Complete Algorithm Inventory

8. Computer Vision Pipeline

9. Retinal Motion & Velocimetry

10. Hemodynamic Feature Engineering

11. Biophysical & Rheological Modeling

12. Spatio-Temporal Anomaly Detection

</details>

<details>
<summary><b>🤖 Chapter III — AI & Inference</b></summary>

13. AI Model: NeuroOcularNet

14. Model Inputs & Outputs

15. Training & Normalization Pipeline

16. ONNX Edge Deployment

17. Application Layer

</details>

<details>
<summary><b>🧰 Chapter IV — Engineering & Evaluation</b></summary>

18. Technology Stack

19. Repository Structure

20. Sample Clinical Scenarios

21. Experimental Results

22. Performance Metrics

</details>

<details>
<summary><b>▶️ Chapter V — Run & Use the Platform</b></summary>

23. Installation

24. Running the Application

25. Cloud Deployment

26. Input Requirements

27. Output Interpretation

</details>

<details>
<summary><b>🔬 Chapter VI — Validation & Next Steps</b></summary>

28. Failure Modes & Limitations

29. Clinical Validation Roadmap

30. Future Development

31. Security & Privacy

32. Intellectual Property

33. License

34. Citation

35. Author

</details>

<div align="center">

CHAPTER I

📌 System Foundation

Problem → Hypothesis → Objectives → System Architecture

</div>

<br>

<a id="1-project-summary"></a>

<h2 align="center">📌 1. Project Summary</h2>
<p align="center"><b>Quick orientation: what the platform does and its four core functions.</b></p>

Overview: NeuroOcular AI converts a retinal microvascular video sequence into a structured hemodynamic analysis pipeline.

The system performs four major tasks:

A. Retinal Flow Extraction

Function: Detects motion inside retinal vessel regions and estimates frame-to-frame blood-flow-related movement.

B. Hemodynamic Analysis

Function: Transforms image-space motion into interpretable flow features such as:

mean velocity,

local velocity variance,

flow deceleration,

shear-related descriptors,

stagnation ratio,

and temporal flow instability.

C. Physics-Informed Physiological Estimation

Function: Uses engineered hemodynamic features as inputs to a compact neural network to estimate:

blood glucose level

dynamic blood viscosity

D. Pre-Occlusive Flow Anomaly Detection

Function: Detects and localizes persistent regions of abnormal flow reduction that may represent a pre-occlusive microvascular event.

<br>

<a id="2-clinical--engineering-motivation"></a>

<h2 align="center">🎯 2. Clinical & Engineering Motivation</h2>
<p align="center"><b>Why this problem matters and why retinal microcirculation is being investigated.</b></p>

2.1 Glucose Monitoring

Context: Conventional glucose measurement methods generally depend on:

capillary blood sampling,

venous laboratory analysis,

or interstitial-fluid-based continuous glucose monitoring.

Interstitial glucose and blood glucose are physiologically related but are not identical signals, especially during periods of rapid glucose change.

NeuroOcular AI explores whether retinal microvascular hemodynamics can provide additional non-invasive physiological information correlated with glucose state.

2.2 Microvascular Thrombosis

Context: Many vascular imaging techniques are optimized for macroscopic pathology.

NeuroOcular AI instead focuses on the earlier hemodynamic question:

Can persistent localized flow deceleration be detected before complete microvascular obstruction?

The system therefore analyzes dynamic flow behavior, rather than searching only for a visible static clot.

<br>

<a id="3-research-hypothesis"></a>

<h2 align="center">🧪 3. Research Hypothesis</h2>
<p align="center"><b>The scientific assumptions behind the proposed system.</b></p>

Overview: The project is built around three linked hypotheses.

Hypothesis 1 — Retinal Flow as a Physiological Biomarker

Hypothesis: Retinal microvascular circulation may carry measurable information about systemic physiological state.

Hypothesis 2 — Rheological Coupling

Hypothesis: Changes in variables such as:

glucose,

plasma properties,

erythrocyte deformability,

hematocrit,

vessel diameter,

and shear conditions

can influence microvascular flow patterns.

Hypothesis 3 — Early Flow Restriction Signature

Hypothesis: A developing local obstruction may create a detectable pattern of:

flow reduction,

spatial asymmetry,

abnormal velocity gradients,

and persistent local stagnation

before total lumen occlusion.

<br>

<a id="4-project-objectives"></a>

<h2 align="center">✅ 4. Project Objectives</h2>
<p align="center"><b>The concrete engineering goals of the prototype.</b></p>

Overview: The technical objectives are:

Process retinal microvascular video.

Detect motion-sensitive vascular regions.

Estimate dense frame-to-frame motion.

Extract hemodynamic features.

Estimate relative microvascular velocity.

Approximate wall shear-related parameters.

Estimate dynamic blood viscosity.

Estimate glucose-related physiological output.

Monitor spatial flow behavior over time.

Detect persistent local flow deceleration.

Localize suspicious anomaly coordinates.

Produce real-time visualization through Streamlit.

Deploy the learned predictor through ONNX Runtime.

Keep inference lightweight enough for edge deployment.

<br>

<a id="5-what-the-system-does"></a>

<h2 align="center">⚙️ 5. What the System Does</h2>
<p align="center"><b>A functional view of the pipeline from video input to outputs.</b></p>

INPUT
  ↓
Retinal Microvascular Video
  ↓
Preprocessing
  ↓
Motion-Based ROI Detection
  ↓
Dense Optical Flow
  ↓
Velocity / Flow Feature Extraction
  ↓
────────────────────────────────────────────
│                                          │
▼                                          ▼
Physics-Informed Predictor           Temporal Anomaly Engine
│                                          │
├─ Estimated Glucose                       ├─ Flow Drop
├─ Estimated Viscosity                     ├─ Z-Score Map
└─ Hemodynamic Indicators                  ├─ Anomaly Core
                                           └─ Severity Estimate

<br>

<a id="6-end-to-end-architecture"></a>

<h2 align="center">🏗️ 6. End-to-End Architecture</h2>
<p align="center"><b>How all modules connect in one computational workflow.</b></p>

flowchart TD

    A[Retinal Video Sequence] --> B[Frame Acquisition]
    B --> C[Frame Preprocessing]

    C --> C1[Grayscale / Channel Processing]
    C --> C2[Denoising]
    C --> C3[Normalization]
    C --> C4[Contrast Enhancement]

    C --> D[Cumulative Motion Mask]
    D --> E[Retinal Motion ROI]

    E --> F[Dense Optical Flow]
    F --> F1[Farneback Algorithm]

    F1 --> G[Velocity Magnitude Field]
    F1 --> H[Motion Direction Field]

    G --> I[Hemodynamic Feature Extraction]

    I --> I1[Mean Velocity]
    I --> I2[Velocity STD]
    I --> I3[Temporal Gradient]
    I --> I4[Shear-Related Features]
    I --> I5[Stagnation Ratio]
    I --> I6[Pulsatility Features]

    I --> J[Feature Normalization]
    J --> K[NeuroOcularNet]

    K --> L[Estimated Glucose]
    K --> M[Estimated Dynamic Viscosity]

    G --> N[Spatio-Temporal Baseline]
    N --> O[Z-Score Anomaly Map]
    O --> P[Local Nadir / Stagnation Detection]
    P --> Q[Bounding Region]
    P --> R[Anomaly Coordinates]
    P --> S[Severity Estimate]

    L --> T[Streamlit Dashboard]
    M --> T
    Q --> T
    R --> T
    S --> T

<br>

<p align="right"><a href="#top">↑ Back to top</a> · <a href="#7-complete-algorithm-inventory">Next chapter →</a></p>

<div align="center">

CHAPTER II

👁️ Vision, Flow & Physics

Computer Vision → Velocimetry → Hemodynamics → Anomaly Detection

</div>

<br>

<a id="7-complete-algorithm-inventory"></a>

<h2 align="center">🧠 7. Complete Algorithm Inventory</h2>
<p align="center"><b>A single reference table for every core algorithm and method.</b></p>

Overview: This section documents the core algorithms used by the project.

#

Algorithm / Method

Category

Purpose

Input

Output

1

Frame Sampling

Video Processing

Read retinal sequence frame-by-frame

Video

Frames

2

Grayscale / Intensity Conversion

Image Processing

Reduce dimensionality for motion analysis

RGB frame

Intensity image

3

Denoising

Preprocessing

Reduce image noise before optical flow

Frame

Smoothed frame

4

Intensity Normalization

Preprocessing

Reduce illumination variability

Frame

Normalized frame

5

Cumulative Motion Masking

Motion Segmentation

Identify repeatedly moving vascular regions

Consecutive frames

ROI mask

6

Farneback Dense Optical Flow

Computer Vision

Estimate dense pixel-wise displacement

Frame pair

2D flow field

7

Flow Magnitude Calculation

Motion Analysis

Convert x/y displacement into speed magnitude

u, v vectors

Velocity magnitude

8

Temporal Averaging

Signal Processing

Build local baseline behavior

Velocity history

Baseline map

9

Standard Deviation Mapping

Statistical Modeling

Measure expected temporal flow variation

Velocity history

σ map

10

Spatial-Temporal Z-Score

Anomaly Detection

Detect abnormal local flow reduction

Current + baseline flow

Z-map

11

Local Nadir Detection

Anomaly Localization

Find strongest stagnation point

Z-map / velocity map

(x, y)

12

Threshold-Based Region Detection

Decision Logic

Isolate abnormal region

Z-map

Binary anomaly mask

13

Bounding Box Localization

Computer Vision

Display suspicious region

Binary mask

Bounding box

14

Poiseuille-Type Velocity Model

Fluid Mechanics

Model idealized vessel flow profile

Radius + velocity

Velocity profile

15

Wall Shear Approximation

Hemodynamics

Estimate shear-related flow conditions

Velocity + radius

Shear rate

16

Non-Newtonian Viscosity Model

Rheology

Relate viscosity to shear state

Shear + physiological terms

Viscosity

17

Feature Normalization

Machine Learning

Scale feature vector for model inference

Raw features

Normalized features

18

NeuroOcularNet

Neural Network

Predict physiological targets

Feature vector

Glucose + viscosity

19

ONNX Graph Execution

Edge AI

Lightweight model inference

Tensor

Prediction tensor

20

Severity Estimation

Decision Layer

Quantify anomaly intensity

Anomaly metrics

Severity value

<br>

<a id="8-computer-vision-pipeline"></a>

<h2 align="center">👁️ 8. Computer Vision Pipeline</h2>
<p align="center"><b>How retinal frames are prepared before motion estimation.</b></p>

8.1 Video Frame Acquisition

Purpose: The video is decomposed into a sequence:

[
I_1, I_2, I_3, \dots, I_T
]

Where:

(I_t) is the frame at time (t)

(T) is the total number of frames.

8.2 Frame Preprocessing

Purpose: Preprocessing is used to improve stability before motion estimation.

Typical operations include:

Intensity Conversion

Purpose: Converts the video representation into an intensity domain suitable for motion calculation.

Denoising

Purpose: Reduces:

sensor noise,

compression artifacts,

local flickering.

Normalization

Purpose: Reduces sensitivity to global brightness changes.

Conceptually:

\frac{I-\mu_I}{\sigma_I+\epsilon}
]

Contrast Enhancement

Purpose: Used when vessel boundaries or intravascular intensity patterns have insufficient local contrast.

<br>

<a id="9-retinal-motion--velocimetry"></a>

<h2 align="center">🎞️ 9. Retinal Motion & Velocimetry</h2>
<p align="center"><b>How dense optical flow is used to estimate retinal motion.</b></p>

9.1 Cumulative Motion Mask

Purpose: A cumulative motion representation identifies pixels that repeatedly exhibit frame-to-frame change.

Conceptual formulation:

|I_t(x,y)-I_{t-1}(x,y)|
]

Cumulative formulation:

\sum_{t=2}^{T} M_t(x,y)
]

ROI thresholding:

\begin{cases}
1 & M_{cum}(x,y) > \tau_m \
0 & \text{otherwise}
\end{cases}
]

This suppresses static retinal structures and focuses processing on dynamically changing regions.

9.2 Farneback Dense Optical Flow

Algorithm: The project uses the Gunnar Farneback dense optical flow algorithm.

Unlike sparse tracking methods, Farneback estimates a motion vector for a large fraction of image pixels.

Flow field:

(u(x,y), v(x,y))
]

Where:

(u(x,y)) = horizontal displacement

(v(x,y)) = vertical displacement.

Flow magnitude:

\sqrt{u(x,y)^2+v(x,y)^2}
]

This magnitude is treated as an image-space proxy for local motion velocity.

Why Farneback?

It is useful for this prototype because it is:

dense,

deterministic,

computationally lightweight,

available in OpenCV,

suitable for frame-to-frame motion,

practical for real-time or near-real-time analysis.

Relevant Farneback Parameters

The OpenCV implementation exposes parameters such as:

pyr_scale
levels
winsize
iterations
poly_n
poly_sigma
flags

These parameters control:

image pyramid scale,

multi-resolution depth,

neighborhood size,

refinement iterations,

local polynomial approximation,

smoothing behavior.

Tuning Note: The exact values should be tuned against the retinal acquisition setup.

9.3 Image-Space Velocity

Purpose: If optical flow displacement is measured in:

**`pixels / frame`**

then image-space velocity can be converted to:

**`pixels / second`**

using:

v_{px/frame}\times FPS
]

Absolute physiological velocity requires spatial calibration.

If the calibration factor is:

[
C = \frac{\mu m}{pixel}
]

then:

v_{px/s}\times C
]

Without this calibration, the system must not report optical-flow magnitude as true physical blood velocity.

<br>

<a id="10-hemodynamic-feature-engineering"></a>

<h2 align="center">📈 10. Hemodynamic Feature Engineering</h2>
<p align="center"><b>How motion fields become interpretable flow features.</b></p>

Overview: The optical-flow field is transformed into a compact feature vector.

10.1 Mean Flow Velocity

\frac{1}{N}
\sum_{i=1}^{N}v_i
]

Represents average motion inside the selected vascular ROI.

10.2 Velocity Standard Deviation

\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(v_i-\bar{v})^2
}
]

Measures flow heterogeneity.

10.3 Maximum Local Velocity

\max(v_1,\dots,v_N)
]

Useful for identifying the fastest local motion region.

10.4 Temporal Velocity Gradient

v_t-v_{t-1}
]

Captures acceleration or deceleration over time.

10.5 Relative Flow Drop

\frac{
v_{baseline}-v_{current}
}{
v_{baseline}+\epsilon
}
]

A high value indicates substantial local deceleration.

10.6 Stagnation Ratio

Metric: One useful implementation is:

\frac{
N(v<\tau_s)
}{
N_{ROI}
}
]

Where:

(\tau_s) = low-flow threshold

(N(v<\tau_s)) = number of low-flow pixels.

10.7 Vessel / Motion Occupancy

\frac{
N_{active}
}{
N_{image}
}
]

Describes how much of the analyzed region contains significant motion.

10.8 Pulsatility-Related Index

Metric: If a temporal waveform is available:

\frac{
v_{max}-v_{min}
}{
v_{mean}+\epsilon
}
]

This can characterize temporal flow variation.

<br>

<a id="11-biophysical--rheological-modeling"></a>

<h2 align="center">🩸 11. Biophysical & Rheological Modeling</h2>
<p align="center"><b>The fluid-mechanics equations used to interpret those features.</b></p>

11.1 Poiseuille-Type Velocity Profile

Model: A simplified laminar flow profile can be represented as:

v_{max}
\left(
1-\frac{r^2}{R^2}
\right)
]

Where:

(r) = radial distance from vessel center

(R) = vessel radius

(v_{max}) = centerline velocity.

Important Note

Real microvascular blood flow is more complex than ideal Poiseuille flow because:

blood is non-Newtonian,

red blood cells deform,

hematocrit varies,

vessel walls are biological and elastic,

capillary diameters approach cellular dimensions.

Therefore this equation is used as a modeling approximation, not as a complete description of retinal blood flow.

11.2 Wall Shear Rate

Metric: The general expression is:

\left.
\frac{\partial v}{\partial r}
\right|_{r=R}
]

For simplified laminar flow:

[
\dot{\gamma}_{wall}
\approx
\frac{4\bar{v}}{R}
]

11.3 Shear Stress

Metric: If viscosity is available:

\eta\dot{\gamma}
]

Where:

(\tau) = shear stress

(\eta) = apparent viscosity

(\dot{\gamma}) = shear rate.

11.4 Non-Newtonian Viscosity Model

Model: A generalized Carreau-Yasuda-style formulation can be written as:

\eta_\infty
+
\left(
\eta_0(G)-\eta_\infty
\right)
\left[
1+(\lambda\dot{\gamma})^a
\right]^{\frac{n-1}{a}}
]

Where:

Symbol

Meaning

(\eta)

apparent dynamic viscosity

(\eta_0)

low-shear viscosity

(\eta_\infty)

high-shear limiting viscosity

(\dot{\gamma})

shear rate

(\lambda)

time constant

(a)

transition parameter

(n)

flow behavior index

(G)

glucose-related physiological term

The glucose-to-rheology relationship must be calibrated empirically using paired physiological measurements.

<br>

<a id="12-spatio-temporal-anomaly-detection"></a>

<h2 align="center">🚨 12. Spatio-Temporal Anomaly Detection</h2>
<p align="center"><b>How persistent local flow reduction is detected and localized.</b></p>

Overview: The anomaly subsystem operates independently from the physiological regression output.

12.1 Baseline Flow Map

Purpose: For each pixel or vascular region:

\frac{1}{T_b}
\sum_{t=1}^{T_b}
v(x,y,t)
]

12.2 Local Temporal Variance

\sqrt{
\frac{1}{T_b}
\sum
\left(
v(x,y,t)-\bar{v}_{baseline}(x,y)
\right)^2
}
]

12.3 Spatio-Temporal Z-Score

Purpose: The anomaly score is:

\frac{
\bar{v}_{baseline}(x,y)-v(x,y,t)
}{
\sigma_v(x,y)+\epsilon
}
]

Interpretation:

Low Z-score
    ↓
Flow behaves near baseline

High positive Z-score
    ↓
Current flow is substantially lower than baseline

12.4 Local Nadir Detection

Purpose: The strongest local deceleration region can be estimated from:

\arg\max_{x,y} Z(x,y,t)
]

or equivalently from the local minimum velocity.

The system can then generate:

anomaly center,

bounding region,

deceleration heatmap,

persistence score.

12.5 Persistence Logic

Purpose: A single abnormal frame should not automatically represent a thrombotic event.

A more reliable decision rule includes temporal persistence:

\sum_{t=t_0}^{t_1}
\mathbb{1}
\left[
Z(x,y,t)>\tau_Z
\right]
]

Persistent anomalies are more important than isolated spikes.

<br>

<p align="right"><a href="#top">↑ Back to top</a> · <a href="#13-ai-model-neuroocularnet">Next chapter →</a></p>

<div align="center">

CHAPTER III

🤖 AI & Inference

Feature Vector → NeuroOcularNet → Training → ONNX → Application

</div>

<br>

<a id="13-ai-model-neuroocularnet"></a>

<h2 align="center">🤖 13. AI Model: NeuroOcularNet</h2>
<p align="center"><b>The lightweight regression model used for physiological estimation.</b></p>

Overview: The learned prediction component is exported as:

**`NeuroOcularNet.onnx`**

Current model footprint:

≈ 10.28 KB

The model receives engineered hemodynamic features rather than raw retinal video.

This produces a hybrid architecture:

Physics / Computer Vision
        +
Feature Engineering
        +
Neural Regression

instead of:

Raw Video
   ↓
Opaque End-to-End Black Box

13.1 Model Role

Role: NeuroOcularNet is responsible for mapping the extracted feature space to physiological prediction targets.

Conceptually:

[
x_1,x_2,\dots,x_n
]
]

and:

[
\hat{G},
\hat{\eta}
]
]

Where:

(\mathbf{x}) = normalized hemodynamic feature vector

(\hat{G}) = estimated glucose

(\hat{\eta}) = estimated viscosity

(\theta) = learned network parameters.

13.2 Why a Small Neural Network?

Reason: A compact model provides:

very low inference latency,

low memory consumption,

easier edge deployment,

simple ONNX export,

reduced computational requirements,

easier integration into a real-time dashboard.

<br>

<a id="14-model-inputs--outputs"></a>

<h2 align="center">🔢 14. Model Inputs & Outputs</h2>
<p align="center"><b>What enters the model and what the model returns.</b></p>

14.1 Candidate Input Feature Vector

Input: The feature vector may include:

mean_velocity
velocity_std
maximum_velocity
temporal_velocity_gradient
relative_flow_drop
estimated_shear_rate
stagnation_ratio
motion_occupancy
pulsatility_index
anomaly_score

The exact feature order used during inference must match the feature order used during training.

14.2 Model Output

Output: The model produces physiological estimates such as:

Output 1 → Estimated Blood Glucose (mg/dL)
Output 2 → Estimated Dynamic Viscosity (mPa·s)

<br>

<a id="15-training--normalization-pipeline"></a>

<h2 align="center">🧬 15. Training & Normalization Pipeline</h2>
<p align="center"><b>How features are scaled, trained, evaluated, and exported.</b></p>

Overview: A typical training flow is:

Raw / Simulated Physiological Samples
             ↓
Hemodynamic Feature Construction
             ↓
Feature Matrix X
             ↓
Target Matrix Y
             ↓
Train / Validation Split
             ↓
Feature Normalization
             ↓
NeuroOcularNet Training
             ↓
Loss Optimization
             ↓
Model Evaluation
             ↓
ONNX Export

15.1 Normalization

Purpose: For a feature (x):

\frac{x-\mu_x}{\sigma_x+\epsilon}
]

or using min-max scaling:

\frac{x-x_{min}}{x_{max}-x_{min}}
]

The same training normalization parameters must be reused during inference.

15.2 Regression Loss

Training Metric: The reported model convergence uses Mean Squared Error.

\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
]

Prototype reported scaled MSE:

**`0.0421`**

This value should be interpreted in the context of the normalization and target scaling used during training.

<br>

<a id="16-onnx-edge-deployment"></a>

<h2 align="center">⚡ 16. ONNX Edge Deployment</h2>
<p align="center"><b>How the trained model is packaged for lightweight inference.</b></p>

Overview: The trained model is converted into Open Neural Network Exchange (ONNX) format.

Files:

**`NeuroOcularNet.onnx`**
**`NeuroOcularNet.onnx.data`**

The ONNX runtime stage is:

Normalized Feature Vector
          ↓
ONNX Runtime Session
          ↓
NeuroOcularNet Graph
          ↓
Prediction Tensor
          ↓
Postprocessing
          ↓
Dashboard Output

16.1 Why ONNX?

Reason: ONNX provides:

portable model representation,

fast CPU inference,

deployment independence,

compatibility with multiple training frameworks,

smaller production runtime,

edge-device suitability.

16.2 Inference Target

Target: Prototype target:

**`< 5 ms / inference`**

Runtime depends on:

CPU,

ONNX Runtime version,

number of threads,

preprocessing overhead,

model input size,

deployment platform.

<br>

<a id="17-application-layer"></a>

<h2 align="center">🖥️ 17. Application Layer</h2>
<p align="center"><b>How Streamlit exposes the analysis pipeline to the user.</b></p>

Overview: The user interface is implemented using:

**`Streamlit`**

The application is responsible for:

loading retinal videos,

displaying frames,

visualizing flow information,

running the feature pipeline,

loading the ONNX model,

displaying glucose estimation,

displaying viscosity estimation,

rendering anomaly alerts,

showing bounding boxes,

showing severity indicators,

presenting clinical-style status cards.

Custom styling is stored in:

**`style.css`**

<br>

<p align="right"><a href="#top">↑ Back to top</a> · <a href="#18-technology-stack">Next chapter →</a></p>

<div align="center">

CHAPTER IV

🧰 Engineering & Evaluation

Technology Stack → Repository → Scenarios → Results → Metrics

</div>

<br>

<a id="18-technology-stack"></a>

<h2 align="center">🧰 18. Technology Stack</h2>
<p align="center"><b>Languages, libraries, runtimes, and mathematical components.</b></p>

Programming Language

Technology

Role

Python

Main application and analysis language

Computer Vision

Technology

Role

OpenCV

Video decoding, image processing, optical flow, visualization

NumPy

Numerical arrays, vectorized flow calculations, statistics

Machine Learning / Inference

Technology

Role

ONNX

Portable serialized neural-network model

ONNX Runtime

Fast production inference

NeuroOcularNet

Lightweight physiological regression model

Web Application

Technology

Role

Streamlit

Interactive medical research dashboard

CSS

Custom UI styling

Mathematical / Algorithmic Components

Component

Role

Farneback Optical Flow

Dense motion estimation

Cumulative Motion Mask

Dynamic ROI extraction

Velocity Magnitude

Flow proxy

Temporal Baseline

Expected local flow

Z-Score Analysis

Flow anomaly detection

Nadir Localization

Strongest stagnation point

Poiseuille Approximation

Idealized flow profile

Wall Shear Approximation

Hemodynamic feature

Non-Newtonian Rheology

Viscosity modeling

MSE

Regression training objective

<br>

<a id="19-repository-structure"></a>

<h2 align="center">📂 19. Repository Structure</h2>
<p align="center"><b>Where each project file lives and what it does.</b></p>

NeuroOcular-AI/
│
├── app.py
│   ├── Streamlit UI
│   ├── video input
│   ├── preprocessing
│   ├── optical-flow execution
│   ├── feature extraction
│   ├── ONNX inference
│   └── result visualization
│
├── NeuroOcularNet.onnx
│   └── optimized neural-network graph
│
├── NeuroOcularNet.onnx.data
│   └── ONNX external tensor data
│
├── requirements.txt
│   └── project dependencies
│
├── style.css
│   └── dashboard styling
│
├── samples/
│   ├── 01_normal_euglycemic_flow.avi
│   ├── 02_mild_hyperglycemia.avi
│   ├── 03_acute_hyperglycemia.avi
│   ├── 04_early_platelet_cluster.avi
│   ├── 05_focal_micro_thrombus.avi
│   ├── 06_severe_luminal_stenosis.avi
│   ├── 07_pulsatile_shear_stress.avi
│   ├── 08_sluggish_microcirculation.avi
│   ├── 09_transient_ischemic_event.avi
│   └── 10_near_complete_occlusion.avi
│
├── LICENSE
│
└── README.md

<br>

<a id="20-sample-clinical-scenarios"></a>

<h2 align="center">🎥 20. Sample Clinical Scenarios</h2>
<p align="center"><b>The included benchmark scenarios and their intended role.</b></p>

Overview: The repository contains ten example scenarios.

File

Scenario

Main Purpose

01_normal_euglycemic_flow.avi

Normal flow

Baseline reference

02_mild_hyperglycemia.avi

Mild hyperglycemia

Mild rheological change

03_acute_hyperglycemia.avi

Acute hyperglycemia

Stronger hemodynamic shift

04_early_platelet_cluster.avi

Early aggregation

Early localized deceleration

05_focal_micro_thrombus.avi

Focal micro-thrombus

Local anomaly localization

06_severe_luminal_stenosis.avi

Severe stenosis

High resistance / reduced flow

07_pulsatile_shear_stress.avi

Pulsatile flow

Temporal shear analysis

08_sluggish_microcirculation.avi

Sluggish circulation

General flow reduction

09_transient_ischemic_event.avi

Transient ischemic event

Temporary anomaly behavior

10_near_complete_occlusion.avi

Near occlusion

Extreme flow restriction

These scenarios should be clearly identified as simulation / benchmark scenarios unless acquired from validated clinical datasets.

<br>

<a id="21-experimental-results"></a>

<h2 align="center">📊 21. Experimental Results</h2>
<p align="center"><b>Prototype outputs reported during development.</b></p>

Overview: Prototype outputs reported during development include:

Metric

Reported Prototype Value

Scaled MSE

0.0421

Example viscosity

3.42 – 3.46 mPa·s

Example glucose output

118.3 – 143.8 mg/dL

Example anomaly core

(128, 128)

Example pre-occlusive stage

33%

ONNX model size

10.28 KB

Target model inference

< 5 ms

Interpretation

Meaning: These values demonstrate that the computational system can:

process the video,

extract a motion field,

generate hemodynamic features,

execute the regression model,

generate physiological estimates,

detect spatial flow anomalies,

localize an anomaly,

and run a lightweight inference model.

They do not establish clinical diagnostic accuracy.

<br>

<a id="22-performance-metrics"></a>

<h2 align="center">📐 22. Performance Metrics</h2>
<p align="center"><b>Metrics needed to evaluate regression and detection performance.</b></p>

Overview: For technical evaluation, the project may report:

Regression Metrics

Mean Absolute Error

\frac{1}{N}
\sum |y_i-\hat{y}_i|
]

Root Mean Squared Error

\sqrt{
\frac{1}{N}
\sum(y_i-\hat{y}_i)^2
}
]

Mean Absolute Relative Difference

For future glucose validation:

\frac{100}{N}
\sum
\left|
\frac{
G_i-\hat{G}_i
}{
G_i
}
\right|
]

Detection Metrics

Evaluation: For validated anomaly labels:

Sensitivity
Specificity
Precision
Recall
F1-score
ROC-AUC
False Positive Rate

<br>

<p align="right"><a href="#top">↑ Back to top</a> · <a href="#23-installation">Next chapter →</a></p>

<div align="center">

CHAPTER V

▶️ Run & Use the Platform

Install → Run → Deploy → Input → Output

</div>

<br>

<a id="23-installation"></a>

<h2 align="center">💻 23. Installation</h2>
<p align="center"><b>Local environment setup.</b></p>

Requirements

Setup: Recommended:

Python 3.9
Python 3.10
Python 3.11

Check version:

python --version

Clone

git clone https://github.com/yousefosamaahmed/NeuroOcular-AI.git
cd NeuroOcular-AI

Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

Install Packages

pip install --upgrade pip
pip install -r requirements.txt

<br>

<a id="24-running-the-application"></a>

<h2 align="center">▶️ 24. Running the Application</h2>
<p align="center"><b>How to launch the Streamlit interface.</b></p>

streamlit run app.py

Typical local URL:

**`http://localhost:8501`**

<br>

<a id="25-cloud-deployment"></a>

<h2 align="center">☁️ 25. Cloud Deployment</h2>
<p align="center"><b>How to deploy the project to Streamlit Community Cloud.</b></p>

Streamlit Community Cloud

Deployment: Deploy with:

Repository:
yousefosamaahmed/NeuroOcular-AI

Branch:
main

Application file:
app.py

Required deployment files:

app.py
requirements.txt
**`NeuroOcularNet.onnx`**
**`style.css`**

If external tensor data is required:

**`NeuroOcularNet.onnx.data`**

must also be deployed.

<br>

<a id="26-input-requirements"></a>

<h2 align="center">📥 26. Input Requirements</h2>
<p align="center"><b>Recommended properties for retinal video input.</b></p>

Overview: Recommended input properties:

Parameter

Recommendation

File type

AVI / MP4

Acquisition

Stable retinal microvascular imaging

Frame rate

Constant

Exposure

Stable

Focus

Sharp vessel structures

Compression

Low / moderate

Eye movement

Minimal

Camera movement

Minimal

Duration

Enough frames to establish temporal baseline

<br>

<a id="27-output-interpretation"></a>

<h2 align="center">📤 27. Output Interpretation</h2>
<p align="center"><b>How to interpret glucose, viscosity, and anomaly outputs.</b></p>

Glucose Estimate

Output: Example:

Estimated Glucose
126 mg/dL

This is a model prediction, not a direct chemical measurement.

Viscosity Estimate

Output: Example:

Estimated Dynamic Viscosity
3.44 mPa·s

This is an inferred model output.

Anomaly Detection

Development Area: Example:

Potential Localized Flow Anomaly

Core:
X = 128
Y = 128

Estimated Severity:
33%

This means the system detected a region of statistically abnormal flow reduction.

It does not independently confirm the presence of a thrombus.

<br>

<p align="right"><a href="#top">↑ Back to top</a> · <a href="#28-failure-modes--limitations">Next chapter →</a></p>

<div align="center">

CHAPTER VI

🔬 Validation & Next Steps

Limitations → Clinical Validation → Future Work → Governance

</div>

<br>

<a id="28-failure-modes--limitations"></a>

<h2 align="center">⚠️ 28. Failure Modes & Limitations</h2>
<p align="center"><b>Where the current prototype can fail or overstate certainty.</b></p>

28.1 Optical Flow Is Not Direct Blood Velocity

Limitation: Farneback returns image displacement.

Without calibration:

pixel displacement ≠ mm/s

Absolute blood velocity requires:

frame rate,

image scale,

retinal magnification,

vessel geometry.

28.2 Glucose Is Not Uniquely Determined by Flow

Limitation: Flow is influenced by multiple confounders:

glucose,

blood pressure,

hematocrit,

hydration,

temperature,

vessel diameter,

vascular tone,

medications,

cardiovascular status,

retinal disease,

diabetes duration.

Therefore the glucose predictor requires clinical calibration.

28.3 Eye Motion

Limitation: Eye motion can create false optical flow.

Potential improvements:

global image registration,

feature-based stabilization,

homography correction,

retinal landmark tracking.

28.4 Illumination Variation

Limitation: Brightness changes can mimic motion.

Potential countermeasures:

histogram normalization,

CLAHE,

photometric correction,

temporal illumination compensation.

28.5 Motion Blur

Limitation: Blur may reduce vessel-edge quality and corrupt optical-flow estimation.

28.6 Incorrect ROI

Limitation: If the cumulative motion mask includes nonvascular motion, all downstream features may be biased.

28.7 Synthetic Data Risk

Limitation: If training or benchmark samples are simulated, model performance may not generalize to real patients.

This is one of the most important limitations of the current research stage.

<br>

<a id="29-clinical-validation-roadmap"></a>

<h2 align="center">🔬 29. Clinical Validation Roadmap</h2>
<p align="center"><b>The staged path from prototype to credible clinical evidence.</b></p>

Phase 1 — Algorithm Validation

Phase Goal: Validate:

optical-flow consistency,

synthetic displacement recovery,

noise sensitivity,

frame-rate robustness,

motion stabilization.

Phase 2 — Calibration Study

Phase Goal: Collect synchronized:

Retinal video
Venous glucose
Capillary glucose
Hematocrit
Blood pressure
Heart rate
Viscosity measurement
Clinical status

Phase 3 — Prospective Human Study

Phase Goal: Evaluate:

MAE
RMSE
MARD
Bland-Altman Agreement
Clarke Error Grid
Parkes Error Grid
Sensitivity
Specificity
ROC-AUC

where appropriate.

Phase 4 — External Validation

Phase Goal: Validate on:

different hospitals,

different retinal cameras,

different age groups,

different ethnicities,

diabetic and non-diabetic cohorts,

patients with vascular comorbidities.

Phase 5 — Regulatory Development

Phase Goal: Any clinical product would require a defined intended use, quality-management process, risk analysis, medical-device software lifecycle controls, and the appropriate regulatory pathway for the target jurisdiction.

<br>

<a id="30-future-development"></a>

<h2 align="center">🚀 30. Future Development</h2>
<p align="center"><b>Planned technical improvements.</b></p>

Overview: Planned improvements may include:

Vision

retinal vessel segmentation,

U-Net vessel masks,

transformer-based segmentation,

image registration,

optical stabilization,

sub-pixel velocity estimation,

deep optical flow.

Hemodynamics

vessel diameter estimation,

centerline extraction,

branch-level velocity analysis,

calibrated absolute velocity,

vessel-specific shear estimation,

pulsatility analysis.

AI

uncertainty-aware regression,

temporal neural networks,

LSTM / GRU sequence modeling,

temporal transformers,

multimodal fusion,

patient-specific calibration.

Anomaly Detection

adaptive thresholds,

temporal persistence models,

segmentation-based anomaly regions,

confidence scoring,

false-positive suppression.

Deployment

mobile inference,

embedded edge hardware,

hardware acceleration,

quantized ONNX models,

real-time camera integration.

<br>

<a id="31-security--privacy"></a>

<h2 align="center">🔐 31. Security & Privacy</h2>
<p align="center"><b>Data-protection requirements for future clinical use.</b></p>

Overview: A clinical-grade implementation should include:

anonymization,

encryption at rest,

encryption in transit,

access control,

authentication,

audit logs,

secure model storage,

patient consent controls,

retention policies.

Never upload personally identifiable patient data to a public GitHub repository.

<br>

<a id="32-intellectual-property"></a>

<h2 align="center">⚖️ 32. Intellectual Property</h2>
<p align="center"><b>Potentially protectable technical components.</b></p>

Overview: Potential IP-relevant components may include:

the combined retinal hemodynamic inference pipeline,

physics-informed feature construction,

dual glucose/viscosity regression,

spatio-temporal pre-occlusive flow scoring,

edge-deployed retinal anomaly inference,

real-time microvascular risk visualization.

Patentability and filing status should only be stated after review by a qualified intellectual-property professional.

<br>

<a id="33-license"></a>

<h2 align="center">📜 33. License</h2>
<p align="center"><b>Project licensing information.</b></p>

Overview: Distributed under the MIT License unless otherwise specified.

See:

LICENSE

<br>

<a id="34-citation"></a>

<h2 align="center">🧾 34. Citation</h2>
<p align="center"><b>Suggested academic citation.</b></p>

Overview: Suggested repository citation:

@software{neuroocular_ai_2026,
  title  = {NeuroOcular AI: Physics-Informed Retinal Hemodynamics Platform},
  author = {Yousef Osama Ahmed},
  year   = {2026},
  url    = {https://github.com/yousefosamaahmed/NeuroOcular-AI}
}

<br>

<a id="35-author"></a>

<h2 align="center">👤 35. Author</h2>
<p align="center"><b>Project ownership and repository information.</b></p>

Yousef Osama Ahmed

Project:

NeuroOcular AI
Physics-Informed Retinal Hemodynamics Platform

Repository:

https://github.com/yousefosamaahmed/NeuroOcular-AI

<br>

<h2 align="center">🔍 Technical Summary</h2>
<p align="center"><b>End-to-End System at a Glance</b></p>

Overview: For a reviewer who wants the entire project in one view:

RETINAL VIDEO
    │
    ├── Frame Acquisition
    │
    ├── Image Preprocessing
    │     ├── Intensity Conversion
    │     ├── Denoising
    │     └── Normalization
    │
    ├── Cumulative Motion Mask
    │
    ├── Dense Optical Flow
    │     └── Gunnar Farneback
    │
    ├── Velocity Magnitude
    │
    ├── Hemodynamic Features
    │     ├── Mean Velocity
    │     ├── Velocity Variance
    │     ├── Temporal Gradient
    │     ├── Flow Drop
    │     ├── Stagnation Ratio
    │     ├── Shear Rate
    │     └── Pulsatility
    │
    ├── Physics Layer
    │     ├── Poiseuille Approximation
    │     ├── Wall Shear Approximation
    │     └── Non-Newtonian Rheology
    │
    ├── AI Layer
    │     ├── Feature Normalization
    │     ├── NeuroOcularNet
    │     └── ONNX Runtime
    │
    ├── Physiological Outputs
    │     ├── Glucose Estimate
    │     └── Dynamic Viscosity Estimate
    │
    ├── Anomaly Layer
    │     ├── Temporal Baseline
    │     ├── Standard Deviation Map
    │     ├── Spatial-Temporal Z-Score
    │     ├── Local Nadir Detection
    │     ├── Bounding Box
    │     └── Severity Estimation
    │
    └── Streamlit Dashboard

<div align="center">

🩺 NeuroOcular AI

Physics-Informed Retinal Hemodynamics

Computer Vision • Biofluid Mechanics • Machine Learning • Edge AI

Research prototype for next-generation non-invasive microvascular analysis.
