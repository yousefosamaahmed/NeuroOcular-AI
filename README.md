<a id="top"></a>

<div align="center">

🩺 NeuroOcular AI

Physics-Informed Retinal Hemodynamics Platform

Retinal Computer Vision × Hemodynamics × Physics-Informed AI × Edge Inference

<br>









<br>

A research-oriented medical computer vision platform for retinal microvascular flow analysis, hemodynamic feature extraction, physiological estimation, and localized pre-occlusive flow anomaly detection.

</div>

🔎 Project at a Glance

Area

Implementation

Primary Input

Retinal microvascular video

Vision Engine

OpenCV

Motion Estimation

Gunnar Farneback Dense Optical Flow

ROI Extraction

Cumulative Motion Masking

Hemodynamic Features

Velocity, variance, shear-related metrics, stagnation, pulsatility

Anomaly Detection

Spatio-temporal baseline + Z-score analysis

AI Model

NeuroOcularNet

Model Format

ONNX

Runtime

ONNX Runtime

Web Interface

Streamlit

Primary Outputs

Glucose estimate, viscosity estimate, localized flow anomaly

Project Stage

Research prototype

<div align="center">

⚠️ Medical & Research Disclaimer

Research Use Only — Not a Clinically Validated Medical Device

</div>

NeuroOcular AI is an experimental research prototype intended for academic research, engineering validation, and technical demonstration.

It must not be used as a substitute for:

laboratory blood glucose measurement,

certified continuous glucose monitoring,

vascular imaging,

physician diagnosis,

emergency assessment,

anticoagulation decisions,

or any medical treatment decision.

Clinical translation requirement: A deployable clinical system would require prospective human studies, synchronized ground-truth measurements, independent external validation, safety testing, demographic robustness evaluation, and regulatory approval.

<div align="center">

📑 Table of Contents

Structured Technical Documentation

</div>

Core System

Algorithms & AI

Engineering & Validation

1. Project Summary

7. Algorithm Inventory

18. Technology Stack

2. Motivation

8. Computer Vision Pipeline

19. Repository Structure

3. Research Hypothesis

9. Retinal Motion & Velocimetry

20. Sample Scenarios

4. Objectives

10. Hemodynamic Features

21. Experimental Results

5. System Functions

11. Biophysical Modeling

22. Performance Metrics

6. End-to-End Architecture

12. Anomaly Detection

23. Installation



13. NeuroOcularNet

24. Run the App



14. Model Inputs & Outputs

25. Cloud Deployment



15. Training Pipeline

26. Input Requirements



16. ONNX Deployment

27. Output Interpretation



17. Application Layer

28. Limitations





29. Validation Roadmap





30. Future Development





31. Security & Privacy





32. Intellectual Property





33. License





34. Citation





35. Author

<a id="1-project-summary"></a>

<div align="center">

📌 1. Project Summary

What NeuroOcular AI is and what problem it is designed to investigate.

</div>

Overview

NeuroOcular AI is a hybrid medical-AI research platform that transforms retinal microvascular video into a structured hemodynamic analysis pipeline.

Instead of relying only on raw image intensity, the system combines:

retinal computer vision,

dense optical flow,

hemodynamic feature engineering,

fluid-mechanics approximations,

lightweight neural regression,

spatio-temporal anomaly detection,

and edge-oriented ONNX inference.

Core Outputs

The current prototype produces three main categories of output:

Output

Description

Estimated Blood Glucose

Model-based glucose-related estimate in mg/dL

Estimated Dynamic Viscosity

Inferred apparent blood viscosity in mPa·s

Flow Anomaly Alert

Localized region of abnormal persistent flow deceleration

System Philosophy

The project uses a hybrid physics + AI architecture.

Retinal Video
      ↓
Computer Vision
      ↓
Hemodynamic Features
      ↓
Physics-Informed Representation
      ↓
Neural Regression + Anomaly Detection
      ↓
Physiological & Spatial Outputs

The goal is interpretability: the AI model receives meaningful hemodynamic variables instead of being asked to infer everything directly from raw video.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="2-clinical--engineering-motivation"></a>

<div align="center">

🎯 2. Clinical & Engineering Motivation

Why retinal microcirculation is relevant to glucose-related physiology and vascular-flow monitoring.

</div>

2.1 Glucose Monitoring Challenge

Conventional glucose monitoring typically relies on:

finger-stick capillary blood sampling,

venous laboratory testing,

or Continuous Glucose Monitoring (CGM).

CGMs measure glucose primarily in interstitial fluid, not directly in circulating blood. During rapid glucose transitions, physiological lag can occur between blood and interstitial glucose values.

2.2 Microvascular Flow Challenge

Many conventional imaging methods are optimized for:

established vascular occlusion,

visible stenosis,

macroscopic thrombi,

or downstream perfusion deficits.

This project investigates an earlier-stage signal:

Can persistent local flow deceleration reveal a pre-occlusive vascular abnormality before complete obstruction?

2.3 Why the Retina?

The retinal vasculature is attractive for research because it provides direct optical access to microvascular structures.

This makes it a candidate environment for studying:

flow dynamics,

vessel-level motion,

local perfusion changes,

and temporal vascular anomalies.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="3-research-hypothesis"></a>

<div align="center">

🧪 3. Research Hypothesis

The scientific assumptions that connect retinal flow, rheology, glucose state, and pre-occlusive events.

</div>

Hypothesis A — Retinal Flow as a Physiological Signal

Retinal microvascular circulation may contain measurable information related to systemic physiological state.

Hypothesis B — Rheological Coupling

Changes in the following variables may alter local flow dynamics:

glucose,

plasma properties,

erythrocyte deformability,

hematocrit,

vessel diameter,

and shear conditions.

Hypothesis C — Localized Pre-Occlusive Signature

A developing local obstruction may generate a combination of:

reduced velocity,

abnormal spatial gradients,

increased stagnation,

temporal persistence,

and asymmetric flow behavior.

The anomaly subsystem is designed to detect this pattern computationally.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="4-project-objectives"></a>

<div align="center">

✅ 4. Project Objectives

The complete set of engineering goals implemented by the prototype.

</div>

Vision Objectives

Read retinal video sequences.

Preprocess frames for stable motion estimation.

Detect motion-sensitive vascular regions.

Compute dense optical flow.

Extract flow magnitude and direction.

Hemodynamic Objectives

Estimate relative microvascular motion.

Compute flow variability.

Approximate wall shear-related variables.

Detect stagnation and local flow drop.

derive pulsatility-related descriptors.

AI Objectives

Normalize engineered features.

Infer glucose-related output.

Infer viscosity-related output.

Export the trained model to ONNX.

Run low-latency edge inference.

Anomaly Detection Objectives

Build a temporal flow baseline.

Compute local deviation from baseline.

Detect persistent abnormal deceleration.

Localize the anomaly.

Generate a severity indicator.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="5-what-the-system-does"></a>

<div align="center">

⚙️ 5. What the System Does

The functional behavior of the system from raw video to final output.

</div>

Functional Flow

INPUT
  ↓
Retinal Microvascular Video
  ↓
Preprocessing
  ↓
Motion ROI
  ↓
Dense Optical Flow
  ↓
Flow / Hemodynamic Features
  ↓
────────────────────────────────────────────
│                                          │
▼                                          ▼
Physiological Inference             Anomaly Detection
│                                          │
├─ Glucose Estimate                       ├─ Z-Score Map
├─ Viscosity Estimate                     ├─ Nadir Point
└─ Hemodynamic Indicators                 ├─ Bounding Region
                                           └─ Severity Score

Functional Modules

Module

Responsibility

Video Module

Reads and samples retinal frames

Preprocessing Module

Stabilizes image quality

Motion Module

Extracts flow-related displacement

Hemodynamic Module

Converts motion into physiological descriptors

AI Module

Predicts glucose and viscosity

Anomaly Module

Detects local abnormal flow deceleration

UI Module

Displays results in Streamlit

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="6-end-to-end-architecture"></a>

<div align="center">

🏗️ 6. End-to-End Architecture

The complete computational pipeline and the relationship between all system components.

</div>

Architecture Diagram

flowchart TD

    A[Retinal Video] --> B[Frame Acquisition]
    B --> C[Preprocessing]

    C --> C1[Intensity Conversion]
    C --> C2[Denoising]
    C --> C3[Normalization]
    C --> C4[Contrast Enhancement]

    C --> D[Cumulative Motion Mask]
    D --> E[Motion ROI]

    E --> F[Farneback Dense Optical Flow]
    F --> G[Velocity Magnitude]
    F --> H[Flow Direction]

    G --> I[Hemodynamic Feature Engine]

    I --> I1[Mean Velocity]
    I --> I2[Velocity STD]
    I --> I3[Flow Drop]
    I --> I4[Shear Features]
    I --> I5[Stagnation Ratio]
    I --> I6[Pulsatility]

    I --> J[Feature Normalization]
    J --> K[NeuroOcularNet]

    K --> L[Glucose Estimate]
    K --> M[Viscosity Estimate]

    G --> N[Temporal Baseline]
    N --> O[Z-Score Map]
    O --> P[Local Nadir Detection]
    P --> Q[Bounding Box]
    P --> R[Coordinates]
    P --> S[Severity]

    L --> T[Streamlit Dashboard]
    M --> T
    Q --> T
    R --> T
    S --> T

Architecture Layers

Layer

Main Components

Acquisition

Retinal video

Preprocessing

Denoising, normalization, contrast

Computer Vision

Motion mask + optical flow

Hemodynamics

Velocity, shear, stagnation, pulsatility

AI

NeuroOcularNet

Anomaly Detection

Z-score + persistence + localization

Deployment

ONNX Runtime

Presentation

Streamlit

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="7-algorithm-inventory"></a>

<div align="center">

🧠 7. Algorithm Inventory

Every major algorithm and computational method used by the platform.

</div>

Complete Algorithm Table

#

Algorithm / Method

Category

Purpose

Input

Output

1

Frame Sampling

Video Processing

Read sequence frame-by-frame

Video

Frames

2

Grayscale / Intensity Conversion

Image Processing

Simplify motion representation

RGB frame

Intensity frame

3

Denoising

Preprocessing

Suppress noise

Frame

Smoothed frame

4

Intensity Normalization

Preprocessing

Reduce illumination variation

Frame

Normalized frame

5

Contrast Enhancement

Preprocessing

Improve local visual structure

Frame

Enhanced frame

6

Cumulative Motion Masking

Motion Segmentation

Detect repeatedly moving regions

Frame sequence

ROI mask

7

Farneback Dense Optical Flow

Computer Vision

Estimate pixel-wise displacement

Consecutive frames

Flow field

8

Flow Magnitude

Motion Analysis

Convert (u,v) to motion magnitude

Flow field

Velocity map

9

Temporal Averaging

Signal Processing

Build local baseline

Velocity history

Baseline map

10

Standard Deviation Mapping

Statistics

Model expected local variation

Velocity history

σ map

11

Spatio-Temporal Z-Score

Anomaly Detection

Detect abnormal flow drop

Current + baseline

Z-map

12

Local Nadir Detection

Localization

Find strongest stagnation point

Z-map

Coordinates

13

Threshold Region Detection

Decision Logic

Segment abnormal flow

Z-map

Binary mask

14

Bounding Box Localization

Computer Vision

Draw anomaly region

Binary mask

Bounding box

15

Poiseuille Approximation

Fluid Mechanics

Idealized vessel-flow model

Radius + velocity

Flow profile

16

Wall Shear Approximation

Hemodynamics

Estimate shear rate

Velocity + radius

Shear rate

17

Non-Newtonian Rheology

Biofluid Mechanics

Model shear-dependent viscosity

Shear + parameters

Viscosity

18

Feature Normalization

Machine Learning

Scale model features

Raw features

Normalized features

19

NeuroOcularNet

Neural Regression

Predict physiological targets

Feature vector

Glucose + viscosity

20

ONNX Graph Execution

Edge AI

Fast inference

Tensor

Prediction tensor

21

Persistence Logic

Temporal Analysis

Reject isolated anomalies

Z history

Persistent mask

22

Severity Estimation

Decision Layer

Quantify anomaly strength

Anomaly metrics

Severity score

Algorithm Families

Computer Vision

Cumulative Motion Masking

Farneback Dense Optical Flow

Bounding Box Localization

Signal Processing

Temporal Averaging

Velocity Gradient Analysis

Pulsatility Estimation

Statistics

Standard Deviation Mapping

Z-Score Analysis

Physics

Poiseuille Approximation

Wall Shear Rate

Non-Newtonian Viscosity

Machine Learning

Feature Scaling

NeuroOcularNet

ONNX Inference

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="8-computer-vision-pipeline"></a>

<div align="center">

👁️ 8. Computer Vision Pipeline

How raw retinal frames are prepared before motion and hemodynamic analysis.

</div>

8.1 Frame Acquisition

A retinal video is represented as:

[
I_1, I_2, I_3, \dots, I_T
]

where:

(I_t) = frame at time (t)

(T) = total number of frames.

8.2 Intensity Conversion

Frames may be converted from RGB to a lower-dimensional intensity representation for motion estimation.

Purpose

lower computational cost,

reduced sensitivity to color variation,

improved compatibility with classical optical-flow methods.

8.3 Denoising

Denoising suppresses:

sensor noise,

compression artifacts,

random high-frequency variation.

8.4 Intensity Normalization

A standard normalization form is:

\frac{I-\mu_I}{\sigma_I+\epsilon}
]

where:

(\mu_I) = mean frame intensity,

(\sigma_I) = frame intensity standard deviation,

(\epsilon) = numerical stability term.

8.5 Contrast Enhancement

Contrast enhancement can be applied when vessel boundaries or intravascular structures are difficult to distinguish.

Potential methods include:

histogram normalization,

local contrast enhancement,

CLAHE.

The exact preprocessing configuration should remain fixed during validation to avoid introducing acquisition-dependent bias.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="9-retinal-motion--velocimetry"></a>

<div align="center">

🎞️ 9. Retinal Motion & Velocimetry

How the system estimates microvascular motion from consecutive retinal frames.

</div>

9.1 Cumulative Motion Mask

The frame-difference signal can be written as:

|I_t(x,y)-I_{t-1}(x,y)|
]

A cumulative representation is then:

\sum_{t=2}^{T} M_t(x,y)
]

The motion ROI is:

\begin{cases}
1 & M_{cum}(x,y) > \tau_m \
0 & \text{otherwise}
\end{cases}
]

Why It Is Used

The cumulative mask suppresses mostly static retinal background and highlights repeatedly changing regions.

9.2 Gunnar Farneback Dense Optical Flow

Farneback estimates a dense 2D motion vector:

(u(x,y),v(x,y))
]

where:

(u) = horizontal displacement,

(v) = vertical displacement.

The magnitude is:

\sqrt{u(x,y)^2+v(x,y)^2}
]

Why Farneback?

It was selected because it is:

dense,

computationally efficient,

deterministic,

available in OpenCV,

suitable for frame-to-frame motion analysis.

Main OpenCV Parameters

Parameter

Function

pyr_scale

Pyramid image scale

levels

Number of pyramid levels

winsize

Averaging-window size

iterations

Iterations per level

poly_n

Neighborhood polynomial size

poly_sigma

Gaussian sigma

flags

Optional algorithm flags

9.3 Velocity Conversion

Raw optical flow is expressed in:

pixels / frame

Converted to pixels per second:

v_{px/frame}\times FPS
]

If a spatial calibration coefficient exists:

[
C = \frac{\mu m}{pixel}
]

then:

v_{px/s}\times C
]

Without spatial calibration, optical-flow magnitude remains an image-space motion estimate rather than absolute physiological blood velocity.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="10-hemodynamic-feature-engineering"></a>

<div align="center">

📈 10. Hemodynamic Feature Engineering

How optical-flow measurements are converted into interpretable physiological descriptors.

</div>

Feature Set

Feature

Purpose

Mean Velocity

Average local motion

Velocity STD

Flow heterogeneity

Maximum Velocity

Peak local motion

Temporal Gradient

Acceleration / deceleration

Relative Flow Drop

Reduction from baseline

Stagnation Ratio

Fraction of low-flow pixels

Motion Occupancy

Active-flow area

Pulsatility Index

Temporal flow variation

Shear Rate

Near-wall flow gradient

Anomaly Score

Deviation from normal local behavior

10.1 Mean Velocity

\frac{1}{N}
\sum_{i=1}^{N}v_i
]

10.2 Velocity Standard Deviation

\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(v_i-\bar{v})^2
}
]

10.3 Maximum Velocity

\max(v_1,\dots,v_N)
]

10.4 Temporal Velocity Gradient

v_t-v_{t-1}
]

10.5 Relative Flow Drop

\frac{
v_{baseline}-v_{current}
}{
v_{baseline}+\epsilon
}
]

10.6 Stagnation Ratio

\frac{
N(v<\tau_s)
}{
N_{ROI}
}
]

10.7 Motion Occupancy

\frac{
N_{active}
}{
N_{image}
}
]

10.8 Pulsatility Index

\frac{
v_{max}-v_{min}
}{
v_{mean}+\epsilon
}
]

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="11-biophysical--rheological-modeling"></a>

<div align="center">

🩸 11. Biophysical & Rheological Modeling

The physics layer used to interpret motion in hemodynamic terms.

</div>

11.1 Poiseuille-Type Velocity Profile

v_{max}
\left(
1-\frac{r^2}{R^2}
\right)
]

Symbol

Meaning

(v(r))

Velocity at radial location

(v_{max})

Maximum centerline velocity

(r)

Radial distance

(R)

Vessel radius

Real microvascular blood flow is more complex because blood is non-Newtonian, erythrocytes deform, vessel walls are biological, and capillary dimensions approach cellular scale.

11.2 Wall Shear Rate

General form:

\left.
\frac{\partial v}{\partial r}
\right|_{r=R}
]

Simplified laminar approximation:

[
\dot{\gamma}_{wall}
\approx
\frac{4\bar{v}}{R}
]

11.3 Shear Stress

\eta\dot{\gamma}
]

where:

(\tau) = shear stress,

(\eta) = apparent dynamic viscosity,

(\dot{\gamma}) = shear rate.

11.4 Non-Newtonian Viscosity Model

A generalized Carreau-Yasuda-style relationship is represented as:

\eta_\infty
+
\left(
\eta_0(G)-\eta_\infty
\right)
\left[
1+(\lambda\dot{\gamma})^a
\right]^{\frac{n-1}{a}}
]

Parameter

Meaning

(\eta)

Apparent viscosity

(\eta_0)

Low-shear viscosity

(\eta_\infty)

High-shear viscosity

(\dot{\gamma})

Shear rate

(\lambda)

Time constant

(a)

Transition parameter

(n)

Flow behavior index

(G)

Glucose-related physiological term

The mapping between glucose and rheological behavior must ultimately be calibrated with paired real clinical measurements.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="12-spatio-temporal-anomaly-detection"></a>

<div align="center">

🚨 12. Spatio-Temporal Anomaly Detection

How the system detects local flow deceleration that persists beyond expected variation.

</div>

Detection Logic

Velocity History
      ↓
Local Baseline
      ↓
Local Standard Deviation
      ↓
Z-Score Map
      ↓
Thresholding
      ↓
Persistence Check
      ↓
Nadir Localization
      ↓
Bounding Box + Severity

12.1 Local Baseline

\frac{1}{T_b}
\sum_{t=1}^{T_b}
v(x,y,t)
]

12.2 Local Temporal Variation

\sqrt{
\frac{1}{T_b}
\sum
\left(
v(x,y,t)-\bar{v}_{baseline}(x,y)
\right)^2
}
]

12.3 Z-Score Anomaly Map

\frac{
\bar{v}_{baseline}(x,y)-v(x,y,t)
}{
\sigma_v(x,y)+\epsilon
}
]

Interpretation:

Z-Score Behavior

Interpretation

Near zero

Flow close to local baseline

Moderate positive

Reduced local flow

High positive

Strong local deceleration

12.4 Nadir Localization

\arg\max_{x,y} Z(x,y,t)
]

This gives the strongest local stagnation candidate.

12.5 Persistence Logic

\sum_{t=t_0}^{t_1}
\mathbb{1}
\left[
Z(x,y,t)>\tau_Z
\right]
]

Persistence reduces sensitivity to single-frame artifacts.

Detection Output

The anomaly subsystem may return:

anomaly center,

bounding box,

local flow reduction,

persistence level,

severity estimate.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="13-ai-model-neuroocularnet"></a>

<div align="center">

🤖 13. AI Model — NeuroOcularNet

The lightweight neural regression model used for physiological inference.

</div>

Model Card

Property

Value

Model Name

NeuroOcularNet

Task

Multi-output regression

Input Type

Engineered hemodynamic feature vector

Output Type

Glucose + viscosity estimates

Deployment Format

ONNX

Reported Size

~10.28 KB

Runtime

ONNX Runtime

Deployment Goal

Edge / low-latency inference

Model Role

The model maps normalized features:

[x_1,x_2,\dots,x_n]
]

to:

[
\hat{G},
\hat{\eta}
]
]

where:

(\hat{G}) = estimated glucose,

(\hat{\eta}) = estimated viscosity,

(\theta) = learned model parameters.

Why a Compact Network?

A lightweight network provides:

low latency,

low memory usage,

simpler deployment,

easier ONNX export,

better suitability for edge systems.

Architecture Philosophy

Raw Video
   ↓
Interpretable Features
   ↓
Compact Neural Model
   ↓
Physiological Predictions

This is intentionally different from a fully opaque end-to-end raw-video model.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="14-model-inputs--outputs"></a>

<div align="center">

🔢 14. Model Inputs & Outputs

The exact information passed into and returned from the AI inference stage.

</div>

Candidate Input Feature Vector

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

The inference feature order must exactly match the feature order used during model training.

Output Vector

Output 1 → Estimated Blood Glucose (mg/dL)
Output 2 → Estimated Dynamic Viscosity (mPa·s)

Input / Output Summary

Stage

Data Shape

Raw Video

Sequence of frames

Optical Flow

Dense 2D vector field

Feature Engine

Numerical feature vector

NeuroOcularNet

Normalized tensor

Final Regression Output

Two physiological estimates

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="15-training--normalization-pipeline"></a>

<div align="center">

🧬 15. Training & Normalization Pipeline

How feature data is prepared, optimized, evaluated, and exported.

</div>

Training Flow

Raw / Simulated Samples
        ↓
Feature Construction
        ↓
Feature Matrix X
        ↓
Target Matrix Y
        ↓
Train / Validation Split
        ↓
Feature Scaling
        ↓
NeuroOcularNet Training
        ↓
Loss Optimization
        ↓
Evaluation
        ↓
ONNX Export

Feature Normalization

Standard scaling:

\frac{x-\mu_x}{\sigma_x+\epsilon}
]

Alternative min-max scaling:

\frac{x-x_{min}}{x_{max}-x_{min}}
]

Training Loss

Mean Squared Error:

\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
]

Reported prototype scaled MSE:

0.0421

MSE must always be interpreted in the context of the scaling and normalization used for the target variables.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="16-onnx-edge-deployment"></a>

<div align="center">

⚡ 16. ONNX Edge Deployment

How the trained model is packaged and executed for lightweight inference.

</div>

Deployment Files

NeuroOcularNet.onnx
NeuroOcularNet.onnx.data

Inference Flow

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

Why ONNX?

portable model format,

fast CPU inference,

cross-framework compatibility,

compact runtime,

edge-device suitability.

Target Performance

Target model inference latency: < 5 ms

Actual latency depends on:

CPU,

ONNX Runtime build,

thread configuration,

preprocessing cost,

deployment hardware.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="17-application-layer"></a>

<div align="center">

🖥️ 17. Application Layer

How the analytical pipeline is exposed through the user-facing Streamlit application.

</div>

Main Responsibilities

The Streamlit application is responsible for:

video upload and loading,

frame preview,

preprocessing execution,

optical-flow processing,

feature extraction,

ONNX model loading,

prediction,

anomaly visualization,

clinical-style output cards.

Interface Files

app.py
style.css

UI Output Components

Component

Purpose

Video Preview

Display uploaded retinal sequence

Flow Visualization

Show motion / velocity information

Glucose Card

Display model glucose estimate

Viscosity Card

Display inferred viscosity

Anomaly Alert

Highlight abnormal local flow

Bounding Box

Localize the detected region

Severity Indicator

Summarize anomaly magnitude

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="18-technology-stack"></a>

<div align="center">

🧰 18. Technology Stack

Languages, libraries, runtimes, and mathematical components used by the project.

</div>

Programming

Technology

Role

Python

Main application and analysis language

Computer Vision & Numerical Processing

Technology

Role

OpenCV

Video decoding, preprocessing, optical flow, visualization

NumPy

Numerical operations, arrays, statistics, feature calculation

AI & Inference

Technology

Role

NeuroOcularNet

Physiological regression model

ONNX

Portable model representation

ONNX Runtime

Fast inference engine

Web Layer

Technology

Role

Streamlit

Interactive application

CSS

Visual styling

Mathematical Components

Component

Role

Farneback Optical Flow

Dense motion estimation

Cumulative Motion Mask

Motion ROI

Temporal Baseline

Expected flow behavior

Z-Score

Anomaly detection

Poiseuille Approximation

Idealized flow model

Wall Shear Rate

Hemodynamic descriptor

Non-Newtonian Rheology

Viscosity modeling

MSE

Regression loss

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="19-repository-structure"></a>

<div align="center">

📂 19. Repository Structure

How the project files are organized and what each file is responsible for.

</div>

NeuroOcular-AI/
│
├── app.py
│   ├── Streamlit UI
│   ├── video loading
│   ├── preprocessing
│   ├── optical flow
│   ├── feature extraction
│   ├── ONNX inference
│   └── visualization
│
├── NeuroOcularNet.onnx
│   └── optimized inference model
│
├── NeuroOcularNet.onnx.data
│   └── external ONNX tensor data
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

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="20-sample-clinical-scenarios"></a>

<div align="center">

🎥 20. Sample Clinical Scenarios

The benchmark video scenarios included in the current repository.

</div>

File

Scenario

Purpose

01_normal_euglycemic_flow.avi

Normal flow

Baseline

02_mild_hyperglycemia.avi

Mild hyperglycemia

Mild rheological shift

03_acute_hyperglycemia.avi

Acute hyperglycemia

Stronger hemodynamic shift

04_early_platelet_cluster.avi

Early aggregation

Early local deceleration

05_focal_micro_thrombus.avi

Focal micro-thrombus

Local anomaly detection

06_severe_luminal_stenosis.avi

Severe stenosis

High flow restriction

07_pulsatile_shear_stress.avi

Pulsatile flow

Temporal flow analysis

08_sluggish_microcirculation.avi

Sluggish flow

Global reduction

09_transient_ischemic_event.avi

Transient event

Temporary anomaly

10_near_complete_occlusion.avi

Near occlusion

Extreme restriction

These should be labeled as simulation / benchmark scenarios unless they originate from a validated clinical dataset.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="21-experimental-results"></a>

<div align="center">

📊 21. Experimental Results

Prototype-level outputs reported during development.

</div>

Reported Results

Metric

Prototype Value

Scaled MSE

0.0421

Example viscosity

3.42 – 3.46 mPa·s

Example glucose output

118.3 – 143.8 mg/dL

Example anomaly core

(128, 128)

Example pre-occlusive severity

33%

ONNX model size

10.28 KB

Target model inference

< 5 ms

Interpretation

These results demonstrate the technical ability of the prototype to:

process retinal video,

estimate optical motion,

derive hemodynamic features,

execute neural inference,

generate numerical outputs,

detect spatial flow anomalies,

and localize suspicious regions.

These values do not establish clinical diagnostic accuracy.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="22-performance-metrics"></a>

<div align="center">

📐 22. Performance Metrics

Metrics recommended for future technical and clinical evaluation.

</div>

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

Recommended metrics include:

sensitivity,

specificity,

precision,

recall,

F1-score,

ROC-AUC,

false-positive rate.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="23-installation"></a>

<div align="center">

💻 23. Installation

Environment setup for local development and testing.

</div>

Requirements

Recommended Python versions:

Python 3.9
Python 3.10
Python 3.11

Check your version:

python --version

Clone the Repository

git clone https://github.com/yousefosamaahmed/NeuroOcular-AI.git
cd NeuroOcular-AI

Create a Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="24-running-the-application"></a>

<div align="center">

▶️ 24. Running the Application

How to start the Streamlit interface locally.

</div>

Run:

streamlit run app.py

Typical local address:

http://localhost:8501

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="25-cloud-deployment"></a>

<div align="center">

☁️ 25. Cloud Deployment

How to deploy the application using Streamlit Community Cloud.

</div>

Deployment Settings

Repository:
yousefosamaahmed/NeuroOcular-AI

Branch:
main

Main file:
app.py

Required Files

app.py
requirements.txt
NeuroOcularNet.onnx
style.css

If required by the exported model:

NeuroOcularNet.onnx.data

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="26-input-requirements"></a>

<div align="center">

📥 26. Input Requirements

Recommended characteristics for retinal video input.

</div>

Parameter

Recommendation

Format

AVI / MP4

Content

Retinal microvascular video

Frame Rate

Constant

Focus

Sharp vessel structures

Exposure

Stable

Compression

Low / moderate

Eye Motion

Minimal

Camera Motion

Minimal

Duration

Enough frames for temporal baseline

Why Input Quality Matters

Poor acquisition can significantly affect:

optical flow,

vessel motion estimation,

shear-related features,

anomaly maps,

and final model output.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="27-output-interpretation"></a>

<div align="center">

📤 27. Output Interpretation

How to read each value produced by the application.

</div>

Glucose Estimate

Estimated Glucose
126 mg/dL

This is a model prediction, not a direct chemical measurement.

Viscosity Estimate

Estimated Dynamic Viscosity
3.44 mPa·s

This is an inferred model output.

Flow Anomaly

Potential Localized Flow Anomaly

Core:
X = 128
Y = 128

Estimated Severity:
33%

This indicates statistically abnormal local flow behavior.

It does not independently confirm a thrombus.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="28-failure-modes--limitations"></a>

<div align="center">

⚠️ 28. Failure Modes & Limitations

The main technical and scientific limitations of the current research prototype.

</div>

28.1 Optical Flow Is Not Direct Blood Velocity

Farneback returns image displacement.

pixel displacement ≠ mm/s

Absolute velocity requires:

frame rate,

spatial calibration,

retinal magnification,

vessel geometry.

28.2 Glucose Is Not Uniquely Determined by Flow

Microvascular flow is influenced by many confounders:

blood pressure,

hematocrit,

hydration,

temperature,

vessel diameter,

autonomic tone,

medication,

cardiovascular condition,

retinal disease.

28.3 Eye & Camera Motion

Potential artifacts include:

eye motion,

camera shake,

blinking,

focus drift.

28.4 Illumination Variation

Brightness changes can create false apparent motion.

28.5 Incorrect Motion ROI

If nonvascular structures enter the ROI, downstream hemodynamic features may become biased.

28.6 Synthetic Data Generalization

If training data or benchmark videos are synthetic, performance may not generalize to real patients.

This is one of the most important scientific limitations of the current prototype.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="29-clinical-validation-roadmap"></a>

<div align="center">

🔬 29. Clinical Validation Roadmap

The staged pathway required to move from prototype to clinically meaningful evidence.

</div>

Phase 1 — Technical Validation

Validate:

optical-flow recovery,

synthetic displacement accuracy,

frame-rate sensitivity,

noise robustness,

motion stabilization.

Phase 2 — Physiological Calibration

Collect synchronized:

Retinal video
Venous glucose
Capillary glucose
Hematocrit
Blood pressure
Heart rate
Viscosity measurement
Clinical status

Phase 3 — Prospective Human Study

Evaluate:

MAE,

RMSE,

MARD,

Bland-Altman agreement,

Clarke Error Grid,

Parkes Error Grid,

sensitivity,

specificity,

ROC-AUC.

Phase 4 — External Validation

Test across:

multiple hospitals,

multiple retinal cameras,

age groups,

ethnic groups,

diabetic and non-diabetic cohorts,

vascular comorbidities.

Phase 5 — Regulatory Development

A clinical product would require:

defined intended use,

quality-management controls,

risk management,

software lifecycle documentation,

appropriate medical-device regulatory review.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="30-future-development"></a>

<div align="center">

🚀 30. Future Development

Planned directions for improving vision accuracy, physics fidelity, AI robustness, and deployment.

</div>

Computer Vision

retinal vessel segmentation,

U-Net segmentation,

transformer-based segmentation,

image registration,

retinal stabilization,

sub-pixel motion estimation,

deep optical flow.

Hemodynamics

vessel-diameter estimation,

vessel-centerline extraction,

branch-specific flow analysis,

calibrated absolute velocity,

vessel-specific shear estimation,

pulsatility analysis.

AI

uncertainty-aware regression,

temporal neural networks,

LSTM / GRU modeling,

temporal transformers,

multimodal fusion,

patient-specific calibration.

Anomaly Detection

adaptive thresholds,

confidence scoring,

false-positive suppression,

temporal persistence models,

vessel-aware anomaly segmentation.

Deployment

mobile inference,

embedded edge devices,

quantized ONNX models,

hardware acceleration,

direct camera integration.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="31-security--privacy"></a>

<div align="center">

🔐 31. Security & Privacy

Data-protection requirements for any future clinical implementation.

</div>

A production clinical system should include:

patient de-identification,

encryption at rest,

encryption in transit,

role-based access control,

authentication,

audit logs,

secure model storage,

retention policies,

consent management.

Personally identifiable patient data should never be committed to a public repository.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="32-intellectual-property"></a>

<div align="center">

⚖️ 32. Intellectual Property

Potentially protectable technical components of the platform.

</div>

Potential IP-relevant areas include:

retinal hemodynamic inference workflow,

physics-informed feature construction,

dual glucose / viscosity regression,

spatio-temporal flow anomaly scoring,

edge-deployed retinal inference,

real-time microvascular risk visualization.

Patentability or patent status should only be stated after review by a qualified intellectual-property professional.

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="33-license"></a>

<div align="center">

📜 33. License

Project licensing information.

</div>

Distributed under the MIT License unless otherwise specified.

See:

LICENSE

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="34-citation"></a>

<div align="center">

🧾 34. Citation

Suggested citation for academic or technical use.

</div>

@software{neuroocular_ai_2026,
  title  = {NeuroOcular AI: Physics-Informed Retinal Hemodynamics Platform},
  author = {Yousef Osama Ahmed},
  year   = {2026},
  url    = {https://github.com/yousefosamaahmed/NeuroOcular-AI}
}

<p align="right"><a href="#top">↑ Back to top</a></p>

<a id="35-author"></a>

<div align="center">

👤 35. Author

Project ownership and repository information.

</div>

Yousef Osama Ahmed

Project

NeuroOcular AI
Physics-Informed Retinal Hemodynamics Platform

Repository

https://github.com/yousefosamaahmed/NeuroOcular-AI

<p align="right"><a href="#top">↑ Back to top</a></p>

<div align="center">

🔍 Technical Summary

End-to-End System at a Glance

</div>

RETINAL VIDEO
    │
    ├── Preprocessing
    │     ├── Intensity Conversion
    │     ├── Denoising
    │     ├── Normalization
    │     └── Contrast Enhancement
    │
    ├── Motion ROI
    │     └── Cumulative Motion Mask
    │
    ├── Dense Optical Flow
    │     └── Gunnar Farneback
    │
    ├── Hemodynamic Feature Engine
    │     ├── Mean Velocity
    │     ├── Velocity STD
    │     ├── Maximum Velocity
    │     ├── Temporal Gradient
    │     ├── Flow Drop
    │     ├── Stagnation Ratio
    │     ├── Shear Rate
    │     └── Pulsatility
    │
    ├── Physics Layer
    │     ├── Poiseuille Approximation
    │     ├── Wall Shear Rate
    │     ├── Shear Stress
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
    │     ├── Z-Score
    │     ├── Persistence Logic
    │     ├── Nadir Detection
    │     ├── Bounding Box
    │     └── Severity Score
    │
    └── Streamlit Dashboard

<div align="center">

🩺 NeuroOcular AI

Computer Vision • Hemodynamics • Physics-Informed AI • Edge Deployment

Research prototype for next-generation retinal microvascular analysis.

</div>
