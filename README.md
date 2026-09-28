🩺 NeuroOcular AI

Physics-Informed Retinal Hemodynamics Platform









A research-oriented, physics-informed computer vision platform for analyzing retinal microvascular flow and exploring non-invasive estimation of physiological biomarkers and pre-occlusive flow anomalies.

⚠️ Research Disclaimer

NeuroOcular AI is an experimental research prototype and is not a certified medical device.

The platform is intended for academic research, prototyping, simulation, and technical demonstration. Outputs such as glucose estimates, viscosity estimates, or thrombus-risk indicators must not be used for diagnosis, treatment decisions, emergency triage, or replacement of validated clinical measurements.

Clinical deployment would require prospective human studies, calibration against laboratory reference methods, regulatory review, safety validation, demographic robustness testing, and independent external validation.

📑 Table of Contents

Project Overview

Problem Statement

Project Objectives

Core Innovation

System Architecture

Computational Pipeline

Biophysical Formulation

Machine Learning Architecture

Micro-Thrombus Detection Logic

Experimental Results

Technology Stack

Repository Structure

Installation

Running the Application

Cloud Deployment

Input Requirements

Output Interpretation

Validation Roadmap

Known Limitations

Future Development

Intellectual Property

License

Contributing

Citation

Contact

📌 Project Overview

NeuroOcular AI is a physics-informed medical computer vision platform designed to analyze retinal microvascular video sequences.

The system combines:

retinal video processing,

dense optical flow,

microvascular velocimetry,

hemodynamic feature extraction,

physics-informed inference,

lightweight neural networks,

ONNX edge deployment,

and spatio-temporal anomaly localization.

The current prototype investigates whether changes in retinal microvascular flow can be used as indirect physiological markers for:

Blood glucose estimation

Dynamic blood viscosity estimation

Localized detection of abnormal flow deceleration

Identification of potential pre-occlusive microvascular events

The platform is designed around the concept that retinal vessels provide a unique optical window into human microcirculation.

🚨 Problem Statement

Many physiological conditions affect blood rheology and microvascular circulation before major macroscopic complications become visible.

Two particularly important examples are:

1. Blood Glucose Monitoring

Traditional glucose monitoring generally relies on:

finger-stick capillary blood sampling,

venous laboratory testing,

or Continuous Glucose Monitoring (CGM).

CGM devices measure glucose primarily in interstitial fluid, rather than directly in circulating blood.

During rapid glucose changes, physiological lag may occur between blood and interstitial glucose measurements.

A non-invasive method capable of extracting useful glucose-related signals directly from vascular dynamics could therefore represent an important research direction.

2. Microvascular Thrombotic Events

Many conventional imaging modalities are optimized for detecting:

established vascular occlusion,

significant stenosis,

macroscopic thrombi,

or downstream perfusion abnormalities.

NeuroOcular AI explores a different hypothesis:

Can localized microvascular flow deceleration be detected before complete vessel obstruction occurs?

The system therefore analyzes spatial and temporal velocity changes rather than relying exclusively on static visual evidence of an already-formed clot.

🎯 Project Objectives

The platform has five primary technical objectives:

1. Extract Retinal Microvascular Motion

Detect vascular regions and estimate frame-to-frame motion inside retinal capillaries.

2. Estimate Relative Blood Flow Velocity

Use dense optical-flow algorithms to approximate erythrocyte-related motion patterns.

3. Compute Hemodynamic Features

Derive parameters such as:

mean flow velocity,

local velocity gradients,

estimated wall shear rate,

flow variability,

stagnation indices,

and regional perfusion asymmetry.

4. Perform Physics-Informed Inference

Feed extracted hemodynamic features into a compact neural network trained to estimate physiological outputs.

5. Detect Abnormal Local Flow Deceleration

Generate spatial anomaly maps capable of highlighting regions where flow deviates significantly from baseline behavior.

💡 Core Innovation

The project does not rely only on raw RGB intensity prediction.

Instead, it combines computer vision + fluid mechanics + machine learning.

The main concept is:

Retinal Video
     ↓
Motion Extraction
     ↓
Flow Velocity Estimation
     ↓
Hemodynamic Features
     ↓
Physics-Informed Inference
     ↓
Physiological Estimates + Flow Anomaly Detection

This structure improves interpretability compared with a purely black-box image classifier.

🧠 System Architecture

                                ┌─────────────────────────────┐
                                │  Retinal Video Sequence    │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │ Frame Preprocessing         │
                                │ • Resize                    │
                                │ • Denoising                 │
                                │ • Contrast normalization    │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │ Motion-Based ROI Extraction │
                                │ Cumulative Motion Mask      │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │ Dense Optical Flow          │
                                │ Farneback Velocimetry       │
                                └──────────────┬──────────────┘
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                         ▼                                           ▼
          ┌─────────────────────────────┐             ┌─────────────────────────────┐
          │ Hemodynamic Feature Engine  │             │ Spatio-Temporal Analyzer    │
          │ • Mean velocity             │             │ • Local velocity history    │
          │ • Flow variance             │             │ • Z-score anomaly map       │
          │ • Shear-related features    │             │ • Stagnation localization   │
          └──────────────┬──────────────┘             └──────────────┬──────────────┘
                         │                                           │
                         ▼                                           ▼
          ┌─────────────────────────────┐             ┌─────────────────────────────┐
          │ NeuroOcularNet             │             │ Flow Anomaly Alert          │
          │ Physics-Informed NN        │             │ • Bounding region           │
          │ ONNX Edge Runtime          │             │ • Severity indicator        │
          └──────────────┬──────────────┘             │ • Spatial coordinates       │
                         │                            └─────────────────────────────┘
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
        ┌────────────────┐  ┌─────────────────┐
        │ Blood Viscosity│  │ Glucose Estimate│
        │ Estimate       │  │                 │
        └────────────────┘  └─────────────────┘

⚙️ Computational Pipeline

Stage 1 — Video Acquisition

Input is a retinal microvascular video sequence.

The prototype assumes that the video contains sufficiently visible vascular structures and measurable temporal motion.

Stage 2 — Frame Preprocessing

Possible preprocessing operations include:

grayscale conversion,

Gaussian denoising,

local contrast enhancement,

normalization,

frame resizing,

vessel-oriented enhancement.

The goal is to increase signal quality before motion estimation.

Stage 3 — Motion-Based Region of Interest

A cumulative motion mask is generated to suppress static retinal structures and focus processing on regions containing measurable motion.

Conceptually:

Static Background
      ↓
Suppressed

Moving Blood-Flow Regions
      ↓
Retained as ROI

Stage 4 — Dense Optical Flow

The current pipeline uses Farneback dense optical flow to estimate pixel-wise motion vectors between consecutive frames.

For every pixel:

[
\vec{F}(x,y) = (u(x,y), v(x,y))
]

The motion magnitude is:

[
M(x,y) = \sqrt{u(x,y)^2 + v(x,y)^2}
]

This provides a computational approximation of local motion velocity.

Stage 5 — Hemodynamic Feature Extraction

The extracted motion field is converted into physiological and statistical descriptors.

Example features include:

Mean motion velocity

Maximum local velocity

Standard deviation of velocity

Regional flow variance

Temporal velocity gradient

Relative shear estimate

Stagnation ratio

Vessel occupancy

Pulsatility-related features

Spatio-temporal anomaly score

📐 Biophysical Formulation

1. Microvascular Velocity Profile

A simplified Poiseuille-type approximation can be expressed as:

[
v(r)=v_{\max}\left(1-\frac{r^2}{R^2}\right)
]

Where:

(v(r)) = local fluid velocity

(v_{\max}) = centerline velocity

(r) = radial distance from vessel center

(R) = vessel radius

This approximation is used as a conceptual reference. Real microvascular blood flow is more complex because blood is non-Newtonian and capillary-scale flow depends strongly on cell deformation, vessel diameter, hematocrit, and endothelial interactions.

2. Wall Shear Rate

The wall shear rate can be approximated from the velocity gradient:

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

where (\bar{v}) represents mean flow velocity.

3. Non-Newtonian Viscosity Model

The project conceptually supports a generalized shear-dependent rheological formulation:

\eta_{\infty}
+
\left(\eta_0(G)-\eta_{\infty}\right)
\left[
1+(\lambda\dot{\gamma})^a
\right]^{\frac{n-1}{a}}
]

Where:

(\eta) = apparent blood viscosity

(\dot{\gamma}) = shear rate

(G) = glucose-related physiological state

(\eta_0) = low-shear viscosity term

(\eta_{\infty}) = high-shear limiting viscosity

(\lambda) = time constant

(a) = transition parameter

(n) = flow behavior index

The exact mapping between glucose and rheological variables must be empirically calibrated using paired clinical measurements.

4. Spatio-Temporal Flow Anomaly Score

A local stagnation score is calculated relative to baseline behavior:

\frac{
\bar{v}_{baseline}(x,y)-v(x,y,t)
}{
\sigma_v(x,y)+\epsilon
}
]

Where:

(\bar{v}_{baseline}) = baseline local flow velocity

(v(x,y,t)) = current estimated velocity

(\sigma_v) = historical local velocity variation

(\epsilon) = numerical stability constant

Large positive values indicate stronger-than-expected local flow reduction.

🤖 Machine Learning Architecture

The current implementation uses a lightweight neural network called:

NeuroOcularNet

Its purpose is to map extracted hemodynamic features to physiological estimates.

Example input vector:

[
    mean_velocity,
    velocity_std,
    shear_rate,
    vessel_density,
    stagnation_ratio,
    temporal_gradient,
    pulsatility_index,
    anomaly_score
]

Example outputs:

[
    estimated_glucose_mg_dl,
    estimated_viscosity_mPa_s
]

The trained model is exported to:

NeuroOcularNet.onnx

Benefits of ONNX deployment include:

fast inference,

cross-platform execution,

compact runtime,

deployment on edge devices,

reduced Python dependency during inference.

🩸 Micro-Thrombus Detection Logic

The prototype does not claim to visually identify platelet composition directly.

Instead, it attempts to detect a hemodynamic signature consistent with localized flow restriction.

The detection process is:

Baseline Flow
     ↓
Local Velocity Monitoring
     ↓
Persistent Deceleration
     ↓
Spatio-Temporal Z-Score
     ↓
Anomaly Localization
     ↓
Potential Pre-Occlusive Flow Alert

The anomaly engine can return:

anomaly coordinates,

bounding box,

regional flow reduction,

estimated severity,

temporal persistence.

📊 Experimental Results

The results below describe prototype or simulation-stage measurements and should not be interpreted as clinical accuracy claims unless supported by a documented validation dataset.

Metric

Prototype Result

Training / scaled MSE

0.0421

Example viscosity output

3.42 – 3.46 mPa·s

Example glucose output

118.3 – 143.8 mg/dL

Example anomaly coordinate

(128, 128)

Example pre-occlusive severity

33%

ONNX model size

~10.28 KB

Target edge inference latency

< 5 ms

Important Interpretation

These values demonstrate that the software pipeline is technically capable of:

extracting motion features,

generating physiological predictions,

localizing flow abnormalities,

and running lightweight inference.

They do not, by themselves, demonstrate clinical sensitivity, specificity, glucose measurement accuracy, or validated thrombus detection.

🧰 Technology Stack

Core Language

Python 3.9+

Computer Vision

OpenCV

NumPy

Machine Learning

PyTorch / TensorFlow depending on training implementation

ONNX

ONNX Runtime

Data Processing

Pandas

NumPy

Visualization

Matplotlib

Plotly

User Interface

Streamlit

Deployment

Streamlit Community Cloud

Local CPU inference

ONNX Runtime

📂 Repository Structure

NeuroOcular-AI/
│
├── app.py
│   └── Main Streamlit application
│
├── NeuroOcularNet.onnx
│   └── Optimized inference model
│
├── NeuroOcularNet.onnx.data
│   └── External ONNX tensor data if required
│
├── requirements.txt
│   └── Python dependencies
│
├── style.css
│   └── Custom clinical dashboard styling
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

💻 Installation

Prerequisites

Recommended environment:

Python 3.9
Python 3.10
Python 3.11

Check your Python version:

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

▶️ Running the Application

Start the Streamlit interface:

streamlit run app.py

Streamlit should automatically open the application in your default browser.

Typical local address:

http://localhost:8501

☁️ Cloud Deployment

Streamlit Community Cloud

Push the repository to GitHub.

Sign in to Streamlit Community Cloud.

Select Create app.

Choose:

Repository:
yousefosamaahmed/NeuroOcular-AI

Branch:
main

Main file:
app.py

Deploy the application.

Deployment Checklist

Make sure the repository contains:

app.py
requirements.txt
NeuroOcularNet.onnx
style.css

If large model files are added later, consider:

Git LFS,

Hugging Face Hub,

cloud object storage,

or runtime model download.

🎥 Input Requirements

Recommended input characteristics:

Parameter

Recommendation

Format

AVI / MP4

Content

Retinal microvascular video

Motion

Stable acquisition

Focus

Clearly visible vessel boundaries

Compression

Low-to-moderate

Frame rate

Consistent

Illumination

Stable

Camera motion

Minimal

Poor acquisition quality can strongly affect optical-flow estimation.

📤 Output Interpretation

The application may display:

Estimated Glucose

Estimated Blood Glucose: 126 mg/dL

This value represents a model prediction, not a laboratory measurement.

Estimated Dynamic Viscosity

Estimated Blood Viscosity: 3.44 mPa·s

This is an inferred value derived from the implemented feature-to-output model.

Flow Anomaly

Potential Localized Flow Anomaly Detected
Coordinates: (128, 128)
Severity: 33%

The detection represents a computational flow abnormality and must not automatically be interpreted as a confirmed thrombus.

🧪 Validation Roadmap

A clinically credible future validation protocol should include:

Phase 1 — Technical Validation

repeatability testing,

optical-flow accuracy benchmarking,

synthetic motion validation,

sensitivity to noise,

camera-motion robustness,

frame-rate sensitivity.

Phase 2 — Physiological Calibration

Collect synchronized:

retinal microvascular video,

venous glucose,

capillary glucose,

hematocrit,

blood viscosity,

blood pressure,

heart rate.

Phase 3 — Prospective Clinical Study

Evaluate performance using:

MAE,

RMSE,

MARD for glucose-related estimation,

Bland-Altman analysis,

Clarke or Parkes Error Grid when appropriate,

sensitivity,

specificity,

precision,

recall,

ROC-AUC for anomaly detection.

Phase 4 — External Validation

Test on independent:

hospitals,

cameras,

patient populations,

age groups,

ethnic groups,

diabetic states,

vascular conditions.

Phase 5 — Regulatory Validation

Potential future pathways would depend on intended medical use and jurisdiction.

⚠️ Known Limitations

The current prototype has several important limitations.

Optical Flow Is Not Absolute Blood Velocity

Pixel displacement must be calibrated using:

retinal image scale,

vessel diameter,

magnification,

acquisition frame rate.

Without calibration, velocity is fundamentally expressed in image-space units such as:

pixels / frame

rather than absolute physiological units.

Blood Glucose Is Not Uniquely Determined by Flow

Microvascular hemodynamics can also be affected by:

blood pressure,

hematocrit,

temperature,

hydration,

vessel diameter,

autonomic tone,

medications,

cardiovascular disease,

diabetes-related vascular remodeling.

Therefore, glucose prediction requires careful multivariable calibration.

Motion Artifacts

Eye motion, camera vibration, blinking, and focus changes can produce false flow estimates.

Vessel Segmentation

Incorrect vessel localization may significantly distort:

velocity estimates,

shear calculations,

anomaly maps.

Clinical Ground Truth

Reliable validation requires synchronized reference measurements acquired from real participants.

🚀 Future Development

Planned or recommended future improvements include:

vessel segmentation using U-Net or transformer architectures,

retinal image stabilization,

sub-pixel motion estimation,

deep optical flow,

vessel diameter estimation,

real-world velocity calibration,

pulsatility analysis,

multi-frame temporal transformers,

uncertainty estimation,

multimodal physiological inputs,

patient-specific calibration,

explainable AI dashboards,

longitudinal monitoring,

mobile/edge inference,

regulatory-grade data logging.

🔐 Data Privacy & Security

Future clinical versions should include:

patient de-identification,

encrypted data storage,

encrypted transmission,

access control,

audit logging,

secure model deployment,

retention policies,

compliance with applicable medical-data regulations.

No personally identifiable clinical data should be committed to the public repository.

🧠 Research Philosophy

NeuroOcular AI is built around a simple principle:

Do not ask a neural network to discover physics that can already be represented explicitly.

The platform therefore uses machine learning to complement — rather than replace — interpretable hemodynamic modeling.

⚖️ Intellectual Property

Potentially protectable technical components may include:

physics-informed hemodynamic feature pipelines,

multimodal glucose-flow inference architecture,

localized pre-occlusive flow anomaly scoring,

real-time retinal microvascular risk mapping,

compact edge inference workflows.

Any statement regarding patentability or patent status should be reviewed by a qualified intellectual-property professional.

If a provisional or PCT application is actually filed, replace this section with the corresponding verified filing information.

📜 License

This project is distributed under the MIT License unless otherwise specified.

See:

LICENSE

for full license terms.

🤝 Contributing

Contributions are welcome for:

retinal image processing,

optical flow,

microvascular modeling,

biomedical signal processing,

AI calibration,

validation methodology,

visualization,

Streamlit UI,

ONNX optimization.

Recommended workflow:

git checkout -b feature/your-feature
git commit -m "Add: your feature"
git push origin feature/your-feature

Then submit a Pull Request.

🧾 Citation

If you use this repository in academic work, you may cite it provisionally as:

@software{neuroocular_ai_2026,
  title  = {NeuroOcular AI: Physics-Informed Retinal Hemodynamics Platform},
  author = {Yousef Osama Ahmed},
  year   = {2026},
  url    = {https://github.com/yousefosamaahmed/NeuroOcular-AI}
}

Update the citation if the project is later associated with a paper, DOI, institution, or research group.

📬 Contact

For research collaboration, technical discussion, or academic inquiries, use the GitHub repository issue tracker or the project owner's preferred professional contact channel.

⭐ Support the Project

If you find the project useful for research or education:

Star the repository

Fork the project

Open an issue

Submit improvements

Share experimental validation results

<div align="center">

NeuroOcular AI

Computer Vision × Hemodynamics × Edge AI

Research prototype for next-generation retinal microvascular analysis.

</div>
