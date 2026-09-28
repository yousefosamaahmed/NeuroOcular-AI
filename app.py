
import os
import json
import tempfile
from datetime import datetime

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import onnxruntime as ort
except Exception:
    ort = None


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroOcular AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "style.css")
DEFAULT_MODEL = "/content/NeuroOcularNet.onnx"
DEFAULT_VIDEO = "/content/real_retinal_flow.avi"


def load_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


# ============================================================
# HELPERS
# ============================================================

def save_upload(uploaded_file, prefix):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"
    path = os.path.join("/content", f"{prefix}{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


@st.cache_resource(show_spinner=False)
def load_onnx_model(path):
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def get_model_path():
    return st.session_state.get("model_path", DEFAULT_MODEL)


def get_video_path():
    return st.session_state.get("video_path", DEFAULT_VIDEO)


def hero():
    st.markdown(
        """
        <section class="hero">
          <div class="hero-copy">
            <div class="eyebrow"><span class="pulse"></span> MEDICAL VISION RESEARCH PLATFORM</div>
            <h1>NeuroOcular <span>AI</span></h1>
            <h2>منصة ذكية لتحليل ديناميكا الأوعية الدقيقة بالشبكية</h2>
            <p>
              واجهة بحثية تجمع معالجة الفيديو، Optical Flow، واستخراج الخصائص
              مع استدلال ONNX في Dashboard واحد واضح وقابل للعرض.
            </p>
            <div class="chip-row">
              <span class="chip">Retinal Imaging</span>
              <span class="chip">Optical Flow</span>
              <span class="chip">ONNX Runtime</span>
              <span class="chip">Research Analytics</span>
            </div>
          </div>

          <div class="retina-wrap" aria-hidden="true">
            <svg viewBox="0 0 320 320" class="retina-svg">
              <defs>
                <radialGradient id="iris" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#dffaff"/>
                  <stop offset="13%" stop-color="#66e8ff"/>
                  <stop offset="28%" stop-color="#0ea5e9"/>
                  <stop offset="44%" stop-color="#0f5b78"/>
                  <stop offset="60%" stop-color="#06101c"/>
                  <stop offset="100%" stop-color="#02050a"/>
                </radialGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="4" result="b"/>
                  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>
              <circle cx="160" cy="160" r="118" fill="none" stroke="#38bdf8" stroke-opacity=".20" stroke-width="2"/>
              <circle cx="160" cy="160" r="96" fill="url(#iris)" stroke="#67e8f9" stroke-opacity=".35" stroke-width="2"/>
              <circle cx="160" cy="160" r="31" fill="#020617"/>
              <circle cx="160" cy="160" r="12" fill="#0b1220"/>
              <circle cx="147" cy="142" r="10" fill="#ffffff" opacity=".7"/>
              <path d="M40 160 H280" stroke="#22d3ee" stroke-width="2" stroke-opacity=".85" filter="url(#glow)"/>
              <circle cx="160" cy="160" r="135" fill="none" stroke="#818cf8" stroke-opacity=".18" stroke-dasharray="8 11"/>
            </svg>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker, title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-kicker">{kicker}</div>
          <div class="section-title">{title}</div>
          <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(icon, title, text):
    st.markdown(
        f"""
        <div class="info-card">
          <div class="icon-box">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyze_video(video_path, model_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    frames = []
    source_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        source_frames += 1

        # Sample every second frame for smoother Colab performance.
        if source_frames % 2:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        if w > 720:
            scale = 720.0 / w
            gray = cv2.resize(gray, (720, int(h * scale)))

        frames.append(gray)

        if len(frames) >= 120:
            break

    cap.release()

    if len(frames) < 5:
        raise RuntimeError("The video needs at least 5 usable frames.")

    speeds = []
    gradients = []

    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i],
            frames[i + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        gy, gx = np.gradient(magnitude)
        shear = np.sqrt(gx ** 2 + gy ** 2)

        speeds.append(float(np.mean(magnitude)))
        gradients.append(float(np.mean(shear)))

    mean_v = float(np.nan_to_num(np.mean(speeds)))
    mean_sr = float(np.nan_to_num(np.mean(gradients)))

    # These transforms MUST match the transforms used when training the model.
    norm_v = (mean_v - 2.0) / 0.8
    norm_sr = (mean_sr - 0.15) / 0.05
    x = np.array([[norm_v, norm_sr]], dtype=np.float32)

    session = load_onnx_model(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})

    flattened = []
    for output in outputs:
        flattened.extend(np.asarray(output).reshape(-1).tolist())

    if len(flattened) < 2:
        raise RuntimeError("The ONNX model must return at least two numerical outputs.")

    raw_1 = float(flattened[0])
    raw_2 = float(flattened[1])

    # Research/demo mapping. Replace with your real inverse transforms.
    viscosity = (raw_1 * 0.5) + 3.8
    glucose = (raw_2 * 25.0) + 120.0

    # Research/demo composite index, not a validated clinical score.
    risk = int(
        np.clip(
            ((viscosity / 4.0) * 45.0)
            + ((0.15 / (mean_sr + 0.01)) * 35.0),
            0,
            100,
        )
    )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frames": frames,
        "speeds": speeds,
        "gradients": gradients,
        "mean_speed": mean_v,
        "mean_gradient": mean_sr,
        "viscosity": viscosity,
        "glucose": glucose,
        "risk": risk,
        "frame_count": len(frames),
        "source_frame_count": source_frames,
    }


def flow_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=data["speeds"],
            mode="lines",
            name="Optical Flow",
            line=dict(color="#22D3EE", width=3),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.06)",
        )
    )

    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,27,0.65)",
        font=dict(color="#CBD5E1"),
        margin=dict(l=35, r=20, t=25, b=40),
        xaxis=dict(title="Frame index", showgrid=False),
        yaxis=dict(
            title="Optical-flow magnitude",
            gridcolor="rgba(148,163,184,.09)",
        ),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def gradient_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=data["gradients"],
            mode="lines",
            name="Spatial Gradient",
            line=dict(color="#818CF8", width=3),
        )
    )

    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,27,0.65)",
        font=dict(color="#CBD5E1"),
        margin=dict(l=35, r=20, t=25, b=40),
        xaxis=dict(title="Frame index", showgrid=False),
        yaxis=dict(
            title="Gradient magnitude",
            gridcolor="rgba(148,163,184,.09)",
        ),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def risk_gauge(value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 42, "color": "#F8FAFC"}},
            title={
                "text": "Experimental Risk Index",
                "font": {"size": 15, "color": "#94A3B8"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": "#22D3EE"},
                "bgcolor": "rgba(15,23,42,.35)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(34,197,94,.12)"},
                    {"range": [33, 66], "color": "rgba(245,158,11,.12)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=65, b=20),
    )
    return fig


def feature_heatmap(frame):
    norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-logo">◉</div>
          <div>
            <div class="brand-name">NeuroOcular AI</div>
            <div class="brand-sub">MEDICAL VISION PLATFORM</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠  Overview",
            "⚡  Live Scan",
            "📊  Analytics",
            "📄  Case Report",
            "⚙️  System",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### Data Source")

    uploaded_video = st.file_uploader(
        "Retinal video",
        type=["avi", "mp4", "mov", "m4v"],
        key="sidebar_video",
    )
    if uploaded_video is not None:
        st.session_state["video_path"] = save_upload(uploaded_video, "neuroocular_video")
        st.success("Video loaded")

    uploaded_model = st.file_uploader(
        "ONNX model",
        type=["onnx"],
        key="sidebar_model",
    )
    if uploaded_model is not None:
        st.session_state["model_path"] = save_upload(uploaded_model, "NeuroOcularNet")
        load_onnx_model.clear()
        st.success("Model loaded")

    model_ok = os.path.exists(get_model_path())
    video_ok = os.path.exists(get_video_path())

    st.markdown(
        f"""
        <div class="system-mini">
          <div><span class="dot {'ok' if model_ok else 'bad'}"></span> AI Engine</div>
          <strong>{'READY' if model_ok else 'MODEL MISSING'}</strong>
        </div>
        <div class="system-mini">
          <div><span class="dot {'ok' if video_ok else 'warn'}"></span> Video Source</div>
          <strong>{'READY' if video_ok else 'NOT LOADED'}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Research prototype · Not for clinical decision-making")


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "🏠  Overview":
    hero()

    section_header(
        "PLATFORM OVERVIEW",
        "من الفيديو الخام إلى Dashboard تحليلي متكامل",
        "Pipeline مرئي وواضح مناسب للعرض الأكاديمي والـdemo.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        info_card("👁️", "Retinal Imaging", "استقبال الفيديو ومعالجة الإطارات وتحضير البيانات البصرية.")

    with c2:
        info_card("〰️", "Flow Dynamics", "استخراج Dense Optical Flow ومؤشرات الحركة بين الإطارات.")

    with c3:
        info_card("🧠", "AI Inference", "تمرير الـfeatures إلى نموذج ONNX وتشغيل الاستدلال.")

    with c4:
        info_card("📈", "Analytics", "عرض المخرجات والمنحنيات ومؤشرات النموذج في واجهة واحدة.")

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        section_header(
            "WORKFLOW",
            "كيف يعمل النظام؟",
            "أربع مراحل أساسية من الإدخال إلى التقرير.",
        )

        s1, s2 = st.columns(2)
        with s1:
            info_card("01", "Capture", "تحميل فيديو الشبكية أو استخدام العينة التجريبية.")
            st.write("")
            info_card("03", "Inference", "تجهيز الـfeature vector وتشغيل نموذج ONNX.")
        with s2:
            info_card("02", "Vision", "Optical Flow + spatial gradients + signal statistics.")
            st.write("")
            info_card("04", "Review", "تحليل بصري، metrics، وتقرير بحثي قابل للتنزيل.")

    with right:
        st.markdown(
            """
            <div class="glass-panel tall-panel">
              <div class="panel-kicker">CURRENT SESSION</div>
              <div class="panel-title">System Readiness</div>
              <div class="check-row"><span>Video source</span><b>Ready when loaded</b></div>
              <div class="check-row"><span>ONNX runtime</span><b>CPU Provider</b></div>
              <div class="check-row"><span>Analysis mode</span><b>Research</b></div>
              <div class="check-row"><span>UI mode</span><b>RTL Dashboard</b></div>
              <div class="research-note">
                المؤشرات الطبية في هذه النسخة يجب اعتبارها مخرجات تجريبية
                حتى يتم التحقق منها سريريًا على بيانات مرجعية مستقلة.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: LIVE SCAN
# ============================================================

elif page == "⚡  Live Scan":
    section_header(
        "LIVE SCAN",
        "مركز تشغيل التحليل",
        "حمّل الفيديو والموديل من الـSidebar ثم ابدأ الفحص البحثي.",
    )

    left, right = st.columns([0.9, 1.6], gap="large")

    with left:
        st.markdown(
            """
            <div class="glass-panel">
              <div class="panel-kicker">SCAN PIPELINE</div>
              <div class="panel-title">Execution Steps</div>
              <div class="timeline-step"><b>01</b><span>Read retinal video</span></div>
              <div class="timeline-step"><b>02</b><span>Convert frames to grayscale</span></div>
              <div class="timeline-step"><b>03</b><span>Extract Optical Flow</span></div>
              <div class="timeline-step"><b>04</b><span>Run ONNX inference</span></div>
              <div class="timeline-step"><b>05</b><span>Build analytics report</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        run_scan = st.button("⚡ Run AI Analysis", type="primary", use_container_width=True)

    with right:
        video_path = get_video_path()

        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.markdown(
                """
                <div class="empty-state">
                  <div class="empty-icon">👁️</div>
                  <div class="empty-title">No retinal video loaded</div>
                  <div class="empty-text">ارفع فيديو من الشريط الجانبي لبدء التحليل.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if run_scan:
        if not os.path.exists(get_video_path()):
            st.error("ارفع فيديو أولاً.")
        elif not os.path.exists(get_model_path()):
            st.error("ارفع ملف NeuroOcularNet.onnx أولاً.")
        else:
            progress = st.progress(0)
            status = st.empty()

            try:
                status.info("Preparing the vision pipeline...")
                progress.progress(15)

                status.info("Extracting frames and optical flow...")
                progress.progress(35)

                data = analyze_video(get_video_path(), get_model_path())

                progress.progress(82)
                status.info("Building dashboard output...")

                st.session_state["analysis"] = data

                progress.progress(100)
                status.empty()
                progress.empty()

                st.success("Analysis completed. Open Analytics or Case Report from the sidebar.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Glucose model output", f"{data['glucose']:.1f} mg/dL")
                m2.metric("Viscosity model output", f"{data['viscosity']:.2f} mPa·s")
                m3.metric("Experimental risk index", f"{data['risk']}%")
                m4.metric("Frames analysed", str(data["frame_count"]))

            except Exception as e:
                progress.empty()
                status.empty()
                st.error(f"Analysis failed: {e}")
                with st.expander("Technical details"):
                    st.exception(e)


# ============================================================
# PAGE: ANALYTICS
# ============================================================

elif page == "📊  Analytics":
    section_header(
        "ANALYTICS",
        "لوحة التحليلات",
        "مؤشرات النموذج، Optical Flow، وFeature Visualization.",
    )

    data = st.session_state.get("analysis")

    if not data:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">📊</div>
              <div class="empty-title">No analysis yet</div>
              <div class="empty-text">شغّل التحليل من صفحة Live Scan أولاً.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Glucose model output", f"{data['glucose']:.1f} mg/dL")
        m2.metric("Viscosity model output", f"{data['viscosity']:.2f} mPa·s")
        m3.metric("Experimental risk index", f"{data['risk']}%")
        m4.metric("Frames analysed", f"{data['frame_count']}")

        st.write("")

        c1, c2 = st.columns([1.55, 1], gap="large")
        with c1:
            st.plotly_chart(flow_chart(data), use_container_width=True)
        with c2:
            st.plotly_chart(risk_gauge(data["risk"]), use_container_width=True)

        st.write("")

        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.plotly_chart(gradient_chart(data), use_container_width=True)

        mid = data["frames"][len(data["frames"]) // 2]
        heat = feature_heatmap(mid)

        with c4:
            tabs = st.tabs(["Retinal frame", "Feature visualization"])
            with tabs[0]:
                st.image(mid, use_container_width=True, clamp=True)
            with tabs[1]:
                st.image(heat, use_container_width=True)

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f"""
                <div class="glass-panel">
                  <div class="panel-kicker">SIGNAL SUMMARY</div>
                  <div class="stat-line"><span>Mean optical flow</span><b>{data['mean_speed']:.6f}</b></div>
                  <div class="stat-line"><span>Mean spatial gradient</span><b>{data['mean_gradient']:.6f}</b></div>
                  <div class="stat-line"><span>Source frames</span><b>{data['source_frame_count']}</b></div>
                  <div class="stat-line"><span>Processed frames</span><b>{data['frame_count']}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown(
                """
                <div class="glass-panel">
                  <div class="panel-kicker">INTERPRETATION</div>
                  <div class="panel-title">Research output only</div>
                  <p class="panel-text">
                    القيم الحالية تعتمد على mapping تجريبي بعد مخرجات ONNX.
                    يجب استبداله بالـinverse normalization الحقيقي المستخدم أثناء التدريب
                    قبل اعتبار الأرقام قابلة للتفسير الطبي.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE: CASE REPORT
# ============================================================

elif page == "📄  Case Report":
    section_header(
        "CASE REPORT",
        "ملخص الحالة البحثية",
        "تقرير منظم يمكن عرضه أو تنزيله بعد تشغيل التحليل.",
    )

    data = st.session_state.get("analysis")

    if not data:
        st.info("لا يوجد تحليل حالي. شغّل Live Scan أولاً.")
    else:
        report = {
            "generated_at": data["timestamp"],
            "glucose_model_output_mg_dl": round(data["glucose"], 2),
            "viscosity_model_output_mpa_s": round(data["viscosity"], 3),
            "experimental_risk_index_percent": data["risk"],
            "mean_optical_flow": round(data["mean_speed"], 6),
            "mean_spatial_gradient": round(data["mean_gradient"], 6),
            "processed_frames": data["frame_count"],
            "note": "Research prototype. Not validated for clinical diagnosis or treatment decisions.",
        }

        top1, top2, top3 = st.columns(3)
        top1.metric("AI output · glucose", f"{data['glucose']:.1f} mg/dL")
        top2.metric("AI output · viscosity", f"{data['viscosity']:.2f} mPa·s")
        top3.metric("Experimental index", f"{data['risk']}%")

        st.markdown(
            f"""
            <div class="report-card">
              <div class="report-head">
                <div>
                  <div class="panel-kicker">NEUROOCULAR AI</div>
                  <div class="panel-title">Research Analysis Summary</div>
                </div>
                <div class="report-time">{data['timestamp']}</div>
              </div>

              <div class="report-grid">
                <div><span>Mean Optical Flow</span><b>{data['mean_speed']:.6f}</b></div>
                <div><span>Mean Spatial Gradient</span><b>{data['mean_gradient']:.6f}</b></div>
                <div><span>Processed Frames</span><b>{data['frame_count']}</b></div>
                <div><span>Source Frames</span><b>{data['source_frame_count']}</b></div>
              </div>

              <div class="research-note">
                هذا التقرير ناتج عن نموذج بحثي. لا يُستخدم كتشخيص طبي ولا لاتخاذ قرار علاجي.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="neuroocular_report.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# PAGE: SYSTEM
# ============================================================

elif page == "⚙️  System":
    section_header(
        "SYSTEM",
        "حالة النظام وإعدادات المشروع",
        "تحقق سريع من الملفات والـruntime قبل العرض.",
    )

    model_path = get_model_path()
    video_path = get_video_path()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="glass-panel">
              <div class="panel-kicker">AI ENGINE</div>
              <div class="panel-title">{'Ready' if os.path.exists(model_path) else 'Model missing'}</div>
              <div class="stat-line"><span>Model path</span><b>{model_path}</b></div>
              <div class="stat-line"><span>Runtime</span><b>{'ONNX Runtime' if ort is not None else 'Not installed'}</b></div>
              <div class="stat-line"><span>Provider</span><b>CPUExecutionProvider</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="glass-panel">
              <div class="panel-kicker">VIDEO SOURCE</div>
              <div class="panel-title">{'Ready' if os.path.exists(video_path) else 'Video missing'}</div>
              <div class="stat-line"><span>Video path</span><b>{video_path}</b></div>
              <div class="stat-line"><span>Session result</span><b>{'Available' if st.session_state.get('analysis') else 'Not generated'}</b></div>
              <div class="stat-line"><span>Interface</span><b>Streamlit RTL</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.warning(
        "Important: the current glucose/viscosity output mapping and the composite risk index "
        "must match the model's actual training targets and validation pipeline."
    )
