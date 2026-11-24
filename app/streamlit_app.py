from __future__ import annotations

import hashlib
import sys
from html import escape
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.inference import (
    InferenceError,
    ModelUnavailableError,
    Prediction,
    load_threshold,
    load_trained_model,
    predict_image,
)
from docdet.upload import UploadValidationError, decode_uploaded_image


@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_trained_model()


st.set_page_config(page_title="ID-GUARD", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #102a43;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.25rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96);
    }
    [data-testid="stTitle"] h1 {
        color: #102a43;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0;
    }
    .id-guard-rule {
        border-top: 1px solid #d9e2ec;
        margin: 0.9rem 0 1.8rem;
    }
    .id-guard-section {
        color: #102a43;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0 0 0.9rem;
    }
    .id-guard-result {
        border: 1px solid #d9e2ec;
        border-radius: 0.5rem;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .id-guard-result-label {
        color: #486581;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .id-guard-result-value {
        color: #102a43;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .id-guard-footer {
        color: #627d98;
        font-size: 0.8rem;
        margin-top: 2rem;
        text-align: center;
    }
    [data-testid="stBaseButton-primary"] {
        background-color: #102a43;
        border-color: #102a43;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #243b53;
        border-color: #243b53;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("ID-GUARD")
st.markdown('<div class="id-guard-rule"></div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    uploaded_file = st.file_uploader(
        "Upload specimen image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )
    st.caption(
        "PNG or JPEG, up to 10 MB. Upload synthetic specimens only; "
        "do not upload real identity documents."
    )
    uploaded_image = None
    upload_id = None
    if uploaded_file is not None:
        upload_data = uploaded_file.getvalue()
        try:
            uploaded_image = decode_uploaded_image(upload_data)
        except UploadValidationError as exc:
            st.error(str(exc))
        else:
            upload_id = hashlib.sha256(upload_data).hexdigest()
            st.image(
                uploaded_image,
                caption="Uploaded specimen preview",
                width="stretch",
            )

with right:
    st.markdown(
        '<h2 class="id-guard-section">Analysis Result</h2>',
        unsafe_allow_html=True,
    )
    result_record = st.session_state.get("id_guard_analysis_result")
    result: Prediction | None = None
    if (
        upload_id is not None
        and isinstance(result_record, tuple)
        and len(result_record) == 2
        and result_record[0] == upload_id
        and isinstance(result_record[1], Prediction)
    ):
        result = result_record[1]
    result_area = st.empty()

    def render_analysis_result(current: Prediction | None) -> None:
        with result_area.container():
            if current is None:
                st.markdown(
                    '<div class="id-guard-result">'
                    '<div class="id-guard-result-label">Prediction</div>'
                    '<div class="id-guard-result-value">No analysis yet</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="id-guard-result">'
                    '<div class="id-guard-result-label">Prediction</div>'
                    f'<div class="id-guard-result-value">{escape(current.label)}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Model score", f"{current.score:.4f}")

    render_analysis_result(result)
    if st.button(
        "Analyze Document",
        type="primary",
        width="stretch",
        disabled=uploaded_image is None,
    ):
        try:
            with st.spinner("Analyzing document..."):
                if uploaded_image is None or upload_id is None:
                    raise UploadValidationError("Upload a specimen image before analyzing.")
                threshold = load_threshold()
                model = _cached_model()
                prediction = predict_image(model, uploaded_image, threshold)
            st.session_state["id_guard_analysis_result"] = (upload_id, prediction)
            render_analysis_result(prediction)
        except ModelUnavailableError as exc:
            st.session_state.pop("id_guard_analysis_result", None)
            render_analysis_result(None)
            st.error(str(exc))
        except InferenceError as exc:
            st.session_state.pop("id_guard_analysis_result", None)
            render_analysis_result(None)
            st.error(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            st.session_state.pop("id_guard_analysis_result", None)
            render_analysis_result(None)
            st.error(str(exc))

st.markdown(
    '<div class="id-guard-footer">Educational prototype. Synthetic specimens only.</div>',
    unsafe_allow_html=True,
)
