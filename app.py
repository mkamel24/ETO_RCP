from __future__ import annotations

from pathlib import Path
import datetime as dt
import io
import json
import re
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import xgboost as xgb
except Exception:
    xgb = None


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FEATURE_COLUMNS = ["Tmax", "Tmin", "RH", "U"]
FEATURE_LABELS = {
    "Tmax": "Tmax (Maximum temperature, °C)",
    "Tmin": "Tmin (Minimum temperature, °C)",
    "RH": "RH (Relative humidity, %)",
    "U": "U (Wind speed, m/s)",
}
TOOLTIPS = {
    "Tmax": "Daily maximum air temperature in degrees Celsius.",
    "Tmin": "Daily minimum air temperature in degrees Celsius.",
    "RH": "Relative humidity in percent.",
    "U": "Wind speed in meters per second.",
}

ALGO_LABELS = {
    "xgb": "XGBoost",
    "lgb": "LightGBM",
    "cgb": "CatBoost",
}
DISPLAY_TO_ALGO = {v: k for k, v in ALGO_LABELS.items()}
METRIC_MODEL_LABELS = {"xgb": "BO-XGB", "lgb": "BO-LGB", "cgb": "BO-CGB"}

CASES_BY_INDEX = {
    1: {"zone": "Central Valley (Stn. 39)", "rcp": "RCP 4.5"},
    2: {"zone": "Central Valley (Stn. 39)", "rcp": "RCP 8.5"},
    3: {"zone": "Imperial Valley (Stn. 87)", "rcp": "RCP 4.5"},
    4: {"zone": "Imperial Valley (Stn. 87)", "rcp": "RCP 8.5"},
}

ZONE_DISPLAY_NAMES = {
    "Central Valley (Stn. 39)": "Parlier (Stn. 39)",
    "Imperial Valley (Stn. 87)": "Meloland (Stn. 87)",
}

PAPER_TITLE = (
    "Modeling Climate Change Effect on Reference Evapotranspiration in Semi-Arid Regions "
    "Using Explainable Hybrid Gradient Boosting Models"
)
AUTHOR_NAMES = [
    "Mohamed Kamel Elshaarawy",
    "Romysaa Elasbah",
    "Mohamed Elsayed Gabr",
    "Khaled M. Bali",
    "Mohamed Galal Eltarabily",
]
AUTHOR_COLORS = ["#D7263D", "#7B2CBF", "#1565C0", "#2E7D32", "#C77D00"]
TAGLINE = "Quick, reliable, and explainable – designed for researchers, engineers, and decision-makers."

ARTICLE_INFO = """Paper Title:
Modeling Climate Change Effect on Reference Evapotranspiration in Semi-Arid Regions Using Explainable Hybrid Gradient Boosting Models

Authors:
Mohamed Kamel Elshaarawy, Romysaa Elasbah, Mohamed Elsayed Gabr,
Khaled M. Bali, and Mohamed Galal Eltarabily
"""

PRESETS = {
    "— choose a preset —": None,
    "Baseline": [32.0, 18.0, 55.0, 2.4],
    "Hot & Dry": [39.0, 24.0, 32.0, 3.1],
    "Moderate": [31.0, 18.0, 52.0, 2.0],
    "Cool & Humid": [25.0, 13.0, 71.0, 1.4],
}

MODEL_METRICS = {
    ("xgb", "Central Valley (Stn. 39)", "RCP 4.5"): {
        "station": "Parlier (Stn. 39)",
        "train": {"R2": 0.99991, "RMSE": 0.019, "RMSRE": 0.007, "MAE": 0.012, "MARE": 0.004, "PBIAS": -0.004, "U95": 0.052},
        "test": {"R2": 0.99559, "RMSE": 0.129, "RMSRE": 0.042, "MAE": 0.085, "MARE": 0.028, "PBIAS": 0.071, "U95": 0.357},
    },
    ("lgb", "Central Valley (Stn. 39)", "RCP 4.5"): {
        "station": "Parlier (Stn. 39)",
        "train": {"R2": 0.99951, "RMSE": 0.043, "RMSRE": 0.020, "MAE": 0.029, "MARE": 0.011, "PBIAS": 0.000, "U95": 0.118},
        "test": {"R2": 0.99421, "RMSE": 0.148, "RMSRE": 0.050, "MAE": 0.103, "MARE": 0.035, "PBIAS": 0.083, "U95": 0.409},
    },
    ("cgb", "Central Valley (Stn. 39)", "RCP 4.5"): {
        "station": "Parlier (Stn. 39)",
        "train": {"R2": 0.99986, "RMSE": 0.023, "RMSRE": 0.010, "MAE": 0.018, "MARE": 0.007, "PBIAS": 0.001, "U95": 0.064},
        "test": {"R2": 0.99548, "RMSE": 0.131, "RMSRE": 0.049, "MAE": 0.090, "MARE": 0.033, "PBIAS": 0.282, "U95": 0.361},
    },
    ("xgb", "Central Valley (Stn. 39)", "RCP 8.5"): {
        "station": "Parlier (Stn. 39)",
        "train": {"R2": 0.99992, "RMSE": 0.018, "RMSRE": 0.009, "MAE": 0.011, "MARE": 0.004, "PBIAS": -0.002, "U95": 0.049},
        "test": {"R2": 0.98282, "RMSE": 0.258, "RMSRE": 0.106, "MAE": 0.171, "MARE": 0.061, "PBIAS": 1.017, "U95": 0.711},
    },
    ("lgb", "Central Valley (Stn. 39)", "RCP 8.5"): {
        "station": "Parlier (Stn. 39)",
        "train": {"R2": 0.99933, "RMSE": 0.050, "RMSRE": 0.021, "MAE": 0.036, "MARE": 0.013, "PBIAS": 0.000, "U95": 0.140},
        "test": {"R2": 0.97878, "RMSE": 0.287, "RMSRE": 0.110, "MAE": 0.198, "MARE": 0.071, "PBIAS": 0.762, "U95": 0.793},
    },
    ("cgb", "Central Valley (Stn. 39)", "RCP 8.5"): {
        "station": "Parlier (Stn. 39)",
        "train": {"R2": 0.99999, "RMSE": 0.007, "RMSRE": 0.003, "MAE": 0.006, "MARE": 0.002, "PBIAS": 0.000, "U95": 0.020},
        "test": {"R2": 0.98341, "RMSE": 0.253, "RMSRE": 0.093, "MAE": 0.162, "MARE": 0.054, "PBIAS": 0.608, "U95": 0.701},
    },
    ("xgb", "Imperial Valley (Stn. 87)", "RCP 4.5"): {
        "station": "Meloland (Stn. 87)",
        "train": {"R2": 0.99899, "RMSE": 0.077, "RMSRE": 0.025, "MAE": 0.050, "MARE": 0.013, "PBIAS": -0.009, "U95": 0.214},
        "test": {"R2": 0.98466, "RMSE": 0.307, "RMSRE": 0.073, "MAE": 0.186, "MARE": 0.044, "PBIAS": -0.152, "U95": 0.850},
    },
    ("lgb", "Imperial Valley (Stn. 87)", "RCP 4.5"): {
        "station": "Meloland (Stn. 87)",
        "train": {"R2": 0.99879, "RMSE": 0.084, "RMSRE": 0.028, "MAE": 0.061, "MARE": 0.016, "PBIAS": 0.000, "U95": 0.234},
        "test": {"R2": 0.98508, "RMSE": 0.302, "RMSRE": 0.076, "MAE": 0.197, "MARE": 0.047, "PBIAS": 0.010, "U95": 0.838},
    },
    ("cgb", "Imperial Valley (Stn. 87)", "RCP 4.5"): {
        "station": "Meloland (Stn. 87)",
        "train": {"R2": 0.99899, "RMSE": 0.077, "RMSRE": 0.021, "MAE": 0.058, "MARE": 0.015, "PBIAS": 0.000, "U95": 0.214},
        "test": {"R2": 0.98246, "RMSE": 0.328, "RMSRE": 0.076, "MAE": 0.196, "MARE": 0.045, "PBIAS": 0.210, "U95": 0.909},
    },
    ("xgb", "Imperial Valley (Stn. 87)", "RCP 8.5"): {
        "station": "Meloland (Stn. 87)",
        "train": {"R2": 0.99797, "RMSE": 0.110, "RMSRE": 0.030, "MAE": 0.070, "MARE": 0.017, "PBIAS": 0.004, "U95": 0.304},
        "test": {"R2": 0.99013, "RMSE": 0.248, "RMSRE": 0.065, "MAE": 0.167, "MARE": 0.040, "PBIAS": -0.267, "U95": 0.688},
    },
    ("lgb", "Imperial Valley (Stn. 87)", "RCP 8.5"): {
        "station": "Meloland (Stn. 87)",
        "train": {"R2": 0.99760, "RMSE": 0.119, "RMSRE": 0.034, "MAE": 0.087, "MARE": 0.021, "PBIAS": 0.000, "U95": 0.330},
        "test": {"R2": 0.98890, "RMSE": 0.263, "RMSRE": 0.068, "MAE": 0.188, "MARE": 0.044, "PBIAS": -0.192, "U95": 0.730},
    },
    ("cgb", "Imperial Valley (Stn. 87)", "RCP 8.5"): {
        "station": "Meloland (Stn. 87)",
        "train": {"R2": 0.99964, "RMSE": 0.046, "RMSRE": 0.012, "MAE": 0.035, "MARE": 0.008, "PBIAS": 0.000, "U95": 0.128},
        "test": {"R2": 0.98940, "RMSE": 0.257, "RMSRE": 0.065, "MAE": 0.173, "MARE": 0.040, "PBIAS": -0.575, "U95": 0.711},
    },
}


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
class NativeXGBRegressorWrapper:
    def __init__(self, model_path: Path):
        if xgb is None:
            raise RuntimeError(
                "xgboost is not installed in this environment, so the native XGBoost model cannot be loaded."
            )
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict(X)
        return self.model.predict(np.asarray(X))


def discover_model_paths(model_dir: Path):
    supported_ext = {".joblib", ".pkl", ".sav", ".json", ".ubj", ".model", ".bin"}
    results = {}
    missing = []

    if not model_dir.exists():
        return results, [(algo, case["zone"], case["rcp"]) for algo in ALGO_LABELS for case in CASES_BY_INDEX.values()]

    files = [p for p in model_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_ext]

    for algo in ALGO_LABELS:
        for idx, case in CASES_BY_INDEX.items():
            selected = None
            exact_names = [
                f"{algo.upper()}{idx}.joblib", f"{algo}{idx}.joblib", f"{algo}_{idx}.joblib", f"{algo}-{idx}.joblib",
                f"{algo.upper()}{idx}.pkl", f"{algo}{idx}.pkl", f"{algo.upper()}{idx}.sav", f"{algo}{idx}.sav",
                f"{algo.upper()}{idx}.json", f"{algo}{idx}.json", f"{algo.upper()}{idx}.ubj", f"{algo}{idx}.ubj",
                f"{algo.upper()}{idx}.model", f"{algo}{idx}.model", f"{algo.upper()}{idx}.bin", f"{algo}{idx}.bin",
            ]
            for name in exact_names:
                candidate = model_dir / name
                if candidate.exists():
                    selected = candidate
                    break

            if selected is None:
                regex = re.compile(rf"^{algo}[\W_]*{idx}(?:\b|_)", re.IGNORECASE)
                fuzzy = [p for p in files if regex.search(p.stem)]
                if fuzzy:
                    fuzzy = sorted(
                        fuzzy,
                        key=lambda p: (
                            p.suffix.lower() not in {".json", ".ubj", ".model", ".bin"} if algo == "xgb" else False,
                            str(p).lower(),
                        )
                    )
                    selected = fuzzy[0]

            key = (algo, case["zone"], case["rcp"])
            if selected is not None:
                results[key] = selected
            else:
                missing.append(key)
    return results, missing


def build_xgb_alternative_candidates(path: Path):
    candidates = []
    for ext in [".json", ".ubj", ".model", ".bin"]:
        alt = path.with_suffix(ext)
        if alt.exists():
            candidates.append(alt)
    return candidates


@st.cache_resource(show_spinner=False)
def load_model_cached(path_str: str, algo_code: str):
    path = Path(path_str)
    suffix = path.suffix.lower()

    if algo_code == "xgb" and suffix in {".json", ".ubj", ".model", ".bin"}:
        return NativeXGBRegressorWrapper(path)

    try:
        return joblib.load(path)
    except Exception as exc:
        if algo_code == "xgb":
            for alt in build_xgb_alternative_candidates(path):
                try:
                    return NativeXGBRegressorWrapper(alt)
                except Exception:
                    pass

            runtime_ver = getattr(xgb, "__version__", "not installed") if xgb is not None else "not installed"
            raise RuntimeError(
                "The selected XGBoost model could not be loaded.\n\n"
                f"File: {path.name}\n"
                f"Current xgboost runtime: {runtime_ver}\n\n"
                "This usually means one of these:\n"
                "1. the model was saved with a different XGBoost version,\n"
                "2. the file is corrupted,\n"
                "3. the XGBoost model should be loaded from a native file (.json / .ubj / .model)."
            ) from exc
        raise RuntimeError(f"Could not load model file '{path.name}'.\n\nOriginal error:\n{exc}") from exc


# ------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Nunito:wght@400;600;700&display=swap');
        .main * { font-family: 'Nunito', sans-serif; }
        .app-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.15rem;
            line-height: 1.25;
            text-align: center;
            font-weight: 700;
            color: #14213d;
            margin-bottom: 0.35rem;
        }
        .authors {
            text-align: center;
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .tagline {
            text-align: center;
            color: #56616f;
            font-style: italic;
            margin-bottom: 1rem;
        }
        .metricbox {
            border: 1px solid #E1E7EF;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: white;
        }
        .small-note {
            font-size: 0.9rem;
            color: #56616f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    title_html = f"<div class='app-title'>{PAPER_TITLE}</div>"
    st.markdown(title_html, unsafe_allow_html=True)
    author_html = "<div class='authors'>"
    for i, name in enumerate(AUTHOR_NAMES):
        author_html += f"<span style='color:{AUTHOR_COLORS[i]}'>{name}</span>"
        if i < len(AUTHOR_NAMES) - 1:
            author_html += " <span style='color:#6C757D'>•</span> "
    author_html += "</div>"
    st.markdown(author_html, unsafe_allow_html=True)
    st.markdown(f"<div class='tagline'>{TAGLINE}</div>", unsafe_allow_html=True)


def format_metrics_block(metrics: Dict[str, float]) -> str:
    return (
        f"R2    = {metrics['R2']:.5f}\n"
        f"RMSE  = {metrics['RMSE']:.3f} mm/day\n"
        f"RMSRE = {metrics['RMSRE']:.3f}\n"
        f"MAE   = {metrics['MAE']:.3f} mm/day\n"
        f"MARE  = {metrics['MARE']:.3f}\n"
        f"PBIAS = {metrics['PBIAS']:.3f} %\n"
        f"U95   = {metrics['U95']:.3f} mm/day"
    )


def metrics_text(key: Tuple[str, str, str], split: str) -> str:
    data = MODEL_METRICS.get(key)
    if not data:
        return f"{split.title()} metrics not available."
    header = (
        f"Model   : {METRIC_MODEL_LABELS[key[0]]}\n"
        f"Station : {data['station']}\n"
        f"Scenario: {key[2]}\n\n"
    )
    return header + format_metrics_block(data[split])


def init_state():
    st.session_state.setdefault("history_rows", [])
    st.session_state.setdefault("preset_name", "— choose a preset —")
    st.session_state.setdefault("tmax", 32.0)
    st.session_state.setdefault("tmin", 18.0)
    st.session_state.setdefault("rh", 55.0)
    st.session_state.setdefault("u", 2.4)


def set_preset_values():
    vals = PRESETS.get(st.session_state.preset_name)
    if vals is not None:
        st.session_state.tmax = float(vals[0])
        st.session_state.tmin = float(vals[1])
        st.session_state.rh = float(vals[2])
        st.session_state.u = float(vals[3])


def selected_key(algo_label: str, zone: str, rcp: str):
    return DISPLAY_TO_ALGO[algo_label], zone, rcp


def add_history_row(key: Tuple[str, str, str], inputs: Dict[str, float], prediction: float):
    algo, zone, rcp = key
    st.session_state.history_rows.append(
        {
            "Time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Algorithm": ALGO_LABELS[algo],
            "Zone": zone,
            "Scenario": rcp,
            "Tmax": inputs["Tmax"],
            "Tmin": inputs["Tmin"],
            "RH": inputs["RH"],
            "U": inputs["U"],
            "ETo": prediction,
        }
    )


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def predict_single(model, inputs_df: pd.DataFrame) -> float:
    try:
        return float(model.predict(inputs_df)[0])
    except Exception:
        return float(model.predict(inputs_df.to_numpy())[0])


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="ETo Smart Predictor", layout="wide")
    inject_css()
    init_state()
    render_header()

    model_paths, missing_models = discover_model_paths(MODEL_DIR)
    if not model_paths:
        st.error(
            "No model files were found in the app's models/ folder. Upload the trained files to models/ before deploying."
        )
    elif missing_models:
        with st.expander("Missing model profiles"):
            for algo, zone, rcp in missing_models:
                st.write(f"- {ALGO_LABELS[algo]} | {zone} | {rcp}")

    tab_predict, tab_batch, tab_history, tab_info = st.tabs(["Predict", "Batch", "History", "Article Info"])

    with tab_predict:
        left, right = st.columns([1.05, 1.0], gap="large")

        with left:
            st.subheader("Input Parameters")
            algo_label = st.selectbox("Algorithm", list(ALGO_LABELS.values()), index=0)
            zone = st.selectbox("Zone", sorted({case["zone"] for case in CASES_BY_INDEX.values()}), index=0)
            rcp = st.selectbox("Scenario", sorted({case["rcp"] for case in CASES_BY_INDEX.values()}), index=0)

            st.caption(f"Selected model: {algo_label} | {zone} | {rcp}")

            st.selectbox(
                "Preset",
                list(PRESETS.keys()),
                key="preset_name",
                on_change=set_preset_values,
            )

            st.session_state.tmax = st.number_input(FEATURE_LABELS["Tmax"], value=float(st.session_state.tmax), help=TOOLTIPS["Tmax"], format="%.4f")
            st.session_state.tmin = st.number_input(FEATURE_LABELS["Tmin"], value=float(st.session_state.tmin), help=TOOLTIPS["Tmin"], format="%.4f")
            st.session_state.rh = st.number_input(FEATURE_LABELS["RH"], value=float(st.session_state.rh), help=TOOLTIPS["RH"], format="%.4f")
            st.session_state.u = st.number_input(FEATURE_LABELS["U"], value=float(st.session_state.u), help=TOOLTIPS["U"], format="%.4f")

            c1, c2, c3 = st.columns(3)
            with c1:
                do_predict = st.button("Predict", use_container_width=True, type="primary")
            with c2:
                do_clear = st.button("Clear", use_container_width=True)
            with c3:
                save_inputs = st.button("Prepare Inputs JSON", use_container_width=True)

            if do_clear:
                st.session_state.tmax = 0.0
                st.session_state.tmin = 0.0
                st.session_state.rh = 0.0
                st.session_state.u = 0.0
                st.rerun()

            inputs_payload = {
                "Tmax": st.session_state.tmax,
                "Tmin": st.session_state.tmin,
                "RH": st.session_state.rh,
                "U": st.session_state.u,
            }
            if save_inputs:
                st.download_button(
                    "Download inputs.json",
                    data=json.dumps(inputs_payload, indent=2).encode("utf-8"),
                    file_name="inputs.json",
                    mime="application/json",
                    use_container_width=True,
                )

            uploaded_json = st.file_uploader("Load inputs from JSON", type=["json"], accept_multiple_files=False)
            if uploaded_json is not None:
                try:
                    loaded = json.loads(uploaded_json.getvalue().decode("utf-8"))
                    st.session_state.tmax = float(loaded.get("Tmax", st.session_state.tmax))
                    st.session_state.tmin = float(loaded.get("Tmin", st.session_state.tmin))
                    st.session_state.rh = float(loaded.get("RH", st.session_state.rh))
                    st.session_state.u = float(loaded.get("U", st.session_state.u))
                    st.success("Inputs loaded from JSON. Click Predict to run the model.")
                except Exception as exc:
                    st.error(f"Could not read JSON file: {exc}")

        with right:
            st.subheader("Prediction & Metrics")
            key = selected_key(algo_label, zone, rcp)
            model_path = model_paths.get(key)

            result_placeholder = st.container(border=True)
            with result_placeholder:
                st.markdown("**Predicted Reference Evapotranspiration**")
                if do_predict and model_path is not None:
                    try:
                        model = load_model_cached(str(model_path), key[0])
                        X = pd.DataFrame([inputs_payload], columns=FEATURE_COLUMNS)
                        pred = predict_single(model, X)
                        add_history_row(key, inputs_payload, pred)
                        st.markdown(f"## {pred:.4f}")
                        st.caption("Reference Evapotranspiration (mm/day)")
                        st.success(f"Prediction completed using {model_path.name}")
                    except Exception as exc:
                        st.error(str(exc))
                else:
                    st.markdown("## —")
                    st.caption("Reference Evapotranspiration (mm/day)")
                    if model_path is None:
                        st.warning("No model file found for the current selection.")

            m1, m2 = st.columns(2, gap="medium")
            with m1:
                st.markdown("### Train Metrics")
                st.code(metrics_text(key, "train"), language=None)
            with m2:
                st.markdown("### Test Metrics")
                st.code(metrics_text(key, "test"), language=None)

    with tab_batch:
        st.subheader("Batch Prediction")
        st.caption("Required columns: Tmax, Tmin, RH, U")
        algo_label = st.selectbox("Algorithm for batch", list(ALGO_LABELS.values()), index=0, key="batch_algo")
        zone = st.selectbox("Zone for batch", sorted({case["zone"] for case in CASES_BY_INDEX.values()}), index=0, key="batch_zone")
        rcp = st.selectbox("Scenario for batch", sorted({case["rcp"] for case in CASES_BY_INDEX.values()}), index=0, key="batch_rcp")
        batch_key = selected_key(algo_label, zone, rcp)
        batch_model_path = model_paths.get(batch_key)

        uploaded_csv = st.file_uploader("Open CSV", type=["csv"], key="batch_csv")
        if uploaded_csv is not None:
            try:
                batch_df = pd.read_csv(uploaded_csv)
                missing = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.dataframe(batch_df.head(200), use_container_width=True)
                    if st.button("Run Batch Prediction", type="primary"):
                        if batch_model_path is None:
                            st.error("No model file found for the current selection.")
                        else:
                            model = load_model_cached(str(batch_model_path), batch_key[0])
                            X = batch_df[FEATURE_COLUMNS].copy()
                            try:
                                preds = model.predict(X)
                            except Exception:
                                preds = model.predict(X.to_numpy())
                            out_df = batch_df.copy()
                            out_df["Predicted_ETo_mm_day"] = np.asarray(preds, dtype=float)
                            st.success(f"Batch prediction completed using {batch_model_path.name}")
                            st.dataframe(out_df.head(200), use_container_width=True)
                            st.download_button(
                                "Download batch_results.csv",
                                data=to_csv_download(out_df),
                                file_name="batch_results.csv",
                                mime="text/csv",
                            )
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    with tab_history:
        st.subheader("Prediction History")
        if st.session_state.history_rows:
            hist_df = pd.DataFrame(st.session_state.history_rows)
            st.dataframe(hist_df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Export History CSV",
                    data=to_csv_download(hist_df),
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                if st.button("Clear History", use_container_width=True):
                    st.session_state.history_rows = []
                    st.rerun()
        else:
            st.info("No history records available yet.")

    with tab_info:
        st.subheader("Article Info")
        st.text(ARTICLE_INFO)
        st.markdown(
            "<div class='small-note'>This Streamlit version replaces the original Tkinter desktop GUI so it can run on Streamlit Cloud.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
