
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="EdgeOps AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 16px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_results():
    comparison_path = RESULTS / "model_comparison.csv"
    summary_path = RESULTS / "dataset_summary.json"
    predictions_path = RESULTS / "example_predictions.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            "model_comparison.csv not found. "
            "Run src/train.py first."
        )

    comparison = pd.read_csv(comparison_path)

    summary = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    predictions = pd.DataFrame()
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)

    return comparison, summary, predictions


def existing_image(filename):
    path = RESULTS / filename
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"Chart not found: {filename}")


# --------------------------------------------------
# Load and validate
# --------------------------------------------------

st.sidebar.title("⚙️ EdgeOps AI")
st.sidebar.caption("IoT Predictive Maintenance")
st.sidebar.divider()

try:
    comparison, summary, predictions = load_results()
except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Unable to load project results: {error}")
    st.stop()

required_metrics = {
    "model", "accuracy", "precision", "recall",
    "f1_score", "pr_auc", "threshold"
}

if not required_metrics.issubset(comparison.columns):
    st.error("The model comparison file is missing required metrics.")
    st.stop()

models = comparison["model"].tolist()

if not models:
    st.error("No trained model results were found.")
    st.stop()

if not MODELS.exists():
    st.warning("The models directory does not exist.")

st.sidebar.subheader("Dashboard navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Model Comparison",
        "Failure Analysis",
        "Feature Importance",
        "Prediction Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Dataset: UCI AI4I 2020\n\n"
    "Evaluation: held-out test set\n\n"
    "Thresholds: selected on validation data"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("⚙️ EdgeOps AI")
st.markdown(
    "### Industrial Machine Predictive Maintenance"
)
st.caption(
    "Machine-learning insights from simulated industrial sensor data."
)

st.divider()


# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":
    st.subheader("System overview")

    total_rows = summary.get("dataset_rows", 10000)
    failures = summary.get("failure_count", 339)
    normal = summary.get("normal_count", 9661)
    failure_pct = summary.get(
        "failure_percentage",
        failures / total_rows * 100 if total_rows else 0,
    )

    best_row = comparison.loc[
        comparison["pr_auc"].idxmax()
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset observations", f"{total_rows:,}")
    col2.metric("Recorded failures", f"{failures:,}")
    col3.metric("Failure rate", f"{failure_pct:.2f}%")
    col4.metric("Best PR-AUC model", str(best_row["model"]))

    st.info(
        "Machine failure is a rare event in this dataset. "
        "Evaluate recall, precision, F1-score and PR-AUC "
        "alongside accuracy."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Machine status distribution")

        status_df = pd.DataFrame({
            "Machine status": ["Normal", "Failure"],
            "Observations": [normal, failures],
        })

        fig = px.bar(
            status_df,
            x="Machine status",
            y="Observations",
            color="Machine status",
            text="Observations",
            title="Normal vs Failure",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model performance snapshot")

        metric = st.selectbox(
            "Select performance metric",
            ["pr_auc", "recall", "f1_score", "precision", "accuracy"],
            format_func=lambda x: {
                "pr_auc": "PR-AUC",
                "recall": "Failure Recall",
                "f1_score": "Failure F1-score",
                "precision": "Failure Precision",
                "accuracy": "Accuracy",
            }[x],
            key="overview_metric",
        )

        fig = px.bar(
            comparison,
            x="model",
            y=metric,
            color="model",
            text=metric,
            title=f"Model comparison: {metric}",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Saved model artifacts")

    artifact_rows = []
    for model_name in models:
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        artifact_path = MODELS / filename

        artifact_rows.append({
            "Model": model_name,
            "Artifact": filename,
            "Available": artifact_path.exists(),
        })

    st.dataframe(
        pd.DataFrame(artifact_rows),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

elif page == "Model Comparison":
    st.subheader("Compare trained models")

    display_names = {
        "accuracy": "Accuracy",
        "precision": "Failure Precision",
        "recall": "Failure Recall",
        "f1_score": "Failure F1-score",
        "pr_auc": "PR-AUC",
    }

    selected_metrics = st.multiselect(
        "Metrics to compare",
        list(display_names.keys()),
        default=["recall", "f1_score", "pr_auc"],
        format_func=lambda x: display_names[x],
    )

    if selected_metrics:
        chart_df = comparison.melt(
            id_vars="model",
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Score",
        )
        chart_df["Metric"] = chart_df["Metric"].map(display_names)

        fig = px.bar(
            chart_df,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            title="Test-set performance by model",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_traces(texttemplate="%{text:.3f}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation metrics")

    st.dataframe(
        comparison.style.format({
            "threshold": "{:.4f}",
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "pr_auc": "{:.4f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download metrics CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="edgeops_model_comparison.csv",
        mime="text/csv",
    )

    st.caption(
        "PR-AUC here is average precision. "
        "Metrics use the thresholds selected on validation data."
    )


# --------------------------------------------------
# Failure analysis
# --------------------------------------------------

elif page == "Failure Analysis":
    st.subheader("Confusion matrices")

    selected_model = st.selectbox(
        "Select model",
        models,
        key="confusion_model",
    )

    filename = (
        selected_model.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    existing_image(filename)

    st.caption(
        "Rows represent actual classes; columns represent "
        "predicted classes. The failure class is 1."
    )

    st.subheader("Failure detection trade-offs")

    tradeoff = comparison[
        ["model", "precision", "recall", "f1_score", "pr_auc"]
    ].melt(
        id_vars="model",
        var_name="Metric",
        value_name="Score",
    )

    fig = px.scatter(
        comparison,
        x="precision",
        y="recall",
        size="f1_score",
        color="model",
        hover_data=["f1_score", "pr_auc", "threshold"],
        range_x=[0, 1],
        range_y=[0, 1],
        title="Failure precision vs recall",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Higher recall means fewer missed failures; higher "
        "precision means fewer false alarms among predicted failures."
    )


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

elif page == "Feature Importance":
    st.subheader("What influences model predictions?")

    feature_models = ["Random Forest", "XGBoost"]

    selected_model = st.selectbox(
        "Select feature-importance model",
        feature_models,
    )

    filename = selected_model.lower().replace(" ", "_")
    importance_path = RESULTS / f"{filename}_feature_importance.csv"

    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)

        top_n = st.slider(
            "Number of features to display",
            min_value=3,
            max_value=max(3, min(20, len(importance_df))),
            value=min(10, len(importance_df)),
        )

        top_df = importance_df.nlargest(
            top_n, "importance"
        ).sort_values("importance")

        fig = px.bar(
            top_df,
            x="importance",
            y="feature",
            orientation="h",
            title=f"{selected_model}: feature importance",
            labels={
                "importance": "Importance score",
                "feature": "Feature",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            importance_df,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download feature importance CSV",
            data=importance_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename}_feature_importance.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            f"Feature-importance data not found: {importance_path.name}"
        )


# --------------------------------------------------
# Prediction explorer
# --------------------------------------------------

elif page == "Prediction Explorer":
    st.subheader("Explore example predictions")

    if predictions.empty:
        st.warning(
            "Example predictions were not found. "
            "Run src/train.py first."
        )
    else:
        probability_columns = [
            c for c in predictions.columns
            if c.endswith("_failure_probability")
        ]

        selected_probability = st.selectbox(
            "Probability column",
            probability_columns,
        )

        threshold_column = (
            selected_probability.removesuffix(
                "_failure_probability"
            ) + "_prediction"
        )

        min_probability = st.slider(
            "Minimum failure probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

        filtered = predictions[
            predictions[selected_probability] >= min_probability
        ].copy()

        st.metric("Matching examples", len(filtered))

        fig = px.histogram(
            predictions,
            x=selected_probability,
            nbins=15,
            title="Distribution of example failure probabilities",
        )
        fig.update_layout(
            xaxis_title="Predicted failure probability",
            yaxis_title="Number of examples",
        )
        st.plotly_chart(fig, use_container_width=True)

        if threshold_column in filtered.columns:
            filtered["Predicted status"] = filtered[
                threshold_column
            ].map({0: "Normal", 1: "Failure"})

        st.dataframe(
            filtered,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download filtered predictions",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="edgeops_filtered_predictions.csv",
            mime="text/csv",
        )

        st.caption(
            "These are saved predictions from held-out test observations, "
            "not live sensor readings."
        )


st.divider()
st.caption(
    "EdgeOps AI | Educational predictive-maintenance prototype | "
    "AI4I 2020 dataset"
)
