# EdgeOps AI — IoT Predictive Maintenance Using Machine Learning

EdgeOps AI is a machine learning project that predicts potential industrial machine failures using sensor data. It uses the AI4I 2020 Predictive Maintenance Dataset to train and evaluate classification models for identifying normal and faulty machine operation.

The project includes a model training pipeline, evaluation visualizations, saved model pipelines, and an interactive Streamlit dashboard.

## Features

- Data loading and exploratory data analysis
- Sensor feature selection and preprocessing
- Training and comparison of three classification models
- Failure prediction probabilities
- Evaluation using accuracy, precision, recall, F1-score, and PR-AUC
- Confusion matrices and feature-importance visualizations
- Interactive dashboard built with Streamlit and Plotly
- Docker-based Python environment

## Dataset

**AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository**

[View the dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

- 10,000 observations
- 14 original columns
- 339 machine failures
- 9,661 normal-operation observations

The target variable is `Machine failure`:

- `0` — Normal operation
- `1` — Machine failure

Identifiers and failure-mode columns are excluded from the model inputs to prevent data leakage.

## Machine Learning Models

Three classifiers are trained using the same stratified train/test split:

1. **Random Forest Classifier** — tree-based classification with class weighting.
2. **XGBoost Classifier** — gradient-boosted trees with class-imbalance handling.
3. **Logistic Regression** — an interpretable baseline with numerical scaling and categorical encoding.

The data is divided into training, validation, and test sets. Decision thresholds are selected using validation data, and final metrics are calculated on the held-out test set.

### Test Results

Results from the current training run:

| Metric | Random Forest | XGBoost | Logistic Regression |
|---|---:|---:|---:|
| Accuracy | 97.55% | 97.95% | 94.70% |
| Failure Precision | 66.67% | 67.09% | 32.73% |
| Failure Recall | 55.88% | 77.94% | 52.94% |
| Failure F1-score | 0.6080 | 0.7211 | 0.4045 |
| PR-AUC (Average Precision) | 0.7181 | 0.8051 | 0.3914 |

XGBoost achieved the highest failure recall, F1-score, and PR-AUC in this run. Since machine failures represent only 3.39% of the dataset, accuracy alone is not sufficient to evaluate model performance.

## Interactive Dashboard

The Streamlit dashboard provides:

- Dataset and failure-rate overview
- Interactive model performance comparisons
- Confusion matrix visualizations
- Failure precision-recall analysis
- Random Forest and XGBoost feature importance
- Example prediction exploration and filtering
- CSV downloads of metrics and predictions

The current dashboard displays saved evaluation results and example predictions. Live sensor streaming is not yet implemented.

## Project Structure

```text
EdgeOpsAI/
├── data/
│   └── raw/
│       └── ai4i2020.csv
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── logistic_regression.joblib
├── results/
│   ├── dataset_summary.json
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── target_distribution.png
│   ├── example_predictions.csv
│   └── confusion matrices and feature importance
├── src/
│   ├── inspect_data.py
│   ├── train.py
│   └── dashboard.py
├── tests/
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Tech Stack

- **Language:** Python 3.12
- **Machine Learning:** scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib
- **Environment:** Docker

## Getting Started

### Prerequisites

Install Docker Desktop and ensure Docker Engine is running.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EdgeOpsAI.git
cd EdgeOpsAI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Build the Docker image

```bash
docker build -t edgeops-ai .
```

### 3. Inspect the dataset

Ensure the dataset is available at `data/raw/ai4i2020.csv`, then run:

```bash
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/app" -w /app edgeops-ai python src/inspect_data.py
```

### 4. Train the models

```bash
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/app" -w /app edgeops-ai python src/train.py
```

The trained models are saved under `models/`, and evaluation outputs are saved under `results/`.

### 5. Launch the dashboard

```bash
docker run --rm -p 8501:8501 --mount "type=bind,source=$($PWD.Path),target=/app" -w /app edgeops-ai streamlit run src/dashboard.py --server.address=0.0.0.0 --server.port=8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

These commands are intended for the Windows PowerShell workflow used during development.

## Limitations

- The dataset contains simulated industrial observations rather than live hardware readings.
- The current system performs offline training and evaluates saved predictions.
- The dashboard does not yet consume live IoT sensor streams.
- Model performance on real industrial machines has not been validated.

## Future Improvements

- Simulated real-time sensor streams
- FastAPI inference service
- Live monitoring dashboard and machine health alerts
- MQTT integration
- Docker Compose deployment
- Automated testing and inference benchmarking
- Optional LLM-based maintenance documentation search using RAG

## Author

Developed as a machine learning and predictive maintenance project to explore industrial sensor analytics, failure classification, model evaluation, and ML application deployment.
