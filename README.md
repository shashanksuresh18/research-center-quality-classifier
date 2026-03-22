# Research Center Quality Classification

This project solves the Research Grid Ltd Phase 1 machine learning assignment.
The goal is to classify synthetic UK research centers into three quality tiers
using unsupervised learning and expose the trained model through a FastAPI API.

## Problem Summary

Each research center has:

- internal infrastructure information,
- nearby healthcare access counts,
- facility diversity and density metrics.

Because the dataset does not contain a ground-truth target label, the problem is
framed as **clustering** rather than supervised classification. I used
`KMeans(n_clusters=3)` to group centers into:

- `Premium`
- `Standard`
- `Basic`

## Dataset

Input file: `research_centers.csv`

Dataset characteristics:

- 50 research centers
- 5 synthetic cities
- 10 original columns
- 0 missing values in required fields

Selected modeling features:

- `internalFacilitiesCount`
- `hospitals_10km`
- `pharmacies_10km`
- `facilityDiversity_10km`
- `facilityDensity_10km`

These five features were chosen because they directly measure the two quality
dimensions described in the brief:

- internal capability,
- access to supporting healthcare infrastructure.

## Approach

### 1. Exploratory Data Analysis

The project generates the required plots:

- histogram of internal facility counts,
- hospital vs pharmacy access scatter plot,
- diversity vs density scatter plot,
- correlation heatmap of the selected numeric features.

EDA artifacts are saved under `artifacts/plots/`.

### 2. Feature Selection

The selected features all have a clear business interpretation and are strongly
positively related. The correlation matrix showed that higher-quality centers
tend to score consistently well across internal facilities, hospital access,
pharmacy access, diversity, and density.

Some of the strongest correlations observed:

- `internalFacilitiesCount` vs `facilityDiversity_10km`: `0.904`
- `internalFacilitiesCount` vs `facilityDensity_10km`: `0.901`
- `internalFacilitiesCount` vs `pharmacies_10km`: `0.889`
- `internalFacilitiesCount` vs `hospitals_10km`: `0.879`

### 3. Clustering

Workflow:

1. Load and validate the dataset.
2. Standardize the selected features with `StandardScaler`.
3. Train `KMeans` with `k=3`, `random_state=42`, and `n_init=20`.
4. Evaluate clustering quality using the silhouette score.
5. Rank the clusters and map them to business-friendly tier labels.

Important note: raw K-Means cluster ids are arbitrary. I did **not** assume
that cluster `0` always means `Premium`. Instead, I ranked the cluster centers
using their average value in standardized feature space because all selected
features are positively aligned with quality.

Final mapping:

- Cluster `1` -> `Premium`
- Cluster `0` -> `Standard`
- Cluster `2` -> `Basic`

## Results

Silhouette score:

- `0.5519`

Tier distribution:

- `Premium`: 17 centers
- `Standard`: 17 centers
- `Basic`: 16 centers

Average feature values by tier:

| Tier | Internal Facilities | Hospitals | Pharmacies | Diversity | Density |
| --- | ---: | ---: | ---: | ---: | ---: |
| Premium | 9.529 | 3.471 | 4.118 | 0.850 | 0.537 |
| Standard | 4.941 | 1.529 | 2.059 | 0.560 | 0.290 |
| Basic | 2.312 | 0.500 | 0.438 | 0.279 | 0.125 |

Interpretation:

- `Premium` centers have the strongest internal infrastructure and the best
  nearby healthcare support.
- `Basic` centers are clearly separated by much lower facility counts and lower
  external access.
- `Standard` centers sit between the two extremes, which is what we would
  expect from a sensible three-tier clustering solution.

City-level distribution:

| City | Premium | Standard | Basic |
| --- | ---: | ---: | ---: |
| City 1 | 4 | 6 | 1 |
| City 2 | 2 | 2 | 2 |
| City 3 | 4 | 5 | 5 |
| City 4 | 4 | 1 | 5 |
| City 5 | 3 | 3 | 3 |

This suggests that stronger centers are not isolated to a single city, although
City 1 skews more toward `Standard` and City 4 has a larger share of `Basic`
centers.

## API

The FastAPI application is implemented in `app.py`.

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict`

Example request:

```json
{
  "internalFacilitiesCount": 9,
  "hospitals_10km": 3,
  "pharmacies_10km": 2,
  "facilityDiversity_10km": 0.82,
  "facilityDensity_10km": 0.45
}
```

Example response:

```json
{
  "predictedCluster": 1,
  "predictedCategory": "Premium"
}
```

## Repository Structure

```text
.
|-- .dockerignore
|-- .env.draft
|-- app.py
|-- artifacts/
|   |-- city_tier_distribution.csv
|   |-- cluster_centers_scaled.csv
|   |-- cluster_profile.csv
|   |-- metrics.json
|   |-- plots/
|   `-- research_centers_clustered.csv
|-- cluster_model.pkl
|-- docker-compose.yaml
|-- Dockerfile
|-- EDA_and_Model.ipynb
|-- README.md
|-- requirements.txt
|-- research_centers.csv
`-- train_model.py
```

## How To Run

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model and generate artifacts

```bash
python train_model.py
```

This creates:

- `cluster_model.pkl`
- `EDA_and_Model.ipynb`
- `artifacts/*.csv`
- `artifacts/plots/*.png`

### 4. Run the API

```bash
uvicorn app:app --reload
```

Swagger docs will be available at:

- `http://127.0.0.1:8000/docs`

## Docker Support

Build the image:

```bash
docker build -t research-center-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 research-center-api
```

Run with Docker Compose:

```bash
docker compose up --build
```

Open the API documentation:

- `http://127.0.0.1:8000/docs`

Test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Test the prediction endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

## Key Design Decisions

- `KMeans` is a good fit because the task is to discover tiers without a labeled
  target column.
- `StandardScaler` is necessary because the features are measured on different
  scales.
- Tier labels are assigned **after** clustering by ranking cluster centers, not
  by trusting the raw cluster id.
- The API will automatically train the model if `cluster_model.pkl` is missing,
  which makes the project easier to run from scratch.

## Files You Should Highlight In Interview

- `train_model.py`: end-to-end data, modeling, artifact generation
- `EDA_and_Model.ipynb`: notebook version of the reasoning and analysis
- `app.py`: serving the trained clustering model through FastAPI

## Possible Improvements

- Use a larger and more realistic dataset.
- Add geographic distance features beyond simple count summaries.
- Compare `KMeans` with hierarchical clustering or Gaussian mixture models.
- Add automated tests for the training pipeline and API.
