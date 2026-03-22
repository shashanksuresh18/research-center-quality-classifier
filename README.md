# Research Center Quality Classifier

Research Grid Ltd Phase 1 submission.

This project groups synthetic UK research centers into three quality tiers using
K-Means clustering and serves predictions through a FastAPI application. The
repository includes the analysis notebook, training pipeline, API, static
frontend, saved model artifact, and Docker setup.

## Live Deployment

- App: `https://research-center-quality-classifier.onrender.com/`
- API docs: `https://research-center-quality-classifier.onrender.com/docs`
- Health endpoint: `https://research-center-quality-classifier.onrender.com/health`

## Ways to Run the Project

There are three main ways to use this project:

1. Clone the repository and run it locally with Python and `uvicorn`
2. Run it in Docker with `docker build` / `docker run` or `docker compose`
3. Use the live Render deployment directly in the browser

The detailed steps for each option are included later in this README.

## Problem Summary

The dataset does not contain a target label such as `qualityTier`, so this is an
unsupervised learning problem rather than a supervised classification problem.
The workflow is:

1. inspect and validate the dataset
2. perform EDA on the key numeric variables
3. select the features that describe center quality
4. scale the features with `StandardScaler`
5. cluster centers with `KMeans(n_clusters=3)`
6. map the raw cluster ids to `Premium`, `Standard`, and `Basic`
7. expose the trained model through FastAPI

## Dataset

Input file: `research_centers.csv`

Dataset summary:

- 50 rows
- 5 synthetic cities
- 10 original columns
- no missing values in the required fields

Selected modelling features:

- `internalFacilitiesCount`
- `hospitals_10km`
- `pharmacies_10km`
- `facilityDiversity_10km`
- `facilityDensity_10km`

Excluded fields:

- `researchCenterId`
- `researchCenterName`
- `city`
- `latitude`
- `longitude`

The selected features were kept because they directly capture internal
capability and nearby healthcare support.

## Approach

### Exploratory Data Analysis

The notebook and training pipeline cover:

- data shape and schema checks
- missing-value validation
- summary statistics
- histogram of `internalFacilitiesCount`
- healthcare access scatter plots
- correlation heatmap
- cluster profile and city-tier summary tables

Notebook deliverable:

- `EDA_and_Model.ipynb`

Generated analysis artifacts:

- `artifacts/plots/internal_facilities_histogram.png`
- `artifacts/plots/hospital_pharmacy_access_scatter.png`
- `artifacts/plots/diversity_density_scatter.png`
- `artifacts/plots/feature_correlation_heatmap.png`

### Model

Training is implemented in `train_model.py`.

Pipeline steps:

1. load and validate the dataset
2. scale the selected features with `StandardScaler`
3. fit `KMeans(n_clusters=3, random_state=42, n_init=20)`
4. evaluate the clustering with silhouette score
5. rank cluster centers in standardized feature space
6. map clusters to business tiers
7. save the trained artifact and interpretation tables

Important detail:

- raw K-Means cluster ids are arbitrary
- the final tier mapping is derived from the cluster-center strength, not from
  assuming that cluster `0` always means the same business label

## Results

Silhouette score:

- `0.5519`

Cluster-to-tier mapping:

- Cluster `1` -> `Premium`
- Cluster `0` -> `Standard`
- Cluster `2` -> `Basic`

Average feature values by tier:

| Tier | Internal Facilities | Hospitals | Pharmacies | Diversity | Density |
| --- | ---: | ---: | ---: | ---: | ---: |
| Premium | 9.529 | 3.471 | 4.118 | 0.850 | 0.537 |
| Standard | 4.941 | 1.529 | 2.059 | 0.560 | 0.290 |
| Basic | 2.312 | 0.500 | 0.438 | 0.279 | 0.125 |

Tier counts:

- `Premium`: 17
- `Standard`: 17
- `Basic`: 16

Saved outputs:

- `cluster_model.pkl`
- `artifacts/cluster_profile.csv`
- `artifacts/cluster_centers_scaled.csv`
- `artifacts/city_tier_distribution.csv`
- `artifacts/research_centers_clustered.csv`
- `artifacts/metrics.json`

## API

The FastAPI app is implemented in `app.py`.

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict`
- `GET /docs`

The root route serves the static frontend in `index.html`.

If `cluster_model.pkl` is missing, the API calls `train_and_save_outputs()` from
`train_model.py` automatically on first use.

### Example Request

```json
{
  "internalFacilitiesCount": 9,
  "hospitals_10km": 3,
  "pharmacies_10km": 2,
  "facilityDiversity_10km": 0.82,
  "facilityDensity_10km": 0.45
}
```

### Example Response

```json
{
  "predictedCluster": 1,
  "predictedCategory": "Premium"
}
```

### Health Response

```json
{
  "status": "ok",
  "modelReady": true,
  "silhouetteScore": 0.5519
}
```

## Frontend

The repository includes a single-file frontend in `index.html`.

It provides:

- five sliders for the model inputs
- a live JSON payload preview
- an automatic `POST /predict` request on load and on slider change
- a tier badge and progress bar for the prediction result
- graceful handling of API and network failures

When served by FastAPI, the frontend uses the same origin automatically. It can
also be pointed at another backend with `?apiBaseUrl=...`.

## Project Structure

```text
.
|-- .dockerignore
|-- .env.draft
|-- .gitignore
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
|-- index.html
|-- README.md
|-- requirements.txt
|-- research_centers.csv
`-- train_model.py
```

## Local Setup

### 1. Create a virtual environment

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

### 3. Retrain the model if needed

```bash
python train_model.py
```

This regenerates:

- `cluster_model.pkl`
- `artifacts/*.csv`
- `artifacts/plots/*.png`

### 4. Run the API locally

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### 5. Test the prediction endpoint

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

## Docker

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

Docker healthcheck:

- `GET /health`

Access the containerised app:

- `http://127.0.0.1:8000/`
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

## Deployment

This project is deployed as a Docker-based Render web service.

Useful routes:

- `/` for the frontend
- `/docs` for Swagger UI
- `/health` for readiness checks
- `/predict` for model inference

## Key Files

- `EDA_and_Model.ipynb`: notebook deliverable with the EDA and modelling flow
- `train_model.py`: reproducible training and artifact generation pipeline
- `app.py`: FastAPI inference service
- `index.html`: single-file frontend served by the API
- `cluster_model.pkl`: saved trained model bundle

## Possible Improvements

- compare K-Means against other clustering methods on a larger dataset
- add automated API and training tests
- add stronger model monitoring for deployment
- extend the frontend with richer result explanations
