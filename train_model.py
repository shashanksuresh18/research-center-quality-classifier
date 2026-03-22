from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "research_centers.csv"
MODEL_PATH = PROJECT_DIR / "cluster_model.pkl"
NOTEBOOK_PATH = PROJECT_DIR / "EDA_and_Model.ipynb"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Final feature set chosen during notebook analysis and reused for reproducible training.
selected_features = [
    "internalFacilitiesCount",
    "hospitals_10km",
    "pharmacies_10km",
    "facilityDiversity_10km",
    "facilityDensity_10km",
]

IDENTIFIER_COLUMNS = [
    "researchCenterId",
    "researchCenterName",
    "city",
    "latitude",
    "longitude",
]

REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + selected_features
QUALITY_TIERS = ("Basic", "Standard", "Premium")
DISPLAY_TIER_ORDER = ["Premium", "Standard", "Basic"]

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the assignment dataset and validate the required schema."""
    dataframe = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if dataframe[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    return dataframe.copy()


def ensure_directories() -> None:
    """Create output directories used for plots and tabular artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_eda_plots(dataframe: pd.DataFrame) -> None:
    """Generate the EDA plots requested in the assignment brief."""
    ensure_directories()

    figure, axis = plt.subplots(figsize=(8, 5))
    bins = range(
        int(dataframe["internalFacilitiesCount"].min()),
        int(dataframe["internalFacilitiesCount"].max()) + 2,
    )
    sns.histplot(
        dataframe["internalFacilitiesCount"],
        bins=bins,
        kde=True,
        color="#2a9d8f",
        edgecolor="white",
        ax=axis,
    )
    axis.set_title("Distribution of Internal Facilities Count")
    axis.set_xlabel("Internal Facilities Count")
    axis.set_ylabel("Number of Research Centers")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "internal_facilities_histogram.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=dataframe,
        x="hospitals_10km",
        y="pharmacies_10km",
        hue="city",
        size="internalFacilitiesCount",
        palette="Set2",
        sizes=(60, 280),
        ax=axis,
    )
    axis.set_title("Hospital vs Pharmacy Access Within 10 km")
    axis.set_xlabel("Hospitals Within 10 km")
    axis.set_ylabel("Pharmacies Within 10 km")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "hospital_pharmacy_access_scatter.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=dataframe,
        x="facilityDiversity_10km",
        y="facilityDensity_10km",
        hue="city",
        size="internalFacilitiesCount",
        palette="viridis",
        sizes=(60, 280),
        ax=axis,
    )
    axis.set_title("Facility Diversity vs Facility Density")
    axis.set_xlabel("Facility Diversity Within 10 km")
    axis.set_ylabel("Facility Density Within 10 km")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "diversity_density_scatter.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    correlation_matrix = dataframe[selected_features].corr(numeric_only=True)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=axis,
    )
    axis.set_title("Correlation Heatmap for Selected Features")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "feature_correlation_heatmap.png", dpi=200)
    plt.close(figure)


def train_clustering_model(
    dataframe: pd.DataFrame,
) -> tuple[StandardScaler, KMeans, pd.DataFrame, pd.Series, float]:
    """Scale the selected features, train K-Means, and score the clustering."""
    feature_frame = dataframe[selected_features].copy()
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_frame)
    scaled_frame = pd.DataFrame(scaled_array, columns=selected_features, index=dataframe.index)

    model = KMeans(n_clusters=3, n_init=20, random_state=42)
    cluster_labels = pd.Series(model.fit_predict(scaled_frame), index=dataframe.index, name="cluster")
    score = silhouette_score(scaled_frame, cluster_labels)

    return scaler, model, scaled_frame, cluster_labels, float(score)


def build_cluster_mapping(model: KMeans) -> tuple[dict[int, str], dict[int, float]]:
    """
    Rank clusters by the mean of their standardized centers.

    Every selected feature is positively aligned with research center quality,
    so a higher average standardized center indicates a stronger overall tier.
    """
    cluster_centers = pd.DataFrame(model.cluster_centers_, columns=selected_features)
    cluster_strength = cluster_centers.mean(axis=1).sort_values()
    ordered_clusters = cluster_strength.index.tolist()
    cluster_to_tier = {
        int(cluster_label): tier
        for cluster_label, tier in zip(ordered_clusters, QUALITY_TIERS, strict=True)
    }
    strength_lookup = {int(cluster_label): float(score) for cluster_label, score in cluster_strength.items()}
    return cluster_to_tier, strength_lookup


def build_cluster_outputs(
    dataframe: pd.DataFrame,
    model: KMeans,
    cluster_labels: pd.Series,
    cluster_to_tier: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create enriched output tables used by the README, notebook, and API docs."""
    clustered_frame = dataframe.copy()
    clustered_frame["cluster"] = cluster_labels.astype(int)
    clustered_frame["qualityTier"] = clustered_frame["cluster"].map(cluster_to_tier)
    clustered_frame["qualityTier"] = pd.Categorical(
        clustered_frame["qualityTier"],
        categories=DISPLAY_TIER_ORDER,
        ordered=True,
    )
    clustered_frame = clustered_frame.sort_values(["qualityTier", "researchCenterId"]).reset_index(drop=True)

    cluster_profile = (
        clustered_frame.groupby(["cluster", "qualityTier"], observed=True)[selected_features]
        .mean()
        .round(3)
        .reset_index()
        .sort_values("qualityTier")
        .reset_index(drop=True)
    )

    cluster_centers_scaled = (
        pd.DataFrame(model.cluster_centers_, columns=selected_features)
        .assign(cluster=lambda frame: frame.index.astype(int))
        .assign(qualityTier=lambda frame: frame["cluster"].map(cluster_to_tier))
        .assign(
            qualityTier=lambda frame: pd.Categorical(
                frame["qualityTier"],
                categories=DISPLAY_TIER_ORDER,
                ordered=True,
            )
        )
        .sort_values("qualityTier", ascending=True)
        .reset_index(drop=True)
    )

    city_tier_distribution = (
        pd.crosstab(clustered_frame["city"], clustered_frame["qualityTier"])
        .reindex(columns=DISPLAY_TIER_ORDER, fill_value=0)
        .reset_index()
    )

    return clustered_frame, cluster_profile, cluster_centers_scaled, city_tier_distribution


def save_cluster_artifacts(
    clustered_frame: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    cluster_centers_scaled: pd.DataFrame,
    city_tier_distribution: pd.DataFrame,
) -> None:
    """Persist tabular artifacts that support model interpretation."""
    ensure_directories()
    clustered_frame.to_csv(ARTIFACTS_DIR / "research_centers_clustered.csv", index=False)
    cluster_profile.to_csv(ARTIFACTS_DIR / "cluster_profile.csv", index=False)
    cluster_centers_scaled.to_csv(ARTIFACTS_DIR / "cluster_centers_scaled.csv", index=False)
    city_tier_distribution.to_csv(ARTIFACTS_DIR / "city_tier_distribution.csv", index=False)


def save_model_bundle(
    model: KMeans,
    scaler: StandardScaler,
    cluster_to_tier: dict[int, str],
    cluster_strength: dict[int, float],
    silhouette: float,
) -> dict[str, Any]:
    """Save the trained model bundle used by the FastAPI application."""
    bundle = {
        "model": model,
        "scaler": scaler,
        "selected_features": selected_features,
        "cluster_to_tier": cluster_to_tier,
        "cluster_strength": cluster_strength,
        "metrics": {"silhouette_score": round(silhouette, 4)},
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    return bundle


def _source_lines(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)}


def code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def create_notebook(
    dataframe: pd.DataFrame,
    silhouette: float,
    cluster_to_tier: dict[int, str],
) -> None:
    """Generate the deliverable notebook with reproducible analysis steps."""
    mapping_lines = "\n".join(
        f"- Cluster `{cluster_label}` maps to `{tier}`"
        for cluster_label, tier in sorted(cluster_to_tier.items())
    )

    notebook = {
        "cells": [
            markdown_cell(
                f"""
                # Research Center Quality Classification

                This notebook walks through the full solution for the Research Grid
                assignment: data validation, EDA, feature selection, K-Means clustering,
                and interpretation of the three quality tiers.

                Dataset snapshot:

                - Rows: `{len(dataframe)}`
                - Cities: `{dataframe['city'].nunique()}`
                - Selected modeling features: `{", ".join(selected_features)}`
                """
            ),
            code_cell(
                """
                from pathlib import Path

                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                from sklearn.preprocessing import StandardScaler

                sns.set_theme(style="whitegrid")

                DATA_PATH = Path("research_centers.csv")
                selected_features = [
                    "internalFacilitiesCount",
                    "hospitals_10km",
                    "pharmacies_10km",
                    "facilityDiversity_10km",
                    "facilityDensity_10km",
                ]
                """
            ),
            code_cell(
                """
                df = pd.read_csv(DATA_PATH)
                df.head()
                """
            ),
            markdown_cell(
                """
                ## 1. Data validation

                The first check is whether the required columns are present and whether
                any missing values need to be handled before clustering.
                """
            ),
            code_cell(
                """
                required_columns = [
                    "researchCenterId",
                    "researchCenterName",
                    "city",
                    "latitude",
                    "longitude",
                    *selected_features,
                ]

                {
                    "missing_columns": [col for col in required_columns if col not in df.columns],
                    "missing_values_per_column": df[required_columns].isna().sum().to_dict(),
                }
                """
            ),
            code_cell(
                """
                df[selected_features].describe().T
                """
            ),
            markdown_cell(
                """
                ## 2. Exploratory Data Analysis

                The assignment specifically asks for:

                - a histogram of internal facilities,
                - scatter plots for healthcare access,
                - a correlation heatmap for numeric variables.
                """
            ),
            code_cell(
                """
                fig, axes = plt.subplots(1, 3, figsize=(20, 5))

                sns.histplot(
                    df["internalFacilitiesCount"],
                    bins=range(
                        int(df["internalFacilitiesCount"].min()),
                        int(df["internalFacilitiesCount"].max()) + 2,
                    ),
                    kde=True,
                    color="#2a9d8f",
                    edgecolor="white",
                    ax=axes[0],
                )
                axes[0].set_title("Internal Facilities Distribution")

                sns.scatterplot(
                    data=df,
                    x="hospitals_10km",
                    y="pharmacies_10km",
                    hue="city",
                    size="internalFacilitiesCount",
                    sizes=(60, 280),
                    palette="Set2",
                    ax=axes[1],
                )
                axes[1].set_title("Hospital vs Pharmacy Access")

                sns.heatmap(
                    df[selected_features].corr(numeric_only=True),
                    annot=True,
                    cmap="YlGnBu",
                    fmt=".2f",
                    linewidths=0.5,
                    square=True,
                    ax=axes[2],
                )
                axes[2].set_title("Feature Correlation Heatmap")

                plt.tight_layout()
                plt.show()
                """
            ),
            markdown_cell(
                """
                ## 3. Feature selection

                The selected features were kept because they directly capture the two
                dimensions the brief cares about:

                - internal infrastructure,
                - nearby healthcare access and variety.

                All five features point in the same business direction: higher values
                suggest a better-equipped, better-supported research center.
                """
            ),
            code_cell(
                """
                X = df[selected_features].copy()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                silhouette = silhouette_score(X_scaled, clusters)
                silhouette
                """
            ),
            markdown_cell(
                f"""
                ## 4. Clustering and tier mapping

                The raw K-Means cluster ids are arbitrary, so they should not be mapped
                directly to `Premium`, `Standard`, and `Basic` without inspection.

                Instead, we rank the clusters using the mean of each cluster center in
                standardized feature space. Because every selected feature is positively
                aligned with quality, the strongest cluster becomes `Premium` and the
                weakest becomes `Basic`.

                Current silhouette score: `{silhouette:.4f}`

                Cluster mapping used in the final solution:

                {mapping_lines}
                """
            ),
            code_cell(
                """
                cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
                cluster_strength = cluster_centers.mean(axis=1).sort_values()
                ordered_clusters = cluster_strength.index.tolist()
                cluster_to_tier = {
                    cluster_label: tier
                    for cluster_label, tier in zip(ordered_clusters, ("Basic", "Standard", "Premium"))
                }

                results = df.copy()
                results["cluster"] = clusters
                results["qualityTier"] = results["cluster"].map(cluster_to_tier)
                results.head()
                """
            ),
            code_cell(
                """
                cluster_profile = (
                    results.groupby(["cluster", "qualityTier"])[selected_features]
                    .mean()
                    .round(3)
                    .reset_index()
                    .sort_values("qualityTier")
                )

                city_tier_distribution = pd.crosstab(results["city"], results["qualityTier"])

                cluster_profile
                """
            ),
            code_cell(
                """
                city_tier_distribution
                """
            ),
            markdown_cell(
                """
                ## 5. Interpretation

                The final interpretation should focus on:

                - which tier has the strongest internal facilities and best local access,
                - whether stronger centers are concentrated in specific cities,
                - whether density and diversity reinforce each other.

                Those talking points are also reflected in the project README.
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def train_and_save_outputs() -> dict[str, Any]:
    """Run the full assignment workflow and save all outputs."""
    dataframe = load_dataset()
    ensure_directories()
    save_eda_plots(dataframe)

    scaler, model, _, cluster_labels, silhouette = train_clustering_model(dataframe)
    cluster_to_tier, cluster_strength = build_cluster_mapping(model)
    clustered_frame, cluster_profile, cluster_centers_scaled, city_tier_distribution = build_cluster_outputs(
        dataframe=dataframe,
        model=model,
        cluster_labels=cluster_labels,
        cluster_to_tier=cluster_to_tier,
    )

    save_cluster_artifacts(
        clustered_frame=clustered_frame,
        cluster_profile=cluster_profile,
        cluster_centers_scaled=cluster_centers_scaled,
        city_tier_distribution=city_tier_distribution,
    )
    bundle = save_model_bundle(
        model=model,
        scaler=scaler,
        cluster_to_tier=cluster_to_tier,
        cluster_strength=cluster_strength,
        silhouette=silhouette,
    )
    if not NOTEBOOK_PATH.exists():
        create_notebook(dataframe=dataframe, silhouette=silhouette, cluster_to_tier=cluster_to_tier)

    return {
        "bundle": bundle,
        "clustered_frame": clustered_frame,
        "cluster_profile": cluster_profile,
        "cluster_centers_scaled": cluster_centers_scaled,
        "city_tier_distribution": city_tier_distribution,
    }


def main() -> None:
    """Execute the training workflow from the command line."""
    results = train_and_save_outputs()
    metrics = results["bundle"]["metrics"]
    print("Training complete.")
    print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
    print("Cluster mapping:")
    for cluster_label, tier in sorted(results["bundle"]["cluster_to_tier"].items()):
        print(f"  Cluster {cluster_label}: {tier}")


if __name__ == "__main__":
    main()
