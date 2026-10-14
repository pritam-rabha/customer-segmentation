"""Customer segmentation toolkit: preprocessing, clustering, evaluation, insights, plots.

The package is split by responsibility so each part can be tested and reused alone:

    schema         column-name normalisation and business-role detection
    data_loader    CSV loading/validation and the synthetic sample generator
    preprocessing  missing values, outliers, scaling, normalisation
    clustering     K-Means, DBSCAN, Hierarchical (+ parameter helpers)
    evaluation     Elbow, Silhouette and other quality metrics
    insights       segment profiles, naming and recommended actions
    visualization  Plotly figures used by the dashboard (needs plotly)
"""

__version__ = "1.0.0"
