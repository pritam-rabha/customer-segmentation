# Customer Segmentation

Group Indian retail customers by behaviour and value using **K-Means**, **DBSCAN**, and **Hierarchical Clustering**. Choose cluster counts with the **Elbow Method** and **Silhouette Analysis**, explore results in an interactive **Streamlit** dashboard, and get a named segment with recommended actions for every group.

**Technology stack:** Python, Pandas, NumPy, Scikit-learn, SciPy, Plotly, Streamlit.

## Author and academic details

| Field | Details |
|---|---|
| Application | Customer Segmentation |
| Author | Pritam Rabha |
| Programme | MCA, 3rd Semester |
| Course | Python Programming |
| Project guide | Dr. Pangambam Sendash Singh |
| Institution | Dr. Hari Singh Gour Vishwavidyalaya, Sagar, Madhya Pradesh |

Copyright © 2026 Pritam Rabha. All rights reserved.

## Contents

1. [Assignment requirements](#assignment-requirements-and-implementation)
2. [Quick start](#quick-start)
3. [Dashboard tour](#dashboard-tour)
4. [Method](#method)
5. [Results on the bundled sample](#results-on-the-bundled-sample)
6. [Project structure](#project-structure)
7. [Tests](#tests)
8. [Limitations](#limitations)
9. [Copyright](#copyright)
10. [Licence](#licence)

## Assignment requirements and implementation

| Requirement | Implementation |
|---|---|
| Data preprocessing and normalisation | `segmentation/preprocessing.py`: duplicate removal, median/mean/drop imputation, IQR outlier clipping, optional log transform, z-score / min-max / robust scaling, optional L2 normalisation, one-hot encoding |
| K-Means, DBSCAN, Hierarchical Clustering | `segmentation/clustering.py`: `run_kmeans`, `run_dbscan`, `run_hierarchical`; four linkages and dendrogram support |
| Elbow Method and Silhouette Analysis | `segmentation/evaluation.py`: `cluster_scan`, `recommend_k`, per-customer silhouette; shown in the **Choose parameters** and **Clusters** tabs |
| Cluster visualisation | `segmentation/visualization.py`: PCA 2-D/3-D scatter, feature scatter, size bars, silhouette plot, box plots, heatmap, radar chart, dendrogram |
| Business insights | `segmentation/insights.py`: segment profiles, rule-based naming, traits, actions, and a KPI per segment |
| Interactive Streamlit dashboard | `app.py`: six tabs, with parameters adjustable from the sidebar |
| Documentation and demonstration | This README and `docs/DEMO_GUIDE.md` |
| Modular, structured, commented code | One module per responsibility; analytics are kept out of the dashboard UI |
| GitHub and continuous integration | `docs/GITHUB_UPLOAD.md`, `.gitignore`, and the CI workflow |

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/customer-segmentation.git
cd customer-segmentation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The dashboard opens at <http://localhost:8501> with the bundled sample of 1,000 Indian retail customers. The sample uses amounts in rupees, cities and states across India, and UPI / card / cash-on-delivery payment modes.

### Run the pipeline from the terminal

If you do not need the browser dashboard, run the pipeline directly. This requires Pandas, NumPy, SciPy, and scikit-learn.

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --algorithm dbscan
python scripts/run_pipeline.py --algorithm hierarchical --k 4 --linkage average
python scripts/run_pipeline.py --data data/Mall_Customers.csv
```

Results are written to `outputs/`:

- `segmented_customers.csv`
- `segment_profiles.csv`
- `cluster_scan.csv`
- `segment_insights.md`

## Dashboard tour

| Tab | Purpose |
|---|---|
| **Data** | Row counts, missing values, correlations, preprocessing changes, and before/after scaling for a selected feature |
| **Choose parameters** | Elbow and silhouette curves, Davies–Bouldin and Calinski–Harabasz metrics, dendrogram, DBSCAN k-distance plot, and `eps` / `min_samples` search |
| **Clusters** | Quality metrics, PCA or feature scatter (2-D and 3-D), segment sizes, silhouette per customer, and metric distributions |
| **Segments and actions** | Customer share vs. spend share, profile table, deviation heatmap, radar chart, segment mix by city tier / state / payment mode / category, and a playbook for each segment |
| **Compare algorithms** | Side-by-side comparison of all three methods and an agreement matrix using Adjusted Rand Index |
| **Export** | Segmented customers, profile table, and insight report |

The interface is designed for analysts: a header showing the active dataset and method, a KPI strip for customers, segments, silhouette, unassigned share, and estimated annual spend, plus chart panels on a quiet grey canvas. It uses IBM Plex Sans typography and a consistent accent colour. Segment colours remain consistent across charts and the playbook. Amounts use Indian formatting (₹3,512, ₹12.35 lakh, ₹3.21 Cr). Columns stack on narrow screens.

The theme is defined in `assets/style.css` and `.streamlit/config.toml`.

## Method

### Preprocessing

Duplicates are removed by customer ID. Missing values are imputed (median by default), or rows can be dropped. Outliers are clipped to 1.5 × IQR fences for **model input only**. The cleaned table used for profiling retains the original values, so segment averages remain in real units. Features are then scaled (z-score by default), because distance-based algorithms can be dominated by columns with larger numeric ranges.

### Choosing *k*

K-Means is run for *k* = 2–10. The elbow is identified as the point on the inertia curve farthest from the line joining its endpoints. The recommended *k* is the silhouette maximum, except when the elbow's silhouette score is within 0.02 of the best score; in that case, the elbow value is selected as the simpler model with comparable quality.

### DBSCAN

`min_samples` defaults to twice the number of features, with a minimum of 4. The dashboard searches a grid of `eps` and `min_samples` values and pre-selects the setting with the best silhouette score that has at least two clusters and no more than 20% noise. Customers DBSCAN cannot assign to a cluster are shown as **Unassigned (noise)**.

### Naming segments and recommending actions

For each cluster, metrics are expressed as z-scores against the whole customer base. Two scores are derived:

- **Value:** income, average order value, and estimated annual spend.
- **Engagement:** spending score, order frequency, and recency (fewer days since the last order means higher engagement).

These scores place each cluster in a 3 × 3 grid. For example, high value and high engagement is labelled **Champions**, while high value and low engagement is labelled **Affluent, low engagement**. Each archetype has a short action plan and a metric to track. Each segment also lists the three traits that differ most from average, with values. If the columns are not recognisable, segments are named **Segment n** and only data-driven traits are shown.

## Results on the bundled sample

Default settings, K-Means:

| Check | Result |
|---|---|
| Elbow *k* / best-silhouette *k* | 5 / 5 |
| Silhouette, Davies–Bouldin, Calinski–Harabasz | 0.378, 0.989, 591.0 |
| Agreement with the generator's hidden groups (ARI) | About 0.95 (K-Means), about 0.96 (Hierarchical, Ward) |
| Champions | 20% of customers, about 53% of estimated spend |
| Dormant / at risk | About 16% of customers, about 2% of estimated spend |

DBSCAN on the same data finds three dense groups plus about 8% noise, because overlapping groups form density bridges. The **Compare algorithms** tab displays this behaviour. Single or average linkage can produce a few tiny clusters; Ward linkage is the recommended default for compact customer groups.

The dataset is synthetic. These figures demonstrate the pipeline on bundled sample data and are **not claims about real customers**.

## Project structure

```text
customer-segmentation/
├── app.py                       # Streamlit dashboard (UI only)
├── segmentation/
│   ├── config.py                # Paths and constants
│   ├── schema.py                # Header normalisation and column-role detection
│   ├── data_loader.py           # CSV loading/validation and sample generator
│   ├── preprocessing.py         # Cleaning, scaling, normalisation
│   ├── clustering.py            # K-Means, DBSCAN, Hierarchical, DBSCAN helpers
│   ├── evaluation.py            # Elbow, silhouette, recommended k, agreement
│   ├── insights.py              # Profiles, segment names, actions, report
│   ├── formatting.py            # Rupee / lakh / crore formatting
│   ├── visualization.py         # Plotly figures
│   └── utils.py                 # Knee detection
├── assets/
│   └── style.css                # Dashboard theme
├── scripts/
│   ├── generate_dataset.py      # Rebuild data/customers.csv
│   └── run_pipeline.py          # Command-line pipeline
├── tests/                       # Unit tests and dashboard smoke test
├── data/                        # Sample dataset and download instructions
├── docs/                        # Demo guide and GitHub upload guide
├── .streamlit/
│   └── config.toml              # Theme configuration
├── .github/workflows/
│   └── tests.yml                # Continuous integration
└── requirements.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
# Or:
python -m unittest discover -s tests -t .
```

The suite covers rupee formatting, the Indian sample's structure, loader validation and the Mall Customers layout, every preprocessing option, all three algorithms (K-Means and Hierarchical must recover the generator's known groups), elbow and silhouette logic, noise handling, insight naming, and report output.

When Plotly and Streamlit are installed, the suite also builds every figure and runs `app.py` headlessly with Streamlit's `AppTest` for each algorithm.

## Limitations

- Hierarchical clustering is limited to 5,000 customers because it needs O(*n*²) memory. Use K-Means for larger datasets.
- Silhouette is estimated on a 5,000-row sample for large datasets.
- At least two numeric features are required; categorical columns can be added on top of them.
- Segment names come from transparent rules, not a learned model; review them before acting.
- Recommended actions are starting hypotheses. Validate them with a holdout group before spending budget.

## Copyright

© 2026 Pritam Rabha. All rights reserved.

## Licence

MIT License. See [`LICENSE`](LICENSE).
