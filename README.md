# Customer Segmentation

Group Indian retail customers by behaviour and value with **K-Means**, **DBSCAN** and **Hierarchical clustering**,
choose the number of clusters with the **Elbow method** and **Silhouette analysis**, explore the
result in an interactive **Streamlit** dashboard, and get a named segment with recommended actions
for every group.

Stack: Python, Pandas, NumPy, Scikit-learn, SciPy, Plotly, Streamlit.

## Assignment requirements and where they are met

| Requirement | Implementation |
|---|---|
| Data preprocessing and normalisation | `segmentation/preprocessing.py`: duplicate removal, median/mean/drop imputation, IQR outlier clipping, optional log transform, z-score / min-max / robust scaling, optional L2 normalisation, one-hot encoding |
| K-Means, DBSCAN, Hierarchical | `segmentation/clustering.py` (`run_kmeans`, `run_dbscan`, `run_hierarchical`; four linkages; dendrogram support) |
| Elbow method and Silhouette analysis | `segmentation/evaluation.py` (`cluster_scan`, `recommend_k`, per-customer silhouette) shown in the **Choose parameters** and **Clusters** tabs |
| Visualisation of clusters | `segmentation/visualization.py`: PCA 2-D/3-D scatter, feature scatter, size bars, silhouette plot, box plots, heatmap, radar, dendrogram |
| Business insights | `segmentation/insights.py`: segment profiles, rule-based naming, traits, actions and a KPI per segment |
| Interactive Streamlit dashboard | `app.py` (six tabs, all parameters adjustable from the sidebar) |
| Documentation and demonstration | this README, `docs/DEMO_GUIDE.md` |
| Modular, structured, commented code | one module per responsibility; the dashboard contains no analytics |
| Code on GitHub | `docs/GITHUB_UPLOAD.md`, `.gitignore`, CI workflow |

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

The dashboard opens at <http://localhost:8501> with the bundled sample of 1,000 Indian retail customers loaded
(amounts in rupees, cities and states across India, UPI / card / cash-on-delivery payment modes).

No browser available? Run the whole pipeline from the terminal (needs only pandas, NumPy, SciPy and
scikit-learn):

```bash
python scripts/run_pipeline.py                                  # K-Means, recommended k
python scripts/run_pipeline.py --algorithm dbscan
python scripts/run_pipeline.py --algorithm hierarchical --k 4 --linkage average
python scripts/run_pipeline.py --data data/Mall_Customers.csv   # your own file
```

Results go to `outputs/`: `segmented_customers.csv`, `segment_profiles.csv`, `cluster_scan.csv`,
`segment_insights.md`.

## Dashboard tour

| Tab | Purpose |
|---|---|
| Data | Row counts, missing values, correlations, what preprocessing changed, before/after scaling of any feature |
| Choose parameters | Elbow and silhouette curves, Davies-Bouldin and Calinski-Harabasz table, dendrogram, DBSCAN k-distance plot and eps / min_samples search |
| Clusters | Quality metrics, PCA or feature scatter (2-D and 3-D), segment sizes, silhouette per customer, metric distributions |
| Segments and actions | Customer share vs spend share, profile table, deviation heatmap, radar chart, segment mix by city tier / state / payment mode / category, and a playbook per segment |
| Compare algorithms | All three methods side by side with an agreement matrix (Adjusted Rand Index) |
| Export | Segmented customers, profile table and insight report |

The interface is built for analysts: a header with the active dataset and method, a KPI strip that is always
visible (customers, segments, silhouette, unassigned share, estimated annual spend), white chart panels on a quiet
grey canvas, IBM Plex Sans typography, and one accent colour. Segment colours are identical in every chart and in
the playbook. Amounts use Indian formatting (₹3,512, ₹12.35 lakh, ₹3.21 Cr). Columns stack on narrow screens.
The theme lives in `assets/style.css` and `.streamlit/config.toml`.

## Method

**Preprocessing.** Duplicates are removed by customer id. Missing values are imputed (median by
default) or the rows dropped. Outliers are clipped to 1.5 x IQR fences for the *model input only*;
the cleaned table used for profiling keeps true values, so segment averages are in real units.
Features are then scaled (z-score by default) because distance-based algorithms are dominated by
whichever column has the largest range.

**Choosing k.** K-Means is run for k = 2..10. The elbow is the point of the inertia curve farthest
from the line joining its ends. The recommended k is the silhouette maximum, except that the elbow
k wins when its silhouette is within 0.02 of the best (simpler model, same quality).

**DBSCAN.** `min_samples` defaults to twice the number of features (minimum 4). The dashboard
searches a grid of eps and min_samples values and pre-selects the setting with the best silhouette
that has at least two clusters and at most 20% noise. Customers DBSCAN cannot place are shown as
"Unassigned (noise)".

**Naming segments and actions.** For each cluster, every metric is expressed as a z-score against
the whole customer base. Two scores are derived:

- *value*: income, average order value, estimated annual spend
- *engagement*: spending score, order frequency, and recency (fewer days since the last order counts as more engaged)

The pair places the cluster in a 3 x 3 grid (for example high value and high engagement is
"Champions"; high value and low engagement is "Affluent, low engagement"). Each archetype carries
a short action plan and a metric to track, and each segment also lists the three traits that
differ most from average, with numbers. If the columns are not recognisable, segments are called
"Segment n" and only the data-driven traits are shown.

## Results on the bundled sample

Default settings, K-Means:

| Check | Result |
|---|---|
| Elbow k / best-silhouette k | 5 / 5 |
| Silhouette, Davies-Bouldin, Calinski-Harabasz | 0.378, 0.989, 591.0 |
| Agreement with the generator's hidden groups (ARI) | about 0.95 (K-Means), about 0.96 (Hierarchical, ward) |
| Champions | 20% of customers, about 53% of estimated spend |
| Dormant / at risk | about 16% of customers, about 2% of estimated spend |

DBSCAN on the same data finds three dense groups plus about 8% noise, because the overlapping
groups form density bridges; the Compare tab shows this honestly. Single or average linkage can
produce a few tiny clusters; ward linkage is the best default for compact customer groups.

The data is synthetic, so these numbers show the pipeline works; they are not claims about real
customers.

## Project structure

```
customer-segmentation/
├── app.py                      Streamlit dashboard (UI only)
├── segmentation/
│   ├── config.py               paths and constants
│   ├── schema.py               header normalisation and column-role detection
│   ├── data_loader.py          CSV loading/validation, sample generator
│   ├── preprocessing.py        cleaning, scaling, normalisation
│   ├── clustering.py           K-Means, DBSCAN, Hierarchical, DBSCAN helpers
│   ├── evaluation.py           elbow, silhouette, recommended k, agreement
│   ├── insights.py             profiles, segment names, actions, report
│   ├── formatting.py           rupee / lakh / crore formatting
│   ├── visualization.py        Plotly figures
│   └── utils.py                knee detection
├── assets/style.css            dashboard theme
├── scripts/
│   ├── generate_dataset.py     rebuild data/customers.csv
│   └── run_pipeline.py         command-line pipeline
├── tests/                      unit tests (and a dashboard smoke test)
├── data/                       sample dataset + download instructions
├── docs/                       demo guide, GitHub upload guide
├── .streamlit/config.toml      theme
├── .github/workflows/tests.yml continuous integration
└── requirements.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # or: python -m unittest discover -s tests -t .
```

Covered: rupee formatting, the Indian sample's structure, loader validation and the Mall Customers layout, every preprocessing option, all three
algorithms (K-Means and Hierarchical must recover the generator's known groups), the elbow and
silhouette logic, noise handling, insight naming and report output. When Plotly and Streamlit are
installed, the suite also builds every figure and runs `app.py` headlessly with Streamlit's
`AppTest` for each algorithm.

## Limitations

- Hierarchical clustering is limited to 5,000 customers (it needs O(n^2) memory); use K-Means beyond that.
- Silhouette is estimated on a 5,000-row sample for large datasets.
- At least two numeric features are required; categorical columns can be added on top of them.
- Segment names come from transparent rules, not a learned model; review them before acting.
- Recommended actions are starting hypotheses. Validate with a holdout group before spending budget.    

## Copyright

© 2026 Pritam Rabha. All rights reserved.

## Licence

MIT, see `LICENSE`.
