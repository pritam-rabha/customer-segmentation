# Demonstration guide

A 10-minute walkthrough for a viva or project demo. Practise it once with the bundled sample.

## Before you start

```bash
source .venv/bin/activate
python -m pytest -q                 # optional: show that the tests pass
streamlit run app.py
```

Keep a terminal open next to the browser; step 9 uses it.

## Script

| Min | Do | Say | What the audience should see |
|---|---|---|---|
| 0:00 | Open the **Data** tab | "The goal is to split customers into groups we can treat differently. The sample has 1,000 Indian retail customers with the usual defects." | 1,000 rows, 173 missing cells, 160 outlier values clipped; the KPI strip stays visible on every tab |
| 1:00 | Point at the missing-values and correlation charts | "Before clustering I fixed gaps and clipped extremes, because K-Means and DBSCAN are distance based." | Which columns had gaps; income and spending behaviour correlations |
| 2:00 | Pick *Avg order value (₹)* under "Inspect a feature" | "Scaling puts every feature on the same footing; otherwise income would swamp age." | Left: raw values on very different ranges. Right: scaled |
| 3:00 | Open **Choose parameters** | "How many groups? The elbow bends at 5 and silhouette peaks at 5, so both methods agree." | Elbow marker at k = 5; silhouette peak about 0.38 |
| 4:00 | Scroll to the dendrogram and the DBSCAN plots | "Hierarchical clustering gives a tree; DBSCAN needs eps and min_samples, which I chose from the k-distance knee and a grid search." | Dendrogram cut line; heatmap of clusters and noise per setting |
| 5:00 | Open **Clusters** | "Five segments, silhouette 0.38 and Davies-Bouldin about 1.0." | Metrics row, PCA scatter with five colours |
| 6:00 | Switch the projection to *Two chosen features*; choose income and spending | "In real units the groups separate along income and spending." | Visible clusters; open the 3-D expander and rotate |
| 7:00 | Open **Segments and actions** | "Each cluster is named from how it differs from the average customer, then given actions." | Revenue-concentration chart, heatmap, radar, then "Champions": 20% of customers, about 53% of spend. Change "Segment mix by" to city tier or payment mode (Champions are about 95% UPI or card and half Tier 1; Dormant customers are 41% cash on delivery and 42% Tier 3) |
| 8:00 | Read one playbook aloud (Dormant or Affluent, low engagement) | "These are hypotheses to test with a holdout group." | Traits with numbers, three actions, one metric |
| 9:00 | In the sidebar choose **DBSCAN**, then open **Compare algorithms** | "The methods disagree, and that is informative: DBSCAN merges the overlapping groups and flags about 8% as noise." | Table plus agreement matrix |
| 10:00 | Open **Export** and click a download | "Marketing gets the segmented customer list and a report." | CSV and Markdown downloads |

## Terminal demonstration (no browser needed)

```bash
python scripts/run_pipeline.py
cat outputs/segment_insights.md
```

## Live experiments that make good talking points

| Change | Expected effect |
|---|---|
| Preprocessing: Outliers = *Keep as is* | Silhouette and segment shapes shift; extreme baskets pull centroids |
| Preprocessing: Scaling = *Min-max* | Same k, slightly different boundaries |
| Hierarchical with *single* linkage | One huge cluster plus tiny ones (chaining), so ward is the right default |
| DBSCAN with a very small eps | Everything becomes noise; the dashboard warns instead of failing |
| Upload `Mall_Customers.csv` (see `data/README.md`) | Works with three features; names come from income and spending |

## Likely questions

**Why scale before clustering?** Distance-based methods treat a column with a bigger range as more important.

**Why does the dashboard recommend k = 5 when silhouette alone might pick another value?** The rule takes the silhouette peak but prefers the elbow k when scores are within 0.02, so the simpler model wins ties.

**Why is DBSCAN's silhouette sometimes higher while its result is worse?** Noise points are excluded from the score. With many points marked noise, the remaining dense cores score well. That is why the unassigned share is shown next to it.

**Are the segment names learned?** No. They come from transparent rules on value and engagement z-scores, so every label can be explained.

**Is the sample real?** No, it is synthetic and reproducible. The loader and tests also support real files such as Mall Customers.

**What would you do next?** Test the recommended actions with a holdout group, refresh the segments monthly, and track how many customers migrate between segments.
