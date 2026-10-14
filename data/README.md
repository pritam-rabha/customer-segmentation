# Data

## Bundled sample: `customers.csv`

1,000 synthetic Indian retail / e-commerce customers, generated reproducibly by `scripts/generate_dataset.py`
(seed 42). It is synthetic on purpose: it works offline, has no privacy issues, and contains the
defects real data has, so preprocessing has genuine work to do.

| Column | Meaning |
|---|---|
| `customer_id` | Unique id, e.g. `IN00001` (ignored by the models) |
| `customer_name` | Synthetic Indian name (display only) |
| `gender` | Female / Male (optional categorical feature) |
| `age` | Years |
| `city`, `state`, `city_tier` | 26 cities across 15 states and union territories, grouped as Tier 1 / 2 / 3 |
| `annual_income_lakh` | Annual income in ₹ lakh |
| `spending_score` | 1-100 score assigned by the retailer |
| `tenure_months` | Months since first order |
| `orders_per_year` | Order frequency |
| `avg_order_value_inr` | Mean basket value in ₹ |
| `days_since_last_order` | Recency |
| `online_share_pct` | Share of orders placed online |
| `preferred_payment` | UPI, Credit/Debit card, Cash on delivery or Wallet |
| `top_category` | Fashion, Electronics, Grocery, Home & Kitchen, Beauty & Personal Care |

City, state, tier, payment and category are not used to form clusters by default; they are used to
describe the segments afterwards (the "Segment mix by" chart). Tick `city_tier`, `preferred_payment`
or `top_category` under "Categorical features" in the sidebar to cluster on them as well.

Injected defects: about 4% missing values in six columns and about 1.5% extreme outliers
(bulk or corporate buyers, entry slips). Five overlapping behavioural groups generate the data, so
clusters are realistic rather than perfectly separated.

To rebuild or resize it:

```bash
python scripts/generate_dataset.py --rows 5000 --seed 7
```

If the file is deleted, the app and scripts recreate it automatically.

## Using a real dataset

Any CSV with one row per customer and at least two numeric columns works. Headers are normalised
automatically (`Annual Income (k$)` becomes `annual_income_k`), the id column is detected, and
column meanings (income, spending, recency, ...) are recognised from their names so segments get
business labels. Unrecognised columns still cluster; segments are then named "Segment 1, 2, ...".

### Option A: Mall Customers (classic, 200 rows)

1. Open <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>
   (confirm the page title is "Customer Segmentation Tutorial in Python").
2. Download `Mall_Customers.csv`, or with the Kaggle CLI (needs a free API token in
   `~/.kaggle/kaggle.json`):

   ```bash
   pip install kaggle
   kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data --unzip
   ```
3. Run it:

   ```bash
   python scripts/run_pipeline.py --data data/Mall_Customers.csv
   # or choose "Upload a CSV" in the dashboard sidebar
   ```

The loader is covered by a unit test that uses this exact column layout.

### Option B: Online Retail II (transaction data, UCI)

<https://archive.ics.uci.edu/dataset/502/online+retail+ii> contains transactions, not customers.
Aggregate it to one row per customer first (recency = days since last invoice, frequency =
number of invoices, monetary = total spend), save that table as a CSV, then load it as above.
Columns named `recency`, `frequency` and `monetary` are recognised automatically.

Do not commit data you are not allowed to share; add such files to `.gitignore`.
