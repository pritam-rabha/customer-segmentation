# Uploading to GitHub

## One-time setup

1. Create a free account at <https://github.com> and install Git (<https://git-scm.com/downloads>).
2. Tell Git who you are (once per computer):

   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```

## Publish the project

1. On GitHub choose **New repository**. Name it `customer-segmentation`, leave it **empty**
   (no README, no .gitignore, no licence; they already exist locally), and click **Create repository**.
2. In the project folder:

   ```bash
   git init -b main
   git add .
   git status                     # check: no .venv/, no outputs/*.csv
   git commit -m "Customer segmentation: K-Means, DBSCAN, hierarchical, Streamlit dashboard"
   git remote add origin https://github.com/<your-username>/customer-segmentation.git
   git push -u origin main
   ```

3. When Git asks for a password, use a **personal access token** (GitHub, Settings,
   Developer settings, Personal access tokens), or sign in with the GitHub CLI: `gh auth login`.

Same thing with the GitHub CLI:

```bash
gh repo create customer-segmentation --public --source=. --push
```

## After the first push

- Open the **Actions** tab: the `tests` workflow runs the test suite on every push.
- Take 3 screenshots (Choose parameters, Clusters, Segments and actions), save them in
  `docs/screenshots/`, and add them to the README with
  `![Clusters](docs/screenshots/clusters.png)`.
- Add a short description and the topics `customer-segmentation`, `clustering`, `streamlit`, `scikit-learn`
  in the repository's **About** panel.
- Optional: deploy free on Streamlit Community Cloud (<https://share.streamlit.io>): choose the
  repository, set the main file to `app.py`, and deploy.

## Everyday workflow

```bash
git add -A
git commit -m "Describe the change"
git push
```

## Before submitting

- [ ] Repository link opens without logging in (Settings, General, Danger zone shows it as Public if required)
- [ ] README renders correctly and the clone-and-run steps work on a clean folder
- [ ] `python -m pytest -q` passes
- [ ] No large data files, passwords or tokens were committed
