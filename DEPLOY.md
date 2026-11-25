# Deploying the Autoformer app to Streamlit Community Cloud

This folder is the complete app, ready for GitHub: app.py, model code, data, 4 trained checkpoints
(ETTm2 96 and 336, Exchange 96 and 720), results/comparison.csv, and requirements.txt (CPU-only PyTorch).

## 1. Put this folder on GitHub (use GitHub Desktop: the website only takes files up to 25 MB)
1. Create a free account at github.com if you do not have one.
2. Install GitHub Desktop (desktop.github.com) and sign in.
3. File -> Add local repository -> choose this folder -> "create a repository" -> Create repository.
4. Commit to main (message: "Autoformer Streamlit app").
5. Publish repository -> untick "Keep this code private" (Community Cloud needs access; public is simplest) -> Publish.
   The 4 checkpoints (63 MB each) upload in a few minutes. GitHub may warn about files over 50 MB: that is fine,
   only files over 100 MB are blocked.

## 2. Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and sign in with GitHub.
2. Create app -> "Deploy a public app from GitHub".
3. Repository: <your-name>/autoformer_streamlit_cloud, Branch: main, Main file path: app.py.
4. Advanced settings -> Python version 3.12.
5. Pick an app URL (e.g. autoformer-forecast) -> Deploy. The first build takes several minutes (PyTorch install).

## 3. If something goes wrong
- "Oh no" / over resource limits: Manage app -> Reboot. The app keeps at most 3 models in memory.
- Install error about torch: open Manage app -> logs and send the error text.
- A model missing in the dropdown: its .pth file name must stay exactly as Colab saved it.

## Adding Phase 3 models later
Copy the .pth files you want from runs_phase3/checkpoints into checkpoints/ (and table_mse.csv, table_mae.csv,
mse_by_horizon.png into results/), commit and push in GitHub Desktop. The app redeploys by itself.
Keep the repository small: each extra model adds 63 MB.
