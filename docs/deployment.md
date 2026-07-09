# Live Deployment Runbook (Render.com)

This document provides step-by-step instructions for deploying the containerized Bangalore House Price Predictor full-stack application to Render.com's free cloud tier.

---

## Prerequisites
1. **GitHub Account**: You must push your local project repository to GitHub.
2. **Render Account**: Create a free account at [Render.com](https://render.com/).
3. **Model Artifact**: Confirm that `models/model_pipeline.joblib` exists in your workspace. (Note: Since we gitignore the raw model binary `models/*.joblib`, we must ensure it is present during Docker build. For Render, we can commit the joblib file or, if it is small, track it. In our case, `model_pipeline.joblib` is ~12.5MB. Render reads from GitHub, so we should commit the trained model to GitHub so the container builds with the serialized pipeline. Ensure you run `git add -f models/model_pipeline.joblib` to force commit it so that Render can build the container with the model included!).

---

## Option 1: One-Click Blueprint Deployment (Recommended)
Render blueprints read the `render.yaml` configuration in the root of the project to automatically configure all resources.

1. Push your repository to GitHub. Ensure the serialized model is committed (`git add -f models/model_pipeline.joblib` and commit it).
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click the **New +** button in the top right and select **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically detect the `render.yaml` file. Verify the service name (`bangalore-house-price-predictor`) and configuration.
6. Click **Apply** (or **Approve**). Render will spin up the container, run the health checks, and expose a public URL (e.g., `https://bangalore-house-price-predictor.onrender.com`).

---

## Option 2: Manual Dashboard Deployment
If you prefer to configure the Web Service manually:

1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your GitHub repository.
4. Configure the Web Service settings:
   - **Name**: `bangalore-house-price-predictor`
   - **Region**: Select the region closest to you (e.g., Oregon or Frankfurt).
   - **Branch**: `main`
   - **Runtime**: `Docker` (Render will automatically locate the root `Dockerfile` and build it).
   - **Instance Type**: `Free`
5. Expand the **Advanced** section:
   - Add Environment Variable:
     - Key: `PORT`
     - Value: `8000`
6. Click **Create Web Service**. Render will start building the Docker image from the source code.

---

## Post-Deployment Verification
Once Render reports that the service is live:

1. Copy the public URL provided by Render.
2. Check API Health:
   - Navigate to `https://<your-subdomain>.onrender.com/health` in your browser.
   - Confirm it returns `{"status": "healthy"}`.
3. Access Frontend:
   - Navigate to `https://<your-subdomain>.onrender.com/`.
   - Confirm the dark-themed glassmorphism interface loads.
   - Enter a test property (e.g. Hebbal, 1800 sqft, 3 BHK, 3 Bath) and click **Calculate Estimated Price** to verify the model predictions and SHAP interpretations serve correctly.
