# ETo Smart Predictor - Streamlit Edition

This package is a web-ready Streamlit version of the ETo Smart Predictor.
It replaces the original Tkinter desktop GUI so it can run on Streamlit Cloud.

## Included
- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies for local run or Streamlit Cloud
- `models/README.md` - where to place trained model files
- `assets/README.md` - optional static assets folder
- `.streamlit/config.toml` - Streamlit theme settings
- `.gitignore`

## Important
The trained model files are **not included** here unless you add them manually.
The app expects the trained models inside the `models/` folder.

## Expected model naming
The app searches for names like:
- `XGB1.joblib`, `xgb1.joblib`, `xgb_1.joblib`, `xgb-1.joblib`
- `LGB1.joblib`, `CGB1.joblib`, etc.
- Native XGBoost files such as `xgb1.json`, `xgb1.ubj`, `xgb1.model`

The 4 cases are mapped as:
1. Central Valley (Stn. 39) - RCP 4.5
2. Central Valley (Stn. 39) - RCP 8.5
3. Imperial Valley (Stn. 87) - RCP 4.5
4. Imperial Valley (Stn. 87) - RCP 8.5

Across 3 algorithms, that is 12 trained models in total.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment
1. Upload this folder to your GitHub repo.
2. Put all trained model files inside the `models/` folder.
3. Deploy the repo on Streamlit Cloud.
4. Set the entrypoint to `app.py`.

## Notes
- Tkinter is not used here because it does not work on Streamlit Cloud.
- If an XGBoost `.joblib` file fails to load, exporting the model as `.json` or `.ubj` is recommended.
