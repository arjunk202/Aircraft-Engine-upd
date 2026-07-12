# Backend (FastAPI) — in progress

Will serve:
- POST /predict     — sensor window -> { rul, status }
- GET  /monitoring  — live counters
- GET  /next        — replay: next cycle for the fleet

The trained model (.joblib) and feature-columns file load here at startup.
