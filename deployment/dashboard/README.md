# Predictive Maintenance of Aircraft Engines — Deployment

Deployment work for the MLOps-based DataScience Process Model. Estimates the Remaining Useful Life (RUL) of turbofan
engines and serves predictions through a dashboard.

## Structure

    deployment/
      dashboard/    Frontend: fleet health monitor (HTML/JS + Chart.js)
      backend/      FastAPI prediction service (in progress)

## Live dashboard

 **https://aircraft-rul.netlify.app/**
 

## Project status

**Deployment is ongoing.** The current focus is the monitoring dashboard, which
is functional and deployed. It runs on fixed sample data drawn from the real
test set, so the interface and all interactions work end to end.

The trained model is not yet connected. All predictions route through a single
function, `predictRUL()`, which is currently a mock. This is a deliberate design
choice: the frontend was built and polished first, and the model is wired in by
replacing that one function with a call to the prediction API — nothing else
changes.

### Done
- [x] Monitoring dashboard (fleet replay, engine detail, manual predict, synthetic stream)
- [x] Deployed and publicly accessible

### In progress
- [ ] FastAPI backend serving the trained model
- [ ] Connect dashboard to live predictions
- [ ] Containerization and cloud deployment

### Possible extensions
The scope may grow as the deployment matures — additional monitoring,
retraining triggers, and drift detection are under consideration.

## Dashboard (current state)

A single-page dashboard running on fixed sample data drawn from the real test
set, so it works with no backend yet. Features:
- Monitoring strip: predictions served, latency, healthy/critical counts
- Operations: fleet replay + click-through engine detail
- Test bench: manual predict + synthetic stream (labelled simulated)
  
