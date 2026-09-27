# FLASHGUARD — Actual-data training package

This package builds a supervised flash-flood model from **real sources**:

1. Official NWIC Uttarakhand hourly telemetry rainfall, 1991–2020.
2. The published Uttarakhand historical flash-flood inventory (122 events, 1970–2020).

The historical inventory is an event inventory, not a station-level sensor log. It contains dates and textual locations, so the pipeline only uses events whose district can be explicitly extracted from the source. It does **not invent coordinates**.

### Label definition
At rainfall observation time **T**, `flood_label=1` when a documented flash-flood event occurs in the **same district during the next 24 hours**; otherwise 0. This prevents using rainfall after the event as an input.

Because the source inventory generally gives an event date rather than an exact event hour and does not provide station coordinates, these are **district/date-matched event labels**, not ground-truth station-level flood labels. This limitation must be stated in the SIH presentation.

### Train
```bash
pip install -r requirements.txt
python train_model.py
```

Outputs:
- `flashguard_training_dataset.csv`
- `flashguard_model.pkl`
- `model_metrics.json`

### Render
Build command:
```bash
pip install -r requirements.txt && python train_model.py
```
Start command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The official NWIC portal currently lists the 1991–2020 telemetry dataset and identifies it as Uttarakhand Department rainfall data. The historical inventory is the Rana & Mahanta dataset published on Zenodo.
