# Create sample CSV and requirements.txt for the reinsurance Streamlit app
import pandas as pd
import numpy as np
from datetime import datetime

# ----- Create sample reinsurance CSV -----
rng = np.random.default_rng(123)
periods = pd.period_range('2024Q1', '2025Q4', freq='Q').to_timestamp()
cedants = ['CedantA', 'CedantB']
lobs = ['Property Cat', 'Casualty']
perils = ['Wind', 'Liability']
regions = ['EU', 'NA']

rows = []
for dt in periods:
    for ced in cedants:
        for lob, peril, region in zip(lobs, perils, regions):
            gross = rng.normal(50, 8)
            ceded = gross * rng.uniform(0.15, 0.45)
            earned = gross * rng.uniform(0.75, 0.95)
            count = rng.poisson(120 if lob == 'Property Cat' else 80)
            exposure = rng.integers(800, 1600)
            severity = rng.lognormal(mean=9.4 if lob == 'Property Cat' else 9.2, sigma=0.35) / 1e6
            incurred = count * severity
            paid = incurred * rng.uniform(0.6, 0.9)
            ibnr = incurred * rng.uniform(0.06, 0.18)
            rbns = incurred * rng.uniform(0.05, 0.15)
            acq = earned * rng.uniform(0.08, 0.14)
            adm = earned * rng.uniform(0.05, 0.09)
            inv = gross * rng.uniform(0.01, 0.03)
            scr = earned * rng.uniform(0.28, 0.42)
            own = scr * rng.uniform(1.25, 1.9)
            rows.append([dt, ced, lob, peril, region, gross, ceded, earned, incurred, paid, ibnr, rbns, acq, adm, count, exposure, scr, own, inv])

cols = [
    "date","cedant","lob","peril","region",
    "gross_premium","ceded_premium","earned_premium",
    "incurred_claims","paid_claims","ibnr","rbns",
    "acq_expense","adm_expense","claims_count","exposure",
    "scr","own_funds","investment_income"
]

df = pd.DataFrame(rows, columns=cols)
csv_path = "/mnt/data/reinsurance_sample.csv"
df.to_csv(csv_path, index=False)

# ----- Create requirements.txt -----
requirements = """streamlit==1.37.0
pandas>=2.2.2
numpy>=1.26.4
plotly>=5.22.0
statsmodels>=0.14.2
scikit-learn>=1.4.2
python-dateutil>=2.9.0
matplotlib>=3.8.4
reportlab>=4.1.0
"""
req_path = "/mnt/data/requirements.txt"
with open(req_path, "w", encoding="utf-8") as f:
    f.write(requirements)

# Provide file paths
(csv_path, req_path)
