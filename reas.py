# app_reassurance.py — IA & Réassurance (KPI + Prévisions 3 ans + Stress tests + PDF)
# -----------------------------------------------------------------------------------
# - Import CSV/Excel (avec cartographie des colonnes)
# - KPI techniques/financiers/risque (Loss, Expense, Combined, Operating, Solvency…)
# - Prévisions jusqu’à 3 ans (SARIMAX), par Global / LOB / Région
# - Stress tests (Δ fréquence, Δ sévérité, événement CAT)
# - Vues Portefeuille (répartition LOB/région, fréquence × sévérité)
# - Exports CSV & Rapport PDF exécutif (métriques + graphe Combined)
# -----------------------------------------------------------------------------------

import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Réassurance — KPI & Prévisions 3 ans", page_icon="🛡️", layout="wide")

# ---------------------------------------
# Référentiel de colonnes & exigences
# ---------------------------------------
SCHEMA = {
    "date": ["date", "period", "periode", "month", "quarter", "year"],
    "lob": ["lob", "branche", "line_of_business"],
    "region": ["region", "zone", "pays", "geography"],
    "cedant": ["cedant", "cedente", "ceding_company"],
    "gross_premium": ["gross_premium", "primes_brutes", "gwp"],
    "ceded_premium": ["ceded_premium", "primes_cedees", "ceded"],
    "earned_premium": ["earned_premium", "primes_acquises", "ep"],
    "incurred_claims": ["incurred_claims", "sinistres_encourus", "icl"],
    "paid_claims": ["paid_claims", "sinistres_payes", "pcl"],
    "ibnr": ["ibnr", "reserves_ibnr"],
    "rbns": ["rbns", "reserves_rbns"],
    "acq_expense": ["acq_expense", "frais_acquisition"],
    "adm_expense": ["adm_expense", "frais_admin", "g&a"],
    "investment_income": ["investment_income", "produits_financiers"],
    "claims_count": ["claims_count", "nombre_sinistres"],
    "exposure": ["exposure", "exposition", "policies", "risks"],
    "scr": ["scr", "exigence_capital"],
    "own_funds": ["own_funds", "fonds_propres"],
}
REQUIRED_BASE = ["date", "earned_premium", "incurred_claims"]

# ---------------------------------------
# Helpers data
# ---------------------------------------
def _infer_date_col(s: pd.Series) -> pd.Series:
    """Tente de parser une colonne date. Fallback: année -> 1er janvier."""
    try:
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.6:
            return parsed
    except Exception:
        pass
    if s.dtype.kind in "if":
        return pd.to_datetime(s.astype(int).astype(str) + "-01-01", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def make_demo(periods=16, seed=42, freq="Q"):
    """Jeu de données de démonstration trimestriel par défaut."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2022Q1", periods=periods, freq=freq).to_timestamp()
    lobs = ["Property Cat", "Casualty"]
    regions = ["EU", "NA"]
    rows = []
    for dt in idx:
        for lob, region in zip(lobs, regions):
            gwp = rng.normal(50, 8)
            ced = gwp * rng.uniform(0.15, 0.45)
            ep = gwp * rng.uniform(0.75, 0.95)
            cnt = rng.poisson(110 if lob == "Property Cat" else 85)
            expo = rng.integers(900, 1600)
            sev = rng.lognormal(mean=9.35 if lob == "Property Cat" else 9.1, sigma=0.35) / 1e6
            inc = float(cnt) * float(sev)
            paid = inc * rng.uniform(0.6, 0.9)
            ibnr = inc * rng.uniform(0.06, 0.18)
            rbns = inc * rng.uniform(0.05, 0.15)
            acq = ep * rng.uniform(0.08, 0.14)
            adm = ep * rng.uniform(0.05, 0.09)
            inv = gwp * rng.uniform(0.01, 0.03)
            scr = ep * rng.uniform(0.28, 0.42)
            own = scr * rng.uniform(1.25, 1.9)
            rows.append([
                dt, "CedantA", lob, region, gwp, ced, ep, inc, paid, ibnr, rbns,
                acq, adm, cnt, expo, scr, own, inv
            ])
    return pd.DataFrame(rows, columns=[
        "date", "cedant", "lob", "region", "gross_premium", "ceded_premium", "earned_premium",
        "incurred_claims", "paid_claims", "ibnr", "rbns", "acq_expense", "adm_expense",
        "claims_count", "exposure", "scr", "own_funds", "investment_income"
    ])

def auto_map_columns(df: pd.DataFrame):
    """Détecte automatiquement les correspondances colonnes utilisateur -> schéma."""
    mapping = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for key, aliases in SCHEMA.items():
        found = None
        for a in aliases:
            if a in cols_lower:
                found = cols_lower[a]
                break
        mapping[key] = found
    return mapping

def compute_kpis(d: pd.DataFrame) -> pd.DataFrame:
    """Calcule les ratios KPI techniques/financiers/risque."""
    df = d.copy()
    ep = df["earned_premium"].replace(0, np.nan)
    gwp = df.get("gross_premium", pd.Series(np.nan, index=df.index))
    ced = df.get("ceded_premium", pd.Series(0.0, index=df.index))

    df["loss_ratio"] = df["incurred_claims"] / ep
    df["acq_ratio"] = df.get("acq_expense", 0) / ep
    df["adm_ratio"] = df.get("adm_expense", 0) / ep
    df["expense_ratio"] = df["acq_ratio"].fillna(0) + df["adm_ratio"].fillna(0)
    df["combined_ratio"] = df["loss_ratio"].fillna(0) + df["expense_ratio"].fillna(0)
    df["operating_ratio"] = df["combined_ratio"] - (df.get("investment_income", 0) / ep)
    df["cession_ratio"] = ced / gwp.replace(0, np.nan)
    df["retention_ratio"] = (gwp - ced) / gwp.replace(0, np.nan)

    if {"claims_count", "exposure"}.issubset(df.columns):
        df["frequency"] = df["claims_count"] / df["exposure"].replace(0, np.nan)
    if {"incurred_claims", "claims_count"}.issubset(df.columns):
        df["severity"] = df["incurred_claims"] / df["claims_count"].replace(0, np.nan)

    if {"ibnr", "rbns"}.issubset(df.columns):
        df["total_reserves"] = df["ibnr"].fillna(0) + df["rbns"].fillna(0)
        df["reserve_coverage"] = df["total_reserves"] / df["incurred_claims"].replace(0, np.nan)

    if {"scr", "own_funds"}.issubset(df.columns):
        df["solvency_ratio"] = df["own_funds"] / df["scr"].replace(0, np.nan)

    return df

def aggregate_kpis(d: pd.DataFrame, by=["date"]) -> pd.DataFrame:
    """Agrège par dimensions et recalcule les KPI au niveau agrégé."""
    grp = d.groupby(by, dropna=False).agg({
        "gross_premium": "sum", "ceded_premium": "sum", "earned_premium": "sum",
        "incurred_claims": "sum", "paid_claims": "sum", "ibnr": "sum", "rbns": "sum",
        "acq_expense": "sum", "adm_expense": "sum", "investment_income": "sum",
        "claims_count": "sum", "exposure": "sum", "scr": "sum", "own_funds": "sum"
    }).reset_index()
    grp = compute_kpis(grp)
    return grp

def sarimax_forecast(ts: pd.Series, steps: int, order=(1,1,1), seasonal=(0,1,1,4)) -> pd.Series:
    """Prévision SARIMAX avec fallback naïf si historique insuffisant."""
    ts = ts.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if ts.shape[0] < max(24, steps):
        last = ts.iloc[-1] if ts.shape[0] else 0.0
        idx = pd.date_range(datetime.today(), periods=steps, freq="MS")
        return pd.Series([last] * steps, index=idx)
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps).predicted_mean
        return fc
    except Exception:
        last = ts.iloc[-1] if ts.shape[0] else 0.0
        idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return pd.Series([last] * steps, index=idx)

def add_month_start(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les dates sur le début de mois pour homogénéiser les séries temporelles."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return out

def download_button(df: pd.DataFrame, filename: str):
    """Lien de téléchargement CSV sans écrire sur disque côté serveur."""
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    st.markdown(
        f'<a download="{filename}" href="data:file/csv;base64,{b64}">📥 Télécharger {filename}</a>',
        unsafe_allow_html=True
    )

# ---------------------------------------
# UI — Header
# ---------------------------------------
st.title("🛡️ Réassurance — KPI & Prévisions sur 3 ans")
st.caption("Analyse technique, capital & scénarios — avec export PDF.")

# ---------------------------------------
# UI — Sidebar (données & mapping)
# ---------------------------------------
with st.sidebar:
    st.subheader("📥 Données")
    up = st.file_uploader("Fichier CSV/Excel", type=["csv", "xlsx", "xls"])
    freq = st.selectbox("Fréquence", ["Mensuelle", "Trimestrielle", "Annuelle"], index=1)
    horizon_years = st.slider("Horizon de prévision (ans)", 1, 5, 3)
    do_demo = st.toggle("Utiliser données démo", value=(up is None))

    if do_demo:
        df_raw = make_demo(periods=16, freq=("Q" if freq == "Trimestrielle" else ("A" if freq == "Annuelle" else "M")))
    else:
        if up is not None:
            if up.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(up)
            else:
                df_raw = pd.read_excel(up)
        else:
            df_raw = pd.DataFrame()

    st.caption("Cartographie colonnes")
    if df_raw.empty:
        st.info("Charge un fichier ou active la démo.")
        mapping = {k: None for k in SCHEMA}
    else:
        auto = auto_map_columns(df_raw)
        mapping = {}
        for key in SCHEMA.keys():
            mapping[key] = st.selectbox(
                key, [None] + list(df_raw.columns),
                index=([None] + list(df_raw.columns)).index(auto.get(key)) if auto.get(key) in df_raw.columns else 0
            )

if df_raw.empty:
    st.stop()

# ---------------------------------------
# Préparation des données
# ---------------------------------------
ren = {v: k for k, v in mapping.items() if v is not None}
df = df_raw.rename(columns=ren)

missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Colonnes indispensables manquantes: {missing}")
    st.stop()

df["date"] = _infer_date_col(df["date"])
df = add_month_start(df)
df_kpi = compute_kpis(df)

# ---------------------------------------
# Header metrics (global)
# ---------------------------------------
agg = aggregate_kpis(df_kpi, by=["date"]).sort_values("date")
last_row = agg.iloc[-1]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Primes acquises", f"{last_row['earned_premium']:,.2f}")
c2.metric("Sinistres encourus", f"{last_row['incurred_claims']:,.2f}")
c3.metric("Loss Ratio", f"{last_row['loss_ratio']*100:,.1f}%")
c4.metric("Expense Ratio", f"{last_row['expense_ratio']*100:,.1f}%")
c5.metric("Combined Ratio", f"{last_row['combined_ratio']*100:,.1f}%")

# ---------------------------------------
# Tabs
# ---------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 KPI", "📈 Prévisions 3 ans", "🧪 Stress Tests", "🗺️ Portefeuille", "📤 Export"]
)

# --- KPI ---
with tab1:
    groupers = []
    if "lob" in df_kpi.columns:
        groupers.append("lob")
    if "region" in df_kpi.columns:
        groupers.append("region")
    selected = st.multiselect("Regrouper par", groupers, default=groupers)

    g = aggregate_kpis(df_kpi, by=["date"] + selected)
    st.caption("Ratios: Loss/Expense/Combined/Operating, Cession/Rétention, Fréquence/Sévérité, Réserves, Solvabilité.")

    fig = px.line(g, x="date", y=["loss_ratio", "expense_ratio", "combined_ratio"],
                  color=selected[0] if selected else None, markers=True)
    st.plotly_chart(fig, use_container_width=True)

    if "solvency_ratio" in g.columns:
        fig2 = px.line(g, x="date", y="solvency_ratio",
                       color=selected[0] if selected else None, markers=True, title="Solvency Ratio")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(g.round(4))

# --- Prévisions ---
with tab2:
    st.subheader("Prévisions (jusqu’à 3 ans)")
    target = st.selectbox("Série à projeter",
                          ["earned_premium", "incurred_claims", "combined_ratio", "loss_ratio", "expense_ratio"], index=0)
    by_dim = st.selectbox("Projection par dimension", ["Global"] + [c for c in ["lob", "region"] if c in df_kpi.columns])

    def project_series(df_in: pd.DataFrame) -> pd.DataFrame:
        d = aggregate_kpis(df_in, by=["date"]).sort_values("date").set_index("date")
        ts = d[target]
        if freq == "Mensuelle":
            order, seas, steps = (1, 1, 1), (0, 1, 1, 12), 12 * int(horizon_years)
            idx = pd.date_range(d.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        elif freq == "Trimestrielle":
            order, seas, steps = (1, 1, 1), (0, 1, 1, 4), 4 * int(horizon_years)
            idx = pd.period_range(d.index[-1].to_period('Q') + 1, periods=steps, freq="Q").to_timestamp()
        else:
            order, seas, steps = (1, 1, 0), (0, 1, 1, 1), int(horizon_years)
            idx = pd.date_range(d.index[-1] + pd.offsets.YearBegin(1), periods=steps, freq="YS")
        fc = sarimax_forecast(ts, steps, order, seas)
        fc.index = idx
        return pd.DataFrame({
            "date": list(ts.index) + list(fc.index),
            target: list(ts.values) + list(fc.values),
            "type": ["historique"] * len(ts) + ["prévision"] * len(fc)
        })

    if by_dim == "Global":
        pr = project_series(df_kpi)
        fig = px.line(pr, x="date", y=target, color="type")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pr.tail(12))
    else:
        vals = sorted(df_kpi[by_dim].dropna().unique())
        tabs = st.tabs(vals)
        for i, v in enumerate(vals):
            with tabs[i]:
                pr = project_series(df_kpi[df_kpi[by_dim] == v])
                fig = px.line(pr, x="date", y=target, color="type", title=f"{by_dim} = {v}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pr.tail(12))

# --- Stress tests ---
with tab3:
    st.subheader("Scénarios de stress")
    colA, colB, colC = st.columns(3)
    with colA:
        shock_freq = st.slider("Δ Fréquence (%)", -30, 100, 10)
    with colB:
        shock_sev = st.slider("Δ Sévérité (%)", -30, 200, 20)
    with colC:
        cat_factor = st.slider("Événement CAT (x sinistres sur une période)", 1.0, 5.0, 1.2, step=0.1)

    dates_available = sorted(agg["date"].unique())
    cat_date = st.selectbox("Période CAT", dates_available, index=len(dates_available) - 1)

    d_str = df_kpi.copy()
    if "claims_count" in d_str.columns:
        d_str["claims_count"] = d_str["claims_count"] * (1 + shock_freq / 100.0)
    d_str["incurred_claims"] = d_str["incurred_claims"] * (1 + shock_sev / 100.0)
    d_str.loc[d_str["date"] == pd.to_datetime(cat_date), "incurred_claims"] = \
        d_str.loc[d_str["date"] == pd.to_datetime(cat_date), "incurred_claims"] * cat_factor

    g_base = aggregate_kpis(df_kpi, by=["date"])
    g_str = aggregate_kpis(d_str, by=["date"])

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(g_base, x="date", y="combined_ratio", title="Baseline — Combined"), use_container_width=True)
    c2.plotly_chart(px.line(g_str, x="date", y="combined_ratio", title="Stress — Combined"), use_container_width=True)

    st.markdown("**Solvabilité**")
    if {"scr", "own_funds"}.issubset(df_kpi.columns):
        base = g_base.assign(solvency=lambda x: x["own_funds"] / x["scr"].replace(0, np.nan))
        stress = g_str.assign(solvency=lambda x: x["own_funds"] / x["scr"].replace(0, np.nan))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=base["date"], y=base["solvency"], name="Baseline"))
        fig3.add_trace(go.Scatter(x=stress["date"], y=stress["solvency"], name="Stress"))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Ajoutez SCR et Fonds propres pour l'analyse solvabilité.")

# --- Portefeuille ---
with tab4:
    st.subheader("Structure du portefeuille")
    dims = [c for c in ["lob", "region"] if c in df_kpi.columns]

    if "lob" in dims:
        g1 = aggregate_kpis(df_kpi, by=["lob"]).sort_values("earned_premium", ascending=False)
        st.plotly_chart(px.pie(g1, names="lob", values="earned_premium", title="Répartition EP par LOB"),
                        use_container_width=True)

    if "region" in dims:
        g2 = aggregate_kpis(df_kpi, by=["region"]).sort_values("earned_premium", ascending=False)
        st.plotly_chart(px.bar(g2, x="region", y="earned_premium", title="EP par région"),
                        use_container_width=True)

    if {"frequency", "severity"}.issubset(df_kpi.columns):
        g3 = aggregate_kpis(df_kpi, by=["lob"] if "lob" in dims else ["region"])
        st.plotly_chart(px.scatter(
            g3, x="frequency", y="severity", size="earned_premium",
            color=("lob" if "lob" in dims else "region"),
            title="Fréquence vs Sévérité (taille = EP)"),
            use_container_width=True
        )
    else:
        st.info("Ajoutez claims_count et exposure pour fréquence/sévérité.")

# --- Export ---
with tab5:
    st.subheader("Exports")

    agg_global = aggregate_kpis(df_kpi, by=["date"])
    st.markdown("**KPI agrégés (global)**")
    st.dataframe(agg_global.round(4))
    download_button(agg_global.round(6), "kpi_reassurance_global.csv")

    if "lob" in df_kpi.columns:
        agg_lob = aggregate_kpis(df_kpi, by=["date", "lob"])
        st.markdown("**KPI par LOB**")
        st.dataframe(agg_lob.round(4))
        download_button(agg_lob.round(6), "kpi_reassurance_par_lob.csv")

    if "region" in df_kpi.columns:
        agg_reg = aggregate_kpis(df_kpi, by=["date", "region"])
        st.markdown("**KPI par région**")
        st.dataframe(agg_reg.round(4))
        download_button(agg_reg.round(6), "kpi_reassurance_par_region.csv")

    # --- Export PDF (rapport exécutif) ---
    import matplotlib.pyplot as plt
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import cm

    def _fig_combined_png(df_line: pd.DataFrame) -> bytes:
        d = df_line.sort_values("date")
        buf = BytesIO()
        plt.figure()
        plt.plot(d["date"], d["combined_ratio"])
        plt.title("Combined Ratio — Baseline")
        plt.xlabel("Date")
        plt.ylabel("Combined")
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return buf.read()

    def build_pdf(agg_df: pd.DataFrame, filename: str = "rapport_reassurance.pdf") -> bytes:
        buf = BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        margin = 2 * cm
        y = height - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Réassurance — Rapport exécutif")
        y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        y -= 24
        last = agg_df.sort_values("date").iloc[-1]
        metrics = [
            ("Primes acquises", f"{last['earned_premium']:,.2f}"),
            ("Sinistres encourus", f"{last['incurred_claims']:,.2f}"),
            ("Loss Ratio", f"{last['loss_ratio']*100:,.1f}%"),
            ("Expense Ratio", f"{last['expense_ratio']*100:,.1f}%"),
            ("Combined Ratio", f"{last['combined_ratio']*100:,.1f}%"),
        ]
        for k, v in metrics:
            c.drawString(margin, y, f"- {k} : {v}")
            y -= 14
        y -= 10
        img_bytes = _fig_combined_png(agg_df)
        img_buf = BytesIO(img_bytes)
        img_w, img_h = 14 * cm, 7 * cm
        c.drawInlineImage(img_buf, margin, y - img_h, img_w, img_h)
        y -= img_h + 12
        c.setFont("Helvetica", 9)
        c.drawString(margin, y, "Notes: Ratios calculés sur base agrégée. Voir app pour segments.")
        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()

    st.markdown("---")
    st.subheader("Export PDF du rapport exécutif")
    if st.button("Générer le PDF"):
        pdf_bytes = build_pdf(agg_global)
        st.download_button(
            label="📄 Télécharger le rapport PDF",
            data=pdf_bytes,
            file_name="rapport_reassurance.pdf",
            mime="application/pdf",
        )

# ---------------------------------------
# Référentiel formules (aide)
# ---------------------------------------
with st.expander("📘 Référentiel formules"):
    st.markdown(
        "- Loss = Incurred / EP\n"
        "- Expense = (Acq+Adm) / EP\n"
        "- Combined = Loss + Expense\n"
        "- Operating = Combined − (Investment / EP)\n"
        "- Cession = Ceded / Gross ; Rétention = (Gross−Ceded) / Gross\n"
        "- Fréquence = N sinistres / Exposition ; Sévérité = Incurred / N sinistres\n"
        "- Couverture réserves = (IBNR+RBNS)/Incurred ; Solvency = OwnFunds / SCR"
    )

st.caption("© 2025 — Outil KPI Réassurance (Streamlit)")
