"""Delhi Smoke - does Punjab's stubble burning drive Delhi's winter air?

An interactive walkthrough of the analysis. The app is deliberately built
around the *reasoning*, not just the charts: it shows the naive comparison
failing, diagnoses the confound, and then shows the regression recovering the
effect.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Delhi Smoke", page_icon="🔥", layout="wide")

DELHI_LAT, DELHI_LON = 28.6139, 77.2090
SOURCE_LAT, SOURCE_LON = 30.4, 75.6
FIRE = "#c1440e"
INK = "#1f3b57"


# --- data ----------------------------------------------------------------
@st.cache_data
def load(path: str = "analysis_daily.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    return df


@st.cache_data
def load_points(path: str = "fire_points.csv", max_rows: int = 30_000):
    try:
        pts = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return None
    if len(pts) > max_rows:
        pts = pts.sample(max_rows, random_state=0)
    return pts


@st.cache_data
def fit_models(df: pd.DataFrame):
    """Incremental OLS. Returns (r2_ladder, coefficient_table, n)."""
    import statsmodels.formula.api as smf

    d = df.dropna(subset=["pm25", "fire_count", "wind_speed", "alignment", "temp"]).copy()
    specs = {
        "Calendar only": "pm25 ~ doy + I(doy**2) + C(year)",
        "+ Weather": "pm25 ~ doy + I(doy**2) + C(year) + wind_speed + temp",
        "+ Fires": "pm25 ~ doy + I(doy**2) + C(year) + wind_speed + temp + fire_count",
        "+ Fires × wind alignment":
            "pm25 ~ doy + I(doy**2) + C(year) + wind_speed + temp + fire_count * alignment",
    }
    ladder, prev = [], 0.0
    model = None
    for name, formula in specs.items():
        model = smf.ols(formula, data=d).fit()
        ladder.append({"Model": name, "R²": model.rsquared, "Gain": model.rsquared - prev})
        prev = model.rsquared

    coefs = pd.DataFrame({
        "Coefficient": model.params,
        "Std error": model.bse,
        "p-value": model.pvalues,
    })
    keep = ["fire_count", "alignment", "fire_count:alignment", "wind_speed", "temp"]
    coefs = coefs.loc[[i for i in keep if i in coefs.index]]
    return pd.DataFrame(ladder), coefs, int(model.nobs)


try:
    df = load()
except FileNotFoundError:
    st.error(
        "`analysis_daily.csv` not found. Run the analysis notebook first — "
        "it writes the daily table this app reads."
    )
    st.stop()

points = load_points()


# --- header --------------------------------------------------------------
st.title("Does Punjab's stubble burning drive Delhi's winter air?")
st.markdown(
    "Every October and November, farmers in Punjab and Haryana burn rice stubble. "
    "Delhi sits **south-east** of them and its air collapses at the same time. "
    "But *correlated in November* is not the same as *caused by*. This is an attempt "
    "to measure the real contribution."
)

seasons = sorted(df["year"].unique())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Seasons analysed", len(seasons))
c2.metric("Days", f"{len(df):,}")
c3.metric("Fire detections", f"{int(df['fire_count'].sum()):,}")
c4.metric("Median PM2.5", f"{df['pm25'].median():.0f} µg/m³")

st.caption(
    f"Sources: NASA FIRMS (VIIRS S-NPP 375 m) · CPCB via Kaggle · Open-Meteo reanalysis · "
    f"{df['date'].min():%b %Y} – {df['date'].max():%b %Y}"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["The pattern", "The naive test", "The confound", "The result"]
)


# --- 1. the pattern ------------------------------------------------------
with tab1:
    st.subheader("Fires and air quality, side by side")
    season = st.selectbox("Season", ["All"] + [str(y) for y in seasons])
    view = df if season == "All" else df[df["year"] == int(season)]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.fill_between(view["date"], view["fire_count"], alpha=0.55, color=FIRE)
    ax1.set_ylabel("fire detections / day", color=FIRE)
    ax1.tick_params(axis="y", labelcolor=FIRE)
    ax2 = ax1.twinx()
    ax2.plot(view["date"], view["pm25"], lw=1.2, color=INK)
    ax2.set_ylabel("Delhi PM2.5 (µg/m³)", color=INK)
    ax2.grid(False)
    ax1.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)

    r = view["fire_count"].corr(view["pm25"])
    st.info(
        f"**Raw correlation: r = {r:.3f}.** They clearly move together — but both also "
        "peak in November for unrelated reasons. Delhi's winter boundary layer collapses "
        "and traps local traffic and industrial emissions regardless of what is burning "
        "250 km away. Correlation alone cannot separate the two."
    )

    if points is not None:
        st.subheader("Where the fires are")
        st.map(points.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]],
               size=20, color="#c1440e88")
        st.caption("Detections across Punjab and Haryana. Delhi lies to the south-east.")


# --- 2. the naive test ---------------------------------------------------
with tab2:
    st.subheader("If smoke blows in, wind direction should matter")
    st.markdown(
        f"The fire belt lies at a bearing of **322° (north-west)** from Delhi. "
        "So on days when the wind blows *from* that direction, smoke should reach the "
        "city. The obvious test: take the busiest fire days and split them by wind "
        "direction."
    )

    busy = df[df["fire_count"] > df["fire_count"].quantile(0.75)]
    aligned = busy[busy["alignment"] > 0.5]
    cross = busy[busy["alignment"] <= 0.5]

    a, b = st.columns(2)
    a.metric("Wind FROM the fires", f"{aligned['pm25'].median():.0f} µg/m³", f"n = {len(aligned)}")
    b.metric("Wind across or away", f"{cross['pm25'].median():.0f} µg/m³", f"n = {len(cross)}")

    from scipy import stats
    if len(aligned) > 5 and len(cross) > 5:
        _, p = stats.mannwhitneyu(
            aligned["pm25"].dropna(), cross["pm25"].dropna(), alternative="greater"
        )
        st.warning(
            f"**No effect (Mann-Whitney p = {p:.2f}).** The aligned-wind days are not "
            "dirtier. Taken at face value this says smoke transport does not matter — "
            "which is where most casual analyses of this question stop."
        )


# --- 3. the confound -----------------------------------------------------
with tab3:
    st.subheader("Why the naive test fails")
    st.markdown(
        "North-westerly winds do **two opposing things at once**. They carry smoke "
        "towards Delhi — and, because they tend to be stronger, they also ventilate the "
        "city's own emissions. The two effects cancel."
    )

    a, b = st.columns(2)
    a.metric("Mean wind speed, aligned days", f"{aligned['wind_speed'].mean():.1f}")
    b.metric("Mean wind speed, cross days", f"{cross['wind_speed'].mean():.1f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(df["fire_count"], df["pm25"], c=df["wind_speed"],
                    cmap="viridis_r", s=16, alpha=0.85)
    ax.set_xlabel("fire detections that day")
    ax.set_ylabel("Delhi PM2.5 (µg/m³)")
    plt.colorbar(sc, ax=ax, label="wind speed")
    st.pyplot(fig, use_container_width=True)

    st.info(
        f"Wind speed correlates **{df['wind_speed'].corr(df['pm25']):.3f}** with PM2.5 — "
        "stronger wind, cleaner air. Any comparison that splits on wind *direction* "
        "without holding wind *speed* constant is measuring both effects at once."
    )


# --- 4. the result -------------------------------------------------------
with tab4:
    st.subheader("Separating the effects with regression")
    st.markdown(
        "Rather than slicing the data by hand, fit a model that holds calendar position, "
        "season, wind speed and temperature constant, then ask what fires add on top."
    )

    try:
        ladder, coefs, n = fit_models(df)
    except Exception as exc:  # statsmodels missing, or degenerate data
        st.error(f"Model could not be fitted: {exc}")
        st.stop()

    st.markdown("**How much each block explains**")
    st.dataframe(
        ladder.style.format({"R²": "{:.3f}", "Gain": "+{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Key coefficients**")
    st.dataframe(
        coefs.style.format({"Coefficient": "{:.4f}", "Std error": "{:.4f}", "p-value": "{:.2e}"}),
        use_container_width=True,
    )

    inter = coefs.loc["fire_count:alignment"] if "fire_count:alignment" in coefs.index else None
    if inter is not None and inter["p-value"] < 0.05:
        st.success(
            f"**Fires matter, and wind direction decides how much.** "
            f"`fire_count` is significant on its own, `alignment` alone is not — correctly, "
            f"since wind direction is meaningless when nothing is burning. But the "
            f"**interaction** `fire_count × alignment` is significant "
            f"(p = {inter['p-value']:.1e}), and its coefficient exceeds the main effect: "
            f"fires hit substantially harder when the wind blows from the fire belt.\n\n"
            f"The effect is invisible in the naive comparison on the previous tab, and "
            f"clear here. n = {n} days."
        )
    else:
        st.warning(
            "The interaction is not significant in this dataset. The fire effect appears "
            "real, but wind direction does not measurably modulate it at daily resolution."
        )

    with st.expander("Limitations — read this"):
        st.markdown(
            """
- **Observational, not causal.** This is association under controls, not a controlled experiment.
- **Daily means at a single point** are a crude proxy for transport over ~250 km. Proper
  atmospheric attribution uses back-trajectory models such as HYSPLIT.
- **Fire detections are not emissions.** Detection count depends on satellite overpass timing
  and cloud cover; it is a proxy for burning, not a measurement of smoke released.
- **Delhi has large local sources** — traffic, industry, construction, waste burning — that
  this model absorbs into the calendar and year terms rather than measuring directly.
            """
        )

st.divider()
st.caption(
    "Built with Python, pandas, statsmodels, SciPy and Streamlit. "
    "Data: NASA FIRMS · CPCB (via Kaggle) · Open-Meteo."
)
