# Delhi Smoke — does Punjab's stubble burning drive Delhi's winter air?

Every October and November, farmers in Punjab and Haryana burn rice stubble to clear fields for the next crop. Delhi sits south-east of them, and its air collapses at the same time each year.

But *correlated in November* is not the same as *caused by*. Delhi's winter boundary layer also collapses, trapping local traffic and industrial emissions regardless of what is burning 250 km away. This project measures how much of Delhi's winter PM2.5 is actually attributable to crop fires — and finds that the honest answer only appears after correcting for a confound that hides it.

**[Live demo](#)** · Built with Python, pandas, statsmodels, SciPy, Streamlit

---

## The headline finding

A naive test says smoke transport **does not matter**. A correctly specified model says it matters **a great deal**. Both use the same data.

| Test | Result |
|---|---|
| High-fire days, wind from fires vs. across | median PM2.5 189 vs 194 — **no effect** (Mann-Whitney p = 0.70) |
| Regression, `fire_count × wind alignment` | **p < 0.001**, interaction coefficient exceeds the main effect |

The reason for the contradiction is the interesting part.

---

## Why the naive test fails

North-westerly winds do two opposing things at once:

1. They **carry smoke** from Punjab towards Delhi.
2. Because they tend to be stronger, they also **ventilate** Delhi's own emissions.

In the data, high-fire days with aligned wind average **8.4** wind speed against **6.2** on cross-wind days. So any comparison that splits on wind *direction* without holding wind *speed* constant is measuring both effects simultaneously — and they cancel.

Holding wind speed roughly constant makes this explicit. Restricting to calm, high-fire days only, the gap between aligned and cross-wind days **collapses from 25 µg/m³ to 1** (240 vs 239). The apparent direction effect in the raw comparison was wind speed all along.

---

## The result

Incremental OLS on 610 days across 5 burning seasons (Sept–Dec, 2015–2019):

| Model | R² | Gain |
|---|---|---|
| Calendar only (`doy`, `doy²`, year) | 0.442 | +0.442 |
| + Weather (wind speed, temperature) | 0.484 | +0.043 |
| + Fire count | 0.585 | **+0.101** |
| + Fire × wind alignment | 0.606 | +0.021 |

Key coefficients from the full model (n = 610):

| Term | Coefficient | p-value | Reading |
|---|---|---|---|
| `fire_count` | 0.0173 | < 0.001 | ~1.7 µg/m³ per 100 detections, controlling for season and weather |
| `alignment` | −0.61 | 0.938 | Wind direction alone does nothing — correctly, since direction is meaningless with no fires |
| `fire_count : alignment` | **0.0275** | **< 0.001** | Fires hit ~2.6× harder when wind blows from the fire belt |
| `wind_speed` | −8.84 | < 0.001 | Ventilation: each unit of wind speed clears ~8.8 µg/m³ |

Fires add **+0.101 to R² after** the model already knows the date, the year, the wind and the temperature. That gain is not seasonality.

The interaction structure is exactly what a transport mechanism should look like: direction is irrelevant on its own, and matters only in proportion to how much is burning.

---

## Method

**1. Fire detections.** NASA FIRMS VIIRS S-NPP (375 m), bounding box `73.5, 27.5, 77.8, 32.6` covering Punjab and Haryana. Filtered to nominal/high confidence, aggregated to daily counts and total Fire Radiative Power.

**2. Air quality.** Daily Delhi PM2.5 from CPCB, via the Kaggle *Air Quality Data in India* dataset.

**3. Weather.** Hourly wind speed, wind direction, temperature and precipitation for Delhi from the Open-Meteo reanalysis archive, aggregated to daily.

**4. Wind alignment feature.** The great-circle bearing from Delhi to the centre of the burning belt is **322°** (north-west). For each day, alignment is `cos(Δ)` between the wind's origin direction and that bearing, clipped to `[0, 1]`. A value of 1 means wind blowing straight from the fires towards Delhi.

**5. Modelling.** Incremental OLS with an interaction term, comparing R² gains block by block.

### One implementation detail worth flagging

**Wind direction is circular, and averaging it arithmetically is silently wrong.** The mean of 350° and 10° is 0° — due north — but `numpy.mean` returns 180°, pointing the wind in exactly the opposite direction. Daily wind direction here is computed as a **speed-weighted circular mean** (averaging unit vectors), so that stronger hours carry more weight. The notebook prints both the correct and incorrect answers as a sanity check.

---

## Repository

```
├── app.py                       # Streamlit dashboard
├── delhi_smoke_analysis.ipynb   # full analysis, runnable in Colab
├── requirements.txt
├── analysis_daily.csv           # daily joined table (generated)
└── fire_points.csv              # detection coordinates for the map (generated)
```

**Run locally**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Reproduce the analysis** — open `delhi_smoke_analysis.ipynb` in Google Colab, upload the FIRMS CSVs and `city_day.csv`, and run the cells in order. A free [FIRMS API key](https://firms.modaps.eosdis.nasa.gov/api/map_key/) is needed only for live data, not for the historical archive.

---

## Limitations

These matter, and the conclusions above should be read against them.

- **Observational, not causal.** This is association under statistical controls, not a controlled experiment.
- **Daily means at a single point** are a crude proxy for atmospheric transport across ~250 km. Proper attribution uses back-trajectory models such as HYSPLIT.
- **Fire detections are not emissions.** Detection counts depend on satellite overpass timing and cloud cover. They proxy burning; they do not measure smoke released.
- **Local sources are absorbed, not measured.** Delhi's traffic, industry, construction and waste burning are captured only indirectly through the calendar and year terms.
- **Five seasons is a modest sample**, and the 2016 year-dummy differs significantly from the 2015 baseline, indicating real between-season variation not fully explained by the model.

---

## Data sources

| Source | Use | Licence / access |
|---|---|---|
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | VIIRS S-NPP active fire detections | Free, open |
| [CPCB via Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) | Delhi daily PM2.5 | Free, open |
| [Open-Meteo](https://open-meteo.com/) | Historical weather reanalysis | Free, no key required |

---

*Built by Nandani Goyal. Findings are exploratory and intended as a demonstration of analytical methodology, not as an authoritative source on air quality attribution.*
