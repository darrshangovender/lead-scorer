<div align="center">

# Sales-Intelligence Lead Scorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![Status](https://img.shields.io/badge/Status-Production-success)](#)
[![Top-decile precision](https://img.shields.io/badge/Top--decile%20precision-~0.6-059669)](#results)

</div>

---

> Lead-scoring system for a B2B services client. Ingests CRM and website-engagement data into Postgres, engineers behavioural and firmographic features, trains a logistic-regression baseline plus a gradient-boosted model, and exposes scores back into the CRM via API. Helps sales prioritise the top 20% of inbound leads.

**Stack:** Python · scikit-learn · XGBoost · PostgreSQL · FastAPI · GitHub Actions

---

## The problem

The client's sales team had hundreds of inbound leads per week and the same number of working hours as the rest of us. Calls were going out in roughly first-in-first-out order, which meant the best leads sometimes got cold while a rep worked through low-intent ones first.

The ask: a number on each lead that says "call this one before that one." Not "predict revenue" — just *order them right*.

## What it does

1. **Pulls** lead records, CRM activity, and website-engagement events into a Postgres staging warehouse.
2. **Engineers features** — firmographic (industry, headcount, region) + behavioural (page visits, content downloads, return frequency, days-since-first-touch).
3. **Scores** each lead with two models:
   - **Logistic regression baseline** — fast, interpretable, lets sales managers see the coefficients and trust them.
   - **XGBoost model** — better-ranking, higher AUC, used as the production scorer.
4. **Pushes scores back** to the CRM (HubSpot in this case) via API every 6 hours so reps see scores in their normal pipeline view.

## Why two models

The logistic regression isn't the production scorer — it's there for trust and debugging.

When a sales manager asks "why does this lead have a score of 87?", I can show them the LR coefficients and a SHAP plot for the XGBoost model side-by-side. If they tell different stories, that's a signal something is off (data drift, leakage). If they agree, the manager learns to trust the box.

Lots of ML systems fail in production not because the model is bad but because the humans don't trust it. Two models with a tight feedback loop solves a real organisational problem.

## Feature highlights

The features that mattered most by SHAP:

- `pages_per_session_p90`
- `days_since_last_visit`
- `pricing_page_visits_30d`
- `company_headcount_band`
- `repeat_visit_count`
- `email_engagement_score`

`pricing_page_visits_30d` was the single strongest behavioural feature — unsurprising in retrospect.

## Architecture

```
HubSpot ─┐
         ├─▶ Postgres staging ─▶ feature build ─▶ score
website ─┘                                       │
                                                 ▼
                                       Postgres scores table
                                                 │
                                                 ▼
                                          push to HubSpot
                                          (every 6h via API)
```

## Repo structure

```
.
├── ingest/
│   ├── hubspot.py
│   └── web_events.py
├── features/
│   └── build.py
├── models/
│   ├── baseline_lr.py
│   └── xgb.py
├── score/
│   └── push_to_crm.py
├── api/
│   └── main.py        # FastAPI for ad-hoc score lookups
└── eval/
    └── ranking_metrics.py
```

## Why ranking metrics, not classification metrics

Sales doesn't care whether a lead is "predicted to close" with > 50% probability. They care whether the top 50 leads on a Monday morning include most of the deals that will close.

So the model is evaluated on:

- **Precision @ top-decile** (target: 0.55+)
- **NDCG@50**
- **Lift over random** at the top-100 cut

This dictates loss-function and threshold choices very differently from a yes/no classifier.

## Results

- Sales reps now work from the score-ranked queue.
- Top-decile precision held at ~0.6 in production over the first quarter.
- Conversion-per-rep-hour on inbound leads improved meaningfully (numbers under NDA).

## Local setup

```bash
docker compose up -d
uv sync
uv run python ingest/hubspot.py --since-days 90
uv run python features/build.py
uv run python models/xgb.py --train
uv run uvicorn api.main:app --reload
```

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)
