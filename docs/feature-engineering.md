# Feature engineering

This reference implementation uses a deliberately small feature set so the pipeline stays readable. The production system at the client has roughly 4x as many features — what's missing here is called out at the bottom.

## Features used

| Feature | Type | Why it's in |
|---|---|---|
| `industry` | one-hot | Conversion rates differ by 2-3x across industries. SaaS and fintech buyers convert faster than manufacturing in our data. |
| `region` | one-hot | Time-zone overlap with the sales team correlates with reachability, which correlates with conversion. |
| `company_size` | one-hot (`micro`/`small`/`mid`/`large`/`enterprise`) | The interesting non-linearity. Mid and large are the sweet spot — micro lacks budget, enterprise has long procurement cycles. Headcount on its own is bucketed because the raw number is log-distributed and the bucket signal is what sales actually uses. |
| `days_since_last_touch` | numeric | Recency proxy. Stronger signal than first-touch date because it captures whether the lead is still warm. |
| `email_engagement_score` | numeric [0, 1] | Aggregated open + click rate from the marketing automation platform. Beta-distributed in real data. |
| `page_views_30d` | numeric | Behavioural intent signal. In the real system this is split by page type (pricing, blog, case study), which matters a lot more — see "what's missing." |
| `demo_requested` | binary | The strongest single feature. A self-served demo request is the closest thing to a buying signal you get pre-conversation. |
| `seniority_score` | ordinal 1-7 | Title-derived seniority. Higher seniority correlates with decision authority. |

## What's deliberately missing from this reference

The real production system also uses:

- **Pricing-page visits** broken out from generic page views. In the live SHAP plot this is the single highest-importance feature; generic page views are much weaker.
- **Content downloads** by content type (whitepaper, case study, ROI calculator).
- **Repeat visit count** and **pages-per-session p90** — proxies for genuine evaluation behaviour vs idle browsing.
- **Intent data** from third-party providers (Bombora, G2). These are expensive feeds and out of scope for a public reference.
- **Account-level features** rolled up from multiple leads at the same company — most B2B deals involve a buying committee, so scoring leads in isolation underweights the account signal.
- **Time-of-day / day-of-week** features for last touch — captures rep availability.

The reference dataset uses `page_views_30d` as a stand-in for the bundle of behavioural signals. This is the main reason headline metrics here are lower than what the production system achieves.

## Encoding choices

- **One-hot, not target encoding**, for the categoricals. Target encoding leaks if you're not careful with cross-validation, and the cardinality here is low enough that one-hot is fine.
- **No scaling for XGBoost**, but the LR baseline scales internally. Tree models are scale-invariant; LR is not.
- **Unseen categories at score time** fall through as all-zero rows for the affected one-hot block. The numeric features still carry signal, so the model degrades gracefully rather than crashing.
