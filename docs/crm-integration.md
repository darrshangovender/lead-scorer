# CRM integration

The reference implementation emits scored leads to a JSON file. In production at the client we push them straight into HubSpot every 6 hours so reps see the score in their normal pipeline view, with no extra dashboard to check.

This doc sketches how that piece works — the reference repo intentionally stops short of a live CRM integration so it can be run by anyone without credentials.

## Pattern

```
scheduled job (every 6h)
  -> pull newly-arrived + recently-updated leads from Postgres
  -> FeatureBuilder.transform
  -> XGBRanker.predict_proba
  -> upsert lead_id, score, scored_at into the CRM's custom property
```

The lead's `score` becomes a HubSpot custom property (`lead_score_ml`) which the sales team can sort and filter on inside their existing list views. No new tool to learn, which matters a lot for adoption.

## HubSpot stub

```python
# Sketch only — needs hubspot-api-client + a private app token.
from hubspot import HubSpot
from hubspot.crm.contacts import SimplePublicObjectInput

client = HubSpot(access_token=HUBSPOT_TOKEN)

def push_scores(scored: list[dict]) -> None:
    """scored = [{'lead_id': '123', 'score': 0.87}, ...] where lead_id is the HubSpot contact id."""
    for row in scored:
        client.crm.contacts.basic_api.update(
            contact_id=row["lead_id"],
            simple_public_object_input=SimplePublicObjectInput(
                properties={
                    "lead_score_ml": str(round(row["score"] * 100)),
                    "lead_score_ml_updated_at": row["scored_at"],
                }
            ),
        )
```

In practice you batch this — HubSpot's batch endpoint takes up to 100 contacts per call and is rate-limited at 100 requests/10s for a private app, so a few thousand leads per cycle is fine.

## Salesforce stub

```python
# Sketch only — needs simple-salesforce + an OAuth connected app.
from simple_salesforce import Salesforce

sf = Salesforce(username=USER, password=PWD, security_token=TOKEN)

def push_scores(scored: list[dict]) -> None:
    payload = [
        {"Id": row["lead_id"], "Lead_Score_ML__c": round(row["score"] * 100)}
        for row in scored
    ]
    # Bulk API for anything larger than a few hundred records.
    sf.bulk.Lead.update(payload)
```

## What this reference does instead

`Scorer.score_to_json` writes the ranked leads to a file. To wire it into a real CRM, replace that step with whichever stub above matches your stack. Everything upstream — feature build, model, calibration — stays the same.

## Operational notes

- **Idempotency.** Score writes must be idempotent. Re-running the job for the same window should produce the same property values, not duplicates.
- **Score age.** Always write a `scored_at` timestamp alongside the score. A 4-day-old score is a different thing from a 4-minute-old one, and sales managers want to see that.
- **Soft launch.** When rolling out, write to a parallel property (`lead_score_ml_shadow`) for a couple of weeks and let sales managers compare it against their gut before the score appears on rep screens.
