# DRIVEIQ v1 — Archived UK Datasets

These CSV files were used in **DRIVEIQ Version 1.0** and contain UK used-car listings
with prices in British Pounds (GBP).

## Files

| File | Brand | Records |
|---|---|---|
| `audi.csv` | Audi | ~10,668 |
| `bmw.csv` | BMW | ~10,781 |
| `ford.csv` | Ford | ~17,965 |
| `hyundai.csv` | Hyundai | ~4,860 |
| `mercedes.csv` | Mercedes-Benz | ~13,119 |
| `skoda.csv` | Skoda | ~6,267 |
| `toyota.csv` | Toyota | ~6,738 |
| `vw.csv` | Volkswagen | ~15,157 |

## Why Archived

In DRIVEIQ 2.0 (Phase 2), these UK datasets were replaced with native Indian
used-car datasets sourced from CarDekho and other Indian platforms. The UK data
had several fundamental limitations:

1. **Prices were in GBP**, requiring conversion via arbitrary market multipliers.
2. **Mileage was in miles**, creating unit confusion with the Indian-facing UI.
3. **Fuel efficiency was in MPG**, not the Indian-standard km/l.
4. **Market dynamics** (brand popularity, resale patterns, taxation) did not
   reflect the Indian used-car market.

## Rollback

If a rollback to v1 data is needed, these files can be restored to `data/`
and the v1 `data_processing.py` can be used to load them.
