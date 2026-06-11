# -*- coding: utf-8 -*-
"""Wariant 2: walidacja leave-one-day-out + diagnostyka per epizod + hybryda."""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import cross_val_scores, wygladz, ocena_zdarzeniowa, baseline_regulowy
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

zb = pd.read_pickle(ROOT / "work" / "zbior_uczacy.pkl")
n_dni = zb["dzien"].nunique()
oof, models = cross_val_scores(zb, n_splits=n_dni)  # leave-one-day-out
oof = wygladz(zb, oof)
m = ~np.isnan(oof)
print(f"LODO ({n_dni} foldow) PR-AUC: {average_precision_score(zb.y[m], oof[m]):.3f}, "
      f"ROC-AUC: {roc_auc_score(zb.y[m], oof[m]):.3f}")

rb = baseline_regulowy(zb)

print("\n--- LightGBM LODO, ocena zdarzeniowa ---")
for prog in [0.5, 0.7, 0.9]:
    r = ocena_zdarzeniowa(zb, oof, prog)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

# hybryda: alarm gdy model >= prog LUB regula aktywna
print("\n--- Hybryda (model OR regula) ---")
for prog in [0.7, 0.9]:
    hyb = np.maximum(oof, rb)
    r = ocena_zdarzeniowa(zb, hyb, prog)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

# diagnostyka per epizod
print("\n--- per epizod (max score modelu / reguly w oknie przedzdarzeniowym) ---")
zb2 = zb.copy(); zb2["score"] = oof; zb2["rb"] = rb
for e in sorted(zb2.loc[zb2.epizod_id >= 0, "epizod_id"].unique()):
    okno = zb2[zb2.epizod_id == e]
    print(f"epizod {e}: {okno.index.min()} .. {okno.index.max()} (mlyn MW{e//100}), "
          f"max_model={okno.score.max():.3f}, max_regula={okno.rb.max():.0f}, n_min={len(okno)}")
np.save(ROOT / "work" / "oof_lodo.npy", oof)
