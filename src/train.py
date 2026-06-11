# -*- coding: utf-8 -*-
"""Trening i ocena modelu wczesnego ostrzegania przed zapchaniem młyna.

Model: LightGBM (klasyfikacja binarna, silna nierównowaga klas).
Walidacja: GroupKFold po dniach kalendarzowych (bez wycieku czasowego).
Ocena minutowa: PR-AUC / ROC-AUC.
Ocena zdarzeniowa: ile epizodów wykryto w oknie ostrzegania, wyprzedzenie [min],
fałszywe alarmy na dobę (sklejane w serie >= 3 kolejnych minut powyżej progu).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ["y", "mlyn", "epizod_id", "dzien"]


def cross_val_scores(zb: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    X = zb.drop(columns=META)
    y = zb["y"].values
    groups = zb["dzien"].values
    oof = np.full(len(zb), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    models = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pos = y[tr].sum()
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0,
            scale_pos_weight=min((len(tr) - pos) / max(pos, 1), 30),
            random_state=seed, verbose=-1)
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        models.append(model)
    return oof, models


def wygladz(zb: pd.DataFrame, oof: np.ndarray, okno: int = 5) -> np.ndarray:
    """Mediana krocząca score w obrębie młyna - tłumi pojedyncze piki."""
    s = pd.Series(oof, index=zb.index)
    out = np.full(len(zb), np.nan)
    for m in zb["mlyn"].unique():
        mask = (zb["mlyn"] == m).values
        out[mask] = s[mask].rolling(okno, min_periods=1).median().values
    return out


def ocena_zdarzeniowa(zb: pd.DataFrame, oof: np.ndarray, prog: float,
                      min_serii: int = 3) -> dict:
    zb = zb.copy()
    zb["score"] = oof
    zb["alarm"] = zb["score"] >= prog

    # wykrywalność epizodów + wyprzedzenie (alarm = seria >= min_serii kolejnych minut)
    wykryte, wyprzedzenia = 0, []
    epizody = sorted(zb.loc[zb.epizod_id >= 0, "epizod_id"].unique())
    for e in epizody:
        okno = zb[zb.epizod_id == e]
        seria = okno["alarm"].rolling(min_serii, min_periods=min_serii).min()
        alarmy = okno[seria == 1]
        if len(alarmy):
            wykryte += 1
            start_zdarzenia = okno.index.max() + pd.Timedelta(minutes=1)
            # początek serii = pierwszy alarm w serii (cofamy o długość serii - 1)
            t_alarm = alarmy.index.min() - pd.Timedelta(minutes=min_serii - 1)
            wyprzedzenia.append((start_zdarzenia - t_alarm).total_seconds() / 60)

    # fałszywe alarmy: serie alarmowe poza oknami przedzdarzeniowymi
    fp = zb[(zb.alarm) & (zb.epizod_id < 0)].sort_values(["mlyn"])
    serie = 0
    for m, g in fp.groupby("mlyn"):
        if len(g) == 0:
            continue
        idx = g.index.sort_values()
        dl, prev = 1, idx[0]
        for t in idx[1:]:
            if (t - prev) <= pd.Timedelta(minutes=2):
                dl += 1
            else:
                if dl >= min_serii:
                    serie += 1
                dl = 1
            prev = t
        if dl >= min_serii:
            serie += 1
    dni = zb["dzien"].nunique()
    return {"prog": prog, "epizody": len(epizody), "wykryte": wykryte,
            "sr_wyprzedzenie_min": float(np.mean(wyprzedzenia)) if wyprzedzenia else 0,
            "falszywe_serie": serie, "falszywe_na_dobe": serie / dni / 4}  # 4 młyny


def baseline_regulowy(zb: pd.DataFrame) -> np.ndarray:
    """Prosty próg fizyczny do porównania: wysycenie + deficyt."""
    return ((zb["sat_min60"] >= 20) & (zb["deficyt_sr15"] > 0.08)).astype(float).values


if __name__ == "__main__":
    zb = pd.read_pickle(ROOT / "work" / "zbior_uczacy.pkl")
    oof, models = cross_val_scores(zb)
    oof = wygladz(zb, oof)
    m = ~np.isnan(oof)
    print(f"PR-AUC (minutowo): {average_precision_score(zb.y[m], oof[m]):.3f}")
    print(f"ROC-AUC (minutowo): {roc_auc_score(zb.y[m], oof[m]):.3f}")

    print("\n--- ocena zdarzeniowa (LightGBM) ---")
    for prog in [0.5, 0.7, 0.8, 0.9, 0.95]:
        r = ocena_zdarzeniowa(zb, oof, prog)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

    print("\n--- baseline regułowy ---")
    rb = baseline_regulowy(zb)
    r = ocena_zdarzeniowa(zb, rb, 0.5)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

    # ważność cech (uśredniona po foldach)
    imp = pd.DataFrame({f"f{i}": mo.feature_importances_ for i, mo in enumerate(models)},
                       index=zb.drop(columns=META).columns).mean(axis=1).sort_values(ascending=False)
    print("\nTOP 15 cech:")
    print(imp.head(15).round(1).to_string())

    np.save(ROOT / "work" / "oof_scores.npy", oof)
    print("\nzapisano work/oof_scores.npy")
