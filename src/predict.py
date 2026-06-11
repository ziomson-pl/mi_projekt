# -*- coding: utf-8 -*-
"""Finalny model produkcyjny + funkcja inferencji.

Użycie:
    python3 src/predict.py            # trenuje na całości i zapisuje model
    python3 src/predict.py --demo     # dodatkowo rysuje przebieg score na MW3/MW4
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_mill
from src.features import cechy_mlyna
from src.train import wygladz

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "work" / "model_zapchanie.joblib"
META = ["y", "mlyn", "epizod_id", "dzien"]
PROG_ALARMU = 0.5
MIN_SERII = 3  # alarm dopiero po 3 kolejnych minutach powyżej progu


def trenuj_final():
    zb = pd.read_pickle(ROOT / "work" / "zbior_uczacy.pkl")
    X = zb.drop(columns=META)
    y = zb["y"].values
    pos = y.sum()
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0,
        scale_pos_weight=min((len(y) - pos) / max(pos, 1), 30),
        random_state=42, verbose=-1)
    model.fit(X, y)
    joblib.dump({"model": model, "cechy": list(X.columns),
                 "prog": PROG_ALARMU, "min_serii": MIN_SERII}, MODEL_PATH)
    print("zapisano", MODEL_PATH)
    return model


def score_mlyna(df_mlyna: pd.DataFrame, art=None) -> pd.DataFrame:
    """Liczy score zapchania i flagę alarmu dla danych jednego młyna (format jak z loadera)."""
    art = art or joblib.load(MODEL_PATH)
    f = cechy_mlyna(df_mlyna.interpolate(limit=5))[art["cechy"]]
    score = art["model"].predict_proba(f)[:, 1]
    score = pd.Series(score, index=f.index).rolling(5, min_periods=1).median()
    praca = df_mlyna["moc_mlyna"] > 20
    score[~praca] = 0.0  # młyn stoi -> brak alarmów
    alarm_raw = score >= art["prog"]
    alarm = alarm_raw.rolling(art["min_serii"], min_periods=art["min_serii"]).min().fillna(0).astype(bool)
    return pd.DataFrame({"score": score, "alarm": alarm})


if __name__ == "__main__":
    model = trenuj_final()
    if "--demo" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        art = joblib.load(MODEL_PATH)
        fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
        for ax, m in zip(axes, [3, 4]):
            df = load_mill(m)
            out = score_mlyna(df, art)
            ax.plot(out.index, out.score, lw=0.5, color="steelblue", label="score modelu")
            ax.fill_between(out.index, 0, out.alarm.astype(int), color="red", alpha=0.4,
                            step="mid", label="ALARM")
            ax.set_ylabel(f"MW{m}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)
        fig.suptitle("Score ryzyka zapchania (uwaga: na danych treningowych - tylko poglądowo)")
        fig.tight_layout()
        fig.savefig(ROOT / "work" / "plots" / "score_final_MW3_MW4.png", dpi=110)
        print("zapisano work/plots/score_final_MW3_MW4.png")
