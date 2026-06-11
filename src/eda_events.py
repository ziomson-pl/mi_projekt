# -*- coding: utf-8 -*-
"""Detekcja kandydatów na epizody zapchania młyna + wykresy zbliżeń.

Sygnatury (młyn wentylatorowy):
A) deficyt wentylacji: pomiar << zadana przy wysyconym sterowaniu (rosnący opór)
B) skok mocy młyna względem mediany kroczącej (przeładowanie komory mielenia)
C) trip: nagły zrzut mocy do ~0 w trakcie pracy
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_all_mills

OUT = Path(__file__).resolve().parent.parent / "work" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def grupuj(mask: pd.Series, min_przerwa_min: int = 30) -> list[tuple]:
    """Skleja sąsiednie minuty spełniające warunek w epizody."""
    idx = mask[mask].index
    if len(idx) == 0:
        return []
    grupy, start, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        if (t - prev) > pd.Timedelta(minutes=min_przerwa_min):
            grupy.append((start, prev))
            start = t
        prev = t
    grupy.append((start, prev))
    return grupy


def detekcja(df: pd.DataFrame, m: int) -> pd.DataFrame:
    praca = df["moc_mlyna"] > 20

    # A) deficyt wentylacji przy wysyconym sterowaniu
    deficyt = (df["went_zadana"] - df["went_pomiar"]) / df["went_zadana"].clip(lower=1)
    sat = df["went_sterowanie"] >= 95
    maskA = praca & sat & (deficyt > 0.15)

    # B) skok mocy względem mediany kroczącej 6h
    med = df["moc_mlyna"].rolling("6h").median()
    mad = (df["moc_mlyna"] - med).abs().rolling("6h").median().clip(lower=1)
    z = (df["moc_mlyna"] - med) / (1.4826 * mad)
    maskB = praca & (z > 6) & (df["moc_mlyna"] > med + 8)

    # C) trip — moc spada z >35 kW do <5 kW w ciągu 3 minut
    moc = df["moc_mlyna"]
    maskC = (moc.shift(3) > 35) & (moc < 5)

    rows = []
    for name, mask in [("A_deficyt_went", maskA), ("B_skok_mocy", maskB), ("C_trip", maskC)]:
        for s, e in grupuj(mask):
            rows.append({"mlyn": m, "typ": name, "start": s, "koniec": e,
                         "czas_min": (e - s).total_seconds() / 60 + 1})
    ev = pd.DataFrame(rows)
    return ev


def zoom_plot(df, m, t0, t1, fname, ev_all=None):
    cols = [("moc_mlyna", None), ("podajnik_predkosc", "podajnik_prad"),
            ("went_pomiar", "went_zadana"), ("went_sterowanie", "kierownica"),
            ("t_mieszanki", "t_mieszanki_zadana"), ("p_mlyn", "p_przed_went")]
    d = df.loc[t0:t1]
    fig, axes = plt.subplots(len(cols), 1, figsize=(16, 13), sharex=True)
    for ax, pair in zip(axes, cols):
        for c in pair:
            if c and c in d.columns:
                ax.plot(d.index, d[c], lw=0.9, label=c)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
    if ev_all is not None and len(ev_all):
        sub = ev_all[(ev_all.mlyn == m) & (ev_all.start <= t1) & (ev_all.koniec >= t0)]
        for _, r in sub.iterrows():
            for ax in axes:
                ax.axvspan(max(r.start, t0), min(r.koniec, t1), color={"A_deficyt_went": "orange", "B_skok_mocy": "red", "C_trip": "purple"}[r.typ], alpha=0.18)
    fig.suptitle(f"MW{m}  {t0} – {t1}")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    mills = load_all_mills()
    all_ev = pd.concat([detekcja(df, m) for m, df in mills.items()], ignore_index=True)
    all_ev = all_ev.sort_values("start")
    all_ev.to_csv(OUT.parent / "kandydaci_zdarzen.csv", index=False)
    print(all_ev.groupby(["mlyn", "typ"]).agg(n=("typ", "size"), sr_czas=("czas_min", "mean")).round(1))
    print()
    print(all_ev.to_string(max_rows=200))
