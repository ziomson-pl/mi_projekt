# -*- coding: utf-8 -*-
"""Przeglądowe wykresy miesięczne kluczowych sygnałów per młyn."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_all_mills

OUT = Path(__file__).resolve().parent.parent / "work" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = [
    (["moc_mlyna", "moc_wentylatora"], "moc [kW]"),
    (["podajnik_predkosc", "podajnik_prad"], "podajnik [%], [A]"),
    (["went_pomiar", "went_zadana"], "wentylacja [kNm3/h]"),
    (["went_sterowanie", "kierownica"], "sterowanie went. [%]"),
    (["t_mieszanki", "t_mieszanki_zadana"], "T mieszanki [°C]"),
    (["t_mieszanki_ster", "klapa_goraca", "klapa_zimna"], "klapy/ster T [%]"),
    (["p_mlyn", "p_przed_went"], "ciśnienia [kPa]"),
]

mills = load_all_mills()
for m, df in mills.items():
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(22, 18), sharex=True)
    for ax, (cols, ylab) in zip(axes, PANELS):
        for c in cols:
            if c in df.columns:
                ax.plot(df.index, df[c], lw=0.4, label=c)
        ax.set_ylabel(ylab, fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("czas")
    fig.suptitle(f"MW{m} – przegląd luty 2026", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"przeglad_MW{m}.png", dpi=110)
    plt.close(fig)
    print("zapisano", OUT / f"przeglad_MW{m}.png")
