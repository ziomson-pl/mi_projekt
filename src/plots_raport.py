# -*- coding: utf-8 -*-
"""Ryciny do raportu LaTeX (work/plots/raport/)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path
import joblib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_mill
from src.labels import maski_zdarzen, epizody

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "plots" / "raport"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
META = ["y", "mlyn", "epizod_id", "dzien"]


# ----------------------------------------------------------- 1. anatomia zdarzenia
def fig_anatomia():
    df = load_mill(3).loc["2026-02-15 00:00":"2026-02-15 06:30"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(df.index, df.went_pomiar, lw=1, label="przepływ (pomiar)")
    axes[0].plot(df.index, df.went_zadana, lw=1, ls="--", label="wartość zadana")
    axes[0].set_ylabel("wentylacja\n[kNm$^3$/h]")
    axes[1].plot(df.index, df.kierownica, lw=1, color="darkorange", label="kierownica wentylatora")
    axes[1].set_ylabel("kierownica [%]")
    axes[2].plot(df.index, df.moc_mlyna, lw=1, color="seagreen", label="moc młyna")
    axes[2].set_ylabel("moc [kW]")
    axes[3].plot(df.index, df.t_mieszanki, lw=1, color="purple", label="T mieszanki")
    axes[3].plot(df.index, df.t_mieszanki_zadana, lw=1, ls="--", color="gray", label="zadana")
    axes[3].set_ylabel("T mieszanki [°C]")
    axes[4].plot(df.index, df.podajnik_predkosc, lw=1, color="firebrick", label="prędkość podajnika")
    axes[4].set_ylabel("podajnik [%]")
    axes[4].set_xlabel("czas (15.02.2026)")

    fazy = [("2026-02-15 01:10", "2026-02-15 02:38", "gold", "I: wysycenie regulatora"),
            ("2026-02-15 02:38", "2026-02-15 04:11", "red", "II: deficyt przepływu (zdarzenie)"),
            ("2026-02-15 04:11", "2026-02-15 05:10", "violet", "III: odciążanie i oczyszczanie")]
    for ax in axes:
        for t0, t1, c, lab in fazy:
            ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color=c, alpha=0.15)
        ax.legend(loc="best", fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for t0, t1, c, lab in fazy:
        axes[0].text(pd.Timestamp(t0), axes[0].get_ylim()[1] * 0.98, " " + lab,
                     fontsize=7, va="top", color="k")
    fig.suptitle("Anatomia epizodu zapchania - MW3, 15.02.2026", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "anatomia_zdarzenia.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------- 2. przegląd miesiąca
def fig_przeglad():
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    df = load_mill(3)
    z = maski_zdarzen(df)
    ep = epizody(z["zdarzenie"])
    axes[0].plot(df.index, df.moc_mlyna, lw=0.35, color="seagreen")
    axes[0].set_ylabel("moc młyna [kW]")
    axes[1].plot(df.index, df.went_pomiar, lw=0.35, color="steelblue", label="pomiar")
    axes[1].plot(df.index, df.went_zadana, lw=0.35, color="darkorange", label="zadana")
    axes[1].set_ylabel("wentylacja [kNm$^3$/h]")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[2].plot(df.index, df.kierownica, lw=0.35, color="firebrick")
    axes[2].set_ylabel("kierownica [%]")
    axes[2].set_xlabel("czas")
    for ax in axes:
        for _, r in ep.iterrows():
            ax.axvspan(r.start - pd.Timedelta(hours=2),
                       r.koniec + pd.Timedelta(hours=2), color="red", alpha=0.30)
    fig.suptitle("MW3 - luty 2026; czerwone pasma = epizody zapchania (poszerzone do widoczności)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "przeglad_MW3.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------ 3. krzywe PR (3 modele)
def fig_pr():
    zb = pd.read_pickle(ROOT / "work" / "zbior_uczacy.pkl")
    zb_lstm = pd.read_pickle(ROOT / "work" / "zb_lstm.pkl")
    oof_mlp = np.load(ROOT / "work" / "oof_mlp.npy")
    # LightGBM: odtworzenie 5-fold (deterministyczne)
    from src.train import cross_val_scores, wygladz
    oof_gbm, _ = cross_val_scores(zb, n_splits=5)
    oof_gbm = wygladz(zb, oof_gbm)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for nazwa, y, s, kolor in [
            ("LightGBM (cechy)", zb.y.values, oof_gbm, "navy"),
            ("MLP (cechy)", zb.y.values, oof_mlp, "darkorange"),
            ("LSTM (sekwencje)", zb_lstm.y.values, zb_lstm.score.values, "seagreen")]:
        m = ~np.isnan(s)
        p, r, _ = precision_recall_curve(y[m], s[m])
        ap = average_precision_score(y[m], s[m])
        ax.plot(r, p, lw=1.6, color=kolor, label=f"{nazwa}: AP={ap:.3f}")
    ax.set_xlabel("czułość (recall) - minuty okna ostrzegania")
    ax.set_ylabel("precyzja")
    ax.set_title("Krzywe precyzja-czułość (minutowo, walidacja 5-fold po dniach)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "krzywe_pr.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------- 4. krzywe uczenia (niestabilność)
def fig_krzywe_uczenia():
    k = pd.read_csv(ROOT / "work" / "nn_learning_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, model in zip(axes, ["MLP", "LSTM"]):
        d = k[k.model == model]
        for fold, g in d.groupby("fold"):
            ax.plot(g.epoka, g.ap_te, marker="o", ms=2.5, lw=0.9, label=f"fold {fold}")
        ax.set_title(f"{model}: AP na foldzie testowym vs epoka")
        ax.set_xlabel("epoka")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("average precision (test)")
    fig.suptitle("Niestabilność uczenia sieci przy 5 zdarzeniach (fold 4 nie zawiera żadnego zdarzenia)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "krzywe_uczenia_nn.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------- 5. ważności cech
def fig_waznosci():
    art = joblib.load(ROOT / "work" / "model_zapchanie.joblib")
    imp = pd.Series(art["model"].feature_importances_, index=art["cechy"])
    imp = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_xlabel("ważność (liczba podziałów w drzewach)")
    ax.set_title("LightGBM: 15 najważniejszych cech (model finalny)")
    fig.tight_layout()
    fig.savefig(OUT / "waznosci_cech.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------- 6. oś czasu score (OOF)
def fig_score():
    zb = pd.read_pickle(ROOT / "work" / "zbior_uczacy.pkl")
    oof = np.load(ROOT / "work" / "oof_lodo.npy")
    zb = zb.assign(score=oof)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, m in zip(axes, [3, 4]):
        d = zb[zb.mlyn == m]
        ax.plot(d.index, d.score, lw=0.5, color="steelblue")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="próg alarmu 0,5")
        poz = d[d.y == 1]
        ax.scatter(poz.index, poz.score, s=6, color="red", zorder=3,
                   label="okno 60 min przed zapchaniem")
        ax.set_ylabel(f"MW{m}: score")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("czas")
    fig.suptitle("Wyjście modelu LightGBM w walidacji leave-one-day-out (predykcje pozapróbkowe)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "score_oof.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------- 7. schemat etykietowania
def fig_etykiety():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    strefy = [(-180, -60, "#d8e6f3", "obserwacja\n(y=0)"),
              (-60, 0, "#f6c8c8", "okno ostrzegania\nH=60 min (y=1)"),
              (0, 90, "#e06666", "ZDARZENIE\n(wyłączone z uczenia)"),
              (90, 150, "#e9d8f3", "rekonwalescencja\n(wyłączona)"),
              (150, 250, "#d8e6f3", "obserwacja\n(y=0)")]
    for x0, x1, c, lab in strefy:
        ax.axvspan(x0, x1, color=c)
        ax.text((x0 + x1) / 2, 0.5, lab, ha="center", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=1.2)
    ax.text(0, 1.02, "początek zdarzenia", ha="center", fontsize=8)
    ax.set_xlim(-180, 250)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("czas względem początku zdarzenia [min]")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "schemat_etykiet.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in [fig_anatomia, fig_przeglad, fig_pr, fig_krzywe_uczenia,
              fig_waznosci, fig_score, fig_etykiety]:
        f()
        print("OK", f.__name__)
