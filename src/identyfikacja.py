# -*- coding: utf-8 -*-
"""Analiza identyfikacyjna młyna jako obiektu dynamicznego.

(a) Charakterystyka statyczna kanału kierownica -> przepływ wentylacji
    z punktów pracy: stan zdrowy vs okna przed zapchaniem. Zapychanie
    obniża wzmocnienie statyczne (drożność) obiektu.
(b) Rekurencyjna identyfikacja ARX(1,1) metodą RLS z zapominaniem - wraz
    z demonstracją obciążenia estymaty w zamkniętej pętli regulacji
    (przy słabym pobudzeniu estymata wzmocnienia potrafi być ujemna,
    bo regulator antykoreluje u z zakłóceniem wyjścia).
(c) Odpowiedź skokowa zamkniętego układu regulacji wentylacji na skok
    wartości zadanej + dopasowanie modelu FOPDT (inercja I rzędu z opóźnieniem).

Wyniki -> work/identyfikacja.json, wykresy -> work/plots/raport/
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_mill
from src.labels import maski_zdarzen, epizody

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "plots" / "raport"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def stany(df):
    """Maski: praca, okna 60 min przed epizodami, stan zdrowy."""
    z = maski_zdarzen(df)
    ep = epizody(z["zdarzenie"])
    praca = df["moc_mlyna"] > 20
    przed = pd.Series(False, index=df.index)
    chore = pd.Series(False, index=df.index)
    for _, r in ep.iterrows():
        przed.loc[r.start - pd.Timedelta(minutes=60):r.start] = True
        chore.loc[r.start - pd.Timedelta(minutes=180):
                  r.koniec + pd.Timedelta(minutes=120)] = True
    zdrowy = praca & ~chore
    return praca, przed & praca, zdrowy, ep


# ------------------------------------------- (a) charakterystyka statyczna
def charakterystyka_statyczna(m: int = 3):
    df = load_mill(m)
    praca, przed, zdrowy, ep = stany(df)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(df.loc[zdrowy, "kierownica"], df.loc[zdrowy, "went_pomiar"],
               s=2, alpha=0.15, color="steelblue", label="stan zdrowy", rasterized=True)
    ax.scatter(df.loc[przed, "kierownica"], df.loc[przed, "went_pomiar"],
               s=4, alpha=0.5, color="crimson", label="60 min przed zapchaniem", rasterized=True)
    # mediana przepływu w pasmach położenia kierownicy (charakterystyka statyczna)
    wyn = {}
    for nazwa, maska, kolor in [("zdrowy", zdrowy, "navy"), ("przed", przed, "darkred")]:
        d = df.loc[maska, ["kierownica", "went_pomiar"]].dropna()
        pasma = pd.cut(d["kierownica"], bins=np.arange(0, 105, 5))
        med = d.groupby(pasma, observed=True)["went_pomiar"].median()
        srod = [iv.mid for iv in med.index]
        ax.plot(srod, med.values, lw=2.2, color=kolor,
                label=f"mediana ({nazwa})")
        wyn[nazwa] = {str(iv): round(float(v), 2) for iv, v in med.items()}
    ax.set_xlabel("położenie kierownicy wentylatora [%]")
    ax.set_ylabel("przepływ wentylacji [kNm$^3$/h]")
    ax.set_title(f"MW{m}: charakterystyka statyczna kierownica→przepływ")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"identyfikacja_charakterystyka_MW{m}.png", dpi=150)
    plt.close(fig)
    return {"mlyn": m, "mediany_pasm": wyn}


# ----------------------------------------------------- (b) RLS / dryf wzmocnienia
def rls_arx(u: np.ndarray, y: np.ndarray, lam: float = 0.998):
    """ARX(1,1): y(t) = a*y(t-1) + b*u(t-1). Zwraca trajektorie a(t), b(t)."""
    theta = np.zeros(2)
    P = np.eye(2) * 1000.0
    A, B = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for t in range(1, len(y)):
        if np.isnan(y[t]) or np.isnan(y[t - 1]) or np.isnan(u[t - 1]):
            A[t], B[t] = theta
            continue
        phi = np.array([y[t - 1], u[t - 1]])
        k = P @ phi / (lam + phi @ P @ phi)
        theta = theta + k * (y[t] - phi @ theta)
        P = (P - np.outer(k, phi @ P)) / lam
        A[t], B[t] = theta
    return A, B


def dryf_wzmocnienia(m: int = 3):
    df = load_mill(m)
    praca, przed, zdrowy, ep = stany(df)

    u = df["kierownica"].where(praca).values
    y = df["went_pomiar"].where(praca).values
    A, B = rls_arx(u, y)
    K = B / np.clip(1 - A, 1e-3, None)          # wzmocnienie statyczne ARX
    K = pd.Series(K, index=df.index).where(praca.values)
    K_smooth = K.rolling(30, min_periods=10).median()

    # drożność: iloraz punktu pracy (odporna estymata wzmocnienia statycznego)
    droznosc = (df["went_pomiar"] / df["kierownica"].clip(lower=1)).where(praca)
    droznosc_smooth = droznosc.rolling(30, min_periods=10).median()

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df.index, df["went_pomiar"], lw=0.4, color="steelblue",
                 label="przepływ wentylacji")
    axes[0].plot(df.index, df["went_zadana"], lw=0.4, color="darkorange",
                 label="wartość zadana")
    axes[0].set_ylabel("przepływ [kNm$^3$/h]")
    axes[1].plot(droznosc_smooth.index, droznosc_smooth, lw=0.7, color="seagreen",
                 label="drożność $y/u$ (mediana 30 min)")
    axes[1].set_ylabel("drożność [kNm$^3$/h / %]")
    axes[2].plot(K_smooth.index, K_smooth, lw=0.7, color="purple",
                 label="$\\hat{K}_{ARX}=\\hat{b}/(1-\\hat{a})$ (RLS, $\\lambda$=0.998)")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_ylim(-0.6, 1.0)
    axes[2].set_ylabel("$\\hat{K}_{ARX}$ [-]")
    axes[2].set_xlabel("czas")
    for ax in axes:
        for _, r in ep.iterrows():
            ax.axvspan(r.start, r.koniec, color="red", alpha=0.35)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"MW{m}: estymaty wzmocnienia kanału kierownica→przepływ; "
                 "czerwone pasma = epizody zapchania", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"identyfikacja_dryf_K_MW{m}.png", dpi=150)
    plt.close(fig)

    # statystyka: drożność w stanie zdrowym vs przed epizodami
    return {"mlyn": m,
            "droznosc_zdrowa_mediana": round(float(droznosc_smooth[zdrowy].median()), 3),
            "droznosc_przed_mediana": round(float(droznosc_smooth[przed].median()), 3)
            if przed.any() else None,
            "K_ARX_zdrowy_mediana": round(float(K_smooth[zdrowy].median()), 3),
            "K_ARX_ujemne_proc": round(float((K_smooth[praca] < 0).mean() * 100), 1),
            "n_epizodow": len(ep)}


# ------------------------------------------------- (c) odpowiedź skokowa (CL)
def fopdt(t, K, T, L):
    """Inercja I rzędu z opóźnieniem; odpowiedź na skok jednostkowy."""
    y = K * (1 - np.exp(-(t - L) / np.clip(T, 1e-3, None)))
    y = np.where(t < L, 0.0, y)
    return y


def odpowiedz_skokowa(m: int = 3, prog_skoku: float = 1.0,
                      przed: int = 10, po: int = 40):
    """Odpowiedź toru regulacji temperatury mieszanki na skok wartości zadanej.

    Pętla temperatury dostaje skokowe zmiany zadanej od operatora (pętla
    wentylacji - tylko rampy z układu obciążenia, brak czystych skoków).
    """
    df = load_mill(m)
    praca = df["moc_mlyna"] > 20
    bez_sat = df["t_mieszanki_ster"] < 98
    sp = df["t_mieszanki_zadana"]
    pv = df["t_mieszanki"]
    d = sp.diff()

    okna = []
    kandydaci = df.index[((d.abs() >= prog_skoku) & praca).fillna(False)]
    for t in kandydaci:
        i = df.index.get_loc(t)
        if i < przed + 2 or i + po + 1 >= len(df):
            continue
        du = float(sp.iloc[i] - sp.iloc[i - 1])
        # plateau zadanej przed i po - pojedynczy czysty skok
        if sp.iloc[i - przed:i].std() > 0.1 or sp.iloc[i + 1:i + po + 1].std() > 0.1:
            continue
        # młyn pracuje i regulator temperatury nie jest wysycony w całym oknie
        if not (praca.iloc[i - przed:i + po + 1].all()
                and bez_sat.iloc[i - przed:i + po + 1].all()):
            continue
        y0 = pv.iloc[i - 5:i].mean()
        odp = (pv.iloc[i - przed:i + po + 1].values - y0) / du
        if np.isnan(odp).any():
            continue
        okna.append(odp)
    t_os = np.arange(-przed, po + 1)
    if len(okna) < 3:
        print(f"MW{m}: za mało czystych skoków ({len(okna)})")
        return {"mlyn": m, "n_skokow": len(okna), "FOPDT": None}
    okna = np.vstack(okna)

    sr = okna.mean(axis=0)
    t_fit = t_os[t_os >= 0].astype(float)
    y_fit = sr[t_os >= 0]
    popt, _ = curve_fit(fopdt, t_fit, y_fit, p0=[1.0, 10.0, 2.0],
                        bounds=([0.1, 0.5, 0.0], [2.0, 60.0, 15.0]))
    K, T, L = popt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for o in okna:
        ax.plot(t_os, o, lw=0.4, color="lightsteelblue", alpha=0.6)
    ax.plot(t_os, sr, lw=2.2, color="navy", label=f"średnia ({len(okna)} skoków)")
    ax.plot(t_fit, fopdt(t_fit, *popt), lw=2.0, ls="--", color="crimson",
            label=f"FOPDT: K={K:.2f}, T={T:.1f} min, L={L:.1f} min")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_ylim(-1, 2.5)
    ax.set_xlabel("czas od skoku wartości zadanej [min]")
    ax.set_ylabel("znormalizowana odpowiedź $\\Delta y/\\Delta u$ [-]")
    ax.set_title(f"MW{m}: odpowiedź toru regulacji temperatury mieszanki na skok zadanej")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"identyfikacja_skok_MW{m}.png", dpi=150)
    plt.close(fig)
    return {"mlyn": m, "n_skokow": int(len(okna)),
            "FOPDT": {"K": round(float(K), 3), "T_min": round(float(T), 2),
                      "L_min": round(float(L), 2)}}


if __name__ == "__main__":
    wyniki = {"charakterystyka": [], "dryf": [], "skok": []}
    for m in [3, 4]:
        wyniki["charakterystyka"].append(charakterystyka_statyczna(m))
        wyniki["dryf"].append(dryf_wzmocnienia(m))
        print(f"dryf MW{m}:", wyniki["dryf"][-1])
        wyniki["skok"].append(odpowiedz_skokowa(m))
        print(f"skok MW{m}:", wyniki["skok"][-1])
    # mediany pasm są obszerne - nie zapisujemy ich do json
    for c in wyniki["charakterystyka"]:
        c.pop("mediany_pasm", None)
    with open(ROOT / "work" / "identyfikacja.json", "w") as f:
        json.dump(wyniki, f, indent=2)
    print("zapisano work/identyfikacja.json")
