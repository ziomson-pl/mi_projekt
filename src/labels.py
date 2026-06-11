# -*- coding: utf-8 -*-
"""Definicja zdarzenia 'zapchanie młyna' (dławienie) na podstawie sygnałów.

Zdarzenie = którykolwiek z warunków (w czasie pracy młyna, moc > 20 kW):
  D1: deficyt wentylacji >20% względem wartości zadanej utrzymujący się >=5 min
      przy sterowaniu regulatora wentylacji >=98% (wysycenie),
  D2: awaryjne odciążenie - prędkość podajnika spada o >=25 pkt w <=3 min,
      a młyn dalej pracuje (moc > 20 kW przez kolejne 10 min)
      -> automatyka/operator ratuje zapychający się młyn,
  D3: przeciążenie - moc młyna > kwantyl 0.995 swojego rozkładu w pracy
      przy jednoczesnym deficycie wentylacji > 10%.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_all_mills


def maski_zdarzen(df: pd.DataFrame) -> pd.DataFrame:
    praca = df["moc_mlyna"] > 20
    deficyt = (df["went_zadana"] - df["went_pomiar"]) / df["went_zadana"].clip(lower=1)

    # D1 - deficyt przy wysyceniu, trwale >= 5 min
    d1_raw = praca & (df["went_sterowanie"] >= 98) & (deficyt > 0.20)
    d1 = d1_raw.rolling(5, min_periods=5).min().fillna(0).astype(bool)  # 5 kolejnych minut

    # D2 - awaryjne ciecie podajnika przy pracujacym mlynie
    spadek = df["podajnik_predkosc"].shift(3) - df["podajnik_predkosc"]
    mlyn_dalej = praca.shift(-10).fillna(False).astype(bool)  # za 10 min wciaz pracuje
    d2 = praca & (spadek >= 25) & mlyn_dalej & (df["podajnik_predkosc"].shift(3) > 30)

    # D3 - przeciazenie mocy + deficyt
    prog_mocy = df.loc[praca, "moc_mlyna"].quantile(0.995)
    d3 = praca & (df["moc_mlyna"] > prog_mocy) & (deficyt > 0.10)

    out = pd.DataFrame({"d1": d1.fillna(False), "d2": d2.fillna(False),
                        "d3": d3.fillna(False), "praca": praca})
    out["zdarzenie"] = out[["d1", "d2", "d3"]].any(axis=1)
    out["deficyt"] = deficyt
    return out


def epizody(zd: pd.Series, sklej_min: int = 90) -> pd.DataFrame:
    """Skleja minuty zdarzenia w epizody (przerwa < sklej_min => ten sam epizod)."""
    idx = zd[zd].index
    rows = []
    if len(idx):
        start = prev = idx[0]
        for t in idx[1:]:
            if (t - prev) > pd.Timedelta(minutes=sklej_min):
                rows.append((start, prev))
                start = t
            prev = t
        rows.append((start, prev))
    return pd.DataFrame(rows, columns=["start", "koniec"])


if __name__ == "__main__":
    mills = load_all_mills()
    wszystkie = []
    for m, df in mills.items():
        z = maski_zdarzen(df)
        ep = epizody(z["zdarzenie"])
        ep.insert(0, "mlyn", m)
        # ile minut kazdego typu w epizodzie
        for i, r in ep.iterrows():
            w = z.loc[r.start:r.koniec]
            ep.loc[i, "min_d1"] = int(w.d1.sum())
            ep.loc[i, "min_d2"] = int(w.d2.sum())
            ep.loc[i, "min_d3"] = int(w.d3.sum())
        wszystkie.append(ep)
        print(f"MW{m}: {len(ep)} epizodow, minut zdarzenia: {int(z.zdarzenie.sum())}, "
              f"d1={int(z.d1.sum())} d2={int(z.d2.sum())} d3={int(z.d3.sum())}")
    ep_all = pd.concat([e for e in wszystkie if len(e)], ignore_index=True)
    ep_all["czas_min"] = (ep_all.koniec - ep_all.start).dt.total_seconds() / 60 + 1
    ep_all.to_csv(Path(__file__).resolve().parent.parent / "work" / "epizody_zapchania.csv", index=False)
    print()
    print(ep_all.to_string())
