# -*- coding: utf-8 -*-
"""Budowa zbioru uczącego: cechy + etykieta 'zapchanie w ciągu H minut'.

Zasady:
- cechy liczone wyłącznie z przeszłości (rolling/diff), brak wycieku przyszłości,
- etykieta y(t)=1 dla t w przedziale [start_epizodu - H, start_epizodu),
- minuty trwania epizodu oraz 60 min po nim wyłączone z uczenia (rekonwalescencja),
- minuty postoju młyna (moc <= 20 kW) i 60 min po rozruchu wyłączone,
- wszystkie młyny sklejone w jeden zbiór (cechy niezależne od numeru młyna).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_all_mills
from src.labels import maski_zdarzen, epizody

HORYZONT_MIN = 60          # ile minut przed zdarzeniem ma ostrzegać model
REKONWALESCENCJA_MIN = 60  # ile minut po epizodzie wykluczyć z uczenia
ROZRUCH_MIN = 60           # ile minut po starcie młyna wykluczyć
MIN_D1_SILNY = 3           # epizod 'silny' = co najmniej tyle minut deficytu d1
# epizody słabe (nagłe d2/d3 bez fazy narastania) są nieprzewidywalne z definicji:
# ich okna przedzdarzeniowe wykluczamy z uczenia i oceny zamiast karać model


def cechy_mlyna(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    eps = 1e-6
    deficyt = (df["went_zadana"] - df["went_pomiar"]) / df["went_zadana"].clip(lower=1)
    droznosc = df["went_pomiar"] / (df["went_sterowanie"].clip(lower=1))

    f["deficyt"] = deficyt
    f["droznosc"] = droznosc
    f["went_sterowanie"] = df["went_sterowanie"]
    f["went_pomiar"] = df["went_pomiar"]
    f["moc_mlyna"] = df["moc_mlyna"]
    f["podajnik_predkosc"] = df["podajnik_predkosc"]
    f["moc_na_wegiel"] = df["moc_mlyna"] / df["podajnik_predkosc"].clip(lower=1)
    f["t_miesz_odchylka"] = df["t_mieszanki"] - df["t_mieszanki_zadana"]
    f["t_miesz_ster"] = df["t_mieszanki_ster"]
    f["klapa_goraca"] = df["klapa_goraca"]
    f["p_mlyn"] = df["p_mlyn"]
    f["p_przed_went"] = df["p_przed_went"]

    # dynamika (przeszłość -> teraz)
    for c, okna in {"deficyt": (15, 60), "droznosc": (15, 60), "moc_mlyna": (15, 60),
                    "t_miesz_odchylka": (30,), "went_pomiar": (15,),
                    "p_przed_went": (30,), "podajnik_predkosc": (15,)}.items():
        for w in okna:
            f[f"{c}_d{w}"] = f[c] - f[c].shift(w)          # zmiana w oknie
            f[f"{c}_sr{w}"] = f[c].rolling(w).mean()       # poziom średni
    f["moc_std30"] = df["moc_mlyna"].rolling(30).std()     # niestabilność mocy
    f["sat_min60"] = (df["went_sterowanie"] >= 98).rolling(60).sum()  # minuty wysycenia
    f["deficyt_min30"] = (deficyt > 0.10).rolling(30).sum()
    return f


def zbuduj_zbior(horyzont: int = HORYZONT_MIN):
    mills = load_all_mills()
    czesci = []
    for m, df in mills.items():
        df = df.interpolate(limit=5)  # krótkie braki
        z = maski_zdarzen(df)
        ep = epizody(z["zdarzenie"])
        f = cechy_mlyna(df)

        # siła epizodu: ile minut warunku d1 (powolne dławienie z prekursorami)
        sila = []
        for _, r in ep.iterrows():
            sila.append(int(z.loc[r.start:r.koniec, "d1"].sum()))
        ep["min_d1"] = sila
        ep["silny"] = ep["min_d1"] >= MIN_D1_SILNY

        y = pd.Series(0, index=df.index, dtype=int)
        wyklucz = ~z["praca"]
        # wyklucz rozruchy: początek pracy po postoju
        start_pracy = z["praca"] & ~z["praca"].shift(1).fillna(False).astype(bool)
        for t in z.index[start_pracy]:
            wyklucz.loc[t:t + pd.Timedelta(minutes=ROZRUCH_MIN)] = True
        for _, r in ep.iterrows():
            if r.silny:
                # okno predykcyjne przed silnym zdarzeniem
                y.loc[r.start - pd.Timedelta(minutes=horyzont):r.start - pd.Timedelta(minutes=1)] = 1
            else:
                # słabe (nagłe) zdarzenie: okno przedzdarzeniowe poza uczeniem i oceną
                wyklucz.loc[r.start - pd.Timedelta(minutes=horyzont):r.start] = True
            # samo zdarzenie + rekonwalescencja poza uczeniem
            wyklucz.loc[r.start:r.koniec + pd.Timedelta(minutes=REKONWALESCENCJA_MIN)] = True

        czesc = f.copy()
        czesc["y"] = y
        czesc["mlyn"] = m
        czesc["epizod_id"] = -1
        for i, r in ep.iterrows():
            if not r.silny:
                continue
            okno = (czesc.index >= r.start - pd.Timedelta(minutes=horyzont)) & (czesc.index < r.start)
            czesc.loc[okno, "epizod_id"] = i + m * 100  # unikalny id epizodu
        czesc = czesc[~wyklucz.values]
        czesci.append(czesc)

    zb = pd.concat(czesci)
    zb["dzien"] = zb.index.date.astype("str")
    return zb


if __name__ == "__main__":
    zb = zbuduj_zbior()
    n_poz = int(zb.y.sum())
    print(f"zbior: {zb.shape}, pozytywne minuty: {n_poz} ({n_poz/len(zb)*100:.2f}%)")
    print("epizody w zbiorze:", sorted(zb.loc[zb.epizod_id >= 0, 'epizod_id'].unique()))
    print("braki w cechach %:", round(zb.drop(columns=['y','mlyn','epizod_id','dzien']).isna().mean().mean()*100, 2))
    zb.to_parquet = None  # brak pyarrow; zapis przez pickle
    zb.to_pickle(Path(__file__).resolve().parent.parent / "work" / "zbior_uczacy.pkl")
    print("zapisano work/zbior_uczacy.pkl")
