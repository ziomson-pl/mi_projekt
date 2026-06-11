# -*- coding: utf-8 -*-
"""Wczytywanie danych młynów węglowych bloku 1 (luty 2026)."""
import pandas as pd
from pathlib import Path

DANE = Path(__file__).resolve().parent.parent / "dane"
NA_VALUES = ["N_A", "NaN", "n_a", ""]

# Wspólne, ujednolicone nazwy kluczowych sygnałów per młyn (m = 1..4)
# tag_template -> krótka nazwa
MILL_SIGNALS = {
    "401HFE{m}2AA401:pos": "klapa_zimna",       # % pozycja klapy zimnego powietrza
    "401HFE{m}1AA401:pos": "klapa_goraca",      # % pozycja klapy gorącego powietrza
    "401HFE{m}0AA401:pos": "kierownica",        # % UAR / kierownica wentylatora młynowego
    "401HFE{m}0DF201:me":  "went_pomiar",       # kNm3/h pomiar UAR wentylacji młyna
    "401HFE{m}0DF201:spa": "went_zadana",       # kNm3/h wartość zadana wentylacji
    "401HFE{m}0DF201:pos": "went_sterowanie",   # % wyjście regulatora wentylacji
    "401HHE{m}0FT950:av":  "t_mieszanki",       # °C temp. mieszanki pyłowo-powietrznej za młynem
    "401HHE{m}0DT901:spa": "t_mieszanki_zadana",# °C wartość zadana temp. mieszanki
    "401HHE{m}0DT901:pos": "t_mieszanki_ster",  # % wyjście regulatora temp. mieszanki
    "401HFB{m}0AF001HS:out": "podajnik_zal",    # sygnał załączenia podajnika węgla
    "401HFB{m}0AF001XQ51:av": "podajnik_predkosc", # % prędkość podajnika węgla (strumień węgla)
    "401HFB{m}0AF001XQ50:av": "podajnik_prad",  # A prąd podajnika węgla
    "401HFC{m}0AJ001XQ50:av": "moc_mlyna",      # kW moc silnika młyna (kruszarki)
    "401HFE{m}0AN001XQ50:av": "moc_wentylatora",# kW moc wentylatora młynowego (brak dla MW1)
    "401HFE{m}0FF901:av":  "f_powietrza",       # kNm3/h przepływ powietrza za WM
    "401HFE{m}0CP201:av":  "p_przed_went",      # kPa ciśnienie (strona wentylatora)
    "401HFC{m}0CP201:av":  "p_mlyn",            # kPa ciśnienie w młynie/komorze
    "401HFY{m}0FP901:av":  "p_uszczeln_1",      # kPa powietrze uszczelniające
    "401HFY{m}0FP902:av":  "p_uszczeln_2",      # kPa
    "401HFW{m}1CP201:av":  "p_hfw",             # kPa
    "401HFV{m}0CP201:av":  "p_olej_mlyn",       # MPa ciśn. oleju smarnego młyna
    "401HFV{m}0CT201:av":  "t_olej_mlyn",       # °C temp. oleju smarnego młyna
    "401HFV{m}1CP201:av":  "p_olej_went",       # MPa ciśn. oleju łożysk wentylatora
    "401HFE{m}0CT206:av":  "t_loz_went_a",      # °C temp. łożyska wentylatora
    "401HFE{m}0CT207:av":  "t_loz_went_b",      # °C
    "401HFE{m}0CT208:av":  "t_za_went",         # °C
    "401HFC{m}0CT201:av":  "t_kom_1",           # °C temperatury w komorze powietrznej młyna
    "401HFC{m}0CT202:av":  "t_kom_2",
    "401HFC{m}0CT203:av":  "t_kom_3",
    "401HFC{m}0CT204:av":  "t_kom_4",
    "401HFC{m}0CT205:av":  "t_kom_5",
    "401HFC{m}0CT206:av":  "t_kom_6",
    "401HFC{m}0CT207:av":  "t_kom_7",
    "401HFC{m}0CT208:av":  "t_kom_8",
    "401HHE{m}1CT201:av":  "t_pyl_1",           # °C temp. pyłoprzewodów / HHE
    "401HHE{m}2CT201:av":  "t_pyl_2",
}


def _read_raw(path: Path, time_format: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="cp1250", header=0,
                     skiprows=[1, 2], na_values=NA_VALUES, decimal=".")
    df.columns = [c.strip() for c in df.columns]
    tcol = df.columns[0]
    # pliki zawierają mieszane formaty czasu (dd.mm.yyyy HH:MM oraz ISO),
    # dlatego parsujemy dwuprzebiegowo zamiast zgadywać dayfirst
    t_iso = pd.to_datetime(df[tcol], format="ISO8601", errors="coerce")
    t_pl = pd.to_datetime(df[tcol], format="%d.%m.%Y %H:%M", errors="coerce")
    df[tcol] = t_iso.fillna(t_pl)
    if df[tcol].isna().any():
        raise ValueError(f"nierozpoznane znaczniki czasu w {path.name}")
    df = df.set_index(tcol).sort_index()
    df.index.name = "time"
    # usuń puste/bezimienne kolumny
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df.apply(pd.to_numeric, errors="coerce")


def load_mill(m: int) -> pd.DataFrame:
    """Wczytuje dane młyna m (1..4) i ujednolica nazwy kolumn."""
    path = DANE / "Mlyny_202602" / f"MW{m}_202602.csv"
    df = _read_raw(path, "%d.%m.%Y %H:%M")
    rename = {}
    for tpl, name in MILL_SIGNALS.items():
        tag = tpl.format(m=m)
        if tag in df.columns:
            rename[tag] = name
    df = df.rename(columns=rename)
    # zostaw tylko zmapowane sygnały (pomija np. wtrącone tagi sąsiedniego młyna)
    df = df[[c for c in MILL_SIGNALS.values() if c in df.columns]]
    df["mlyn"] = m
    return df


def load_all_mills() -> dict[int, pd.DataFrame]:
    return {m: load_mill(m) for m in range(1, 5)}


def load_ogolne() -> pd.DataFrame:
    return _read_raw(DANE / "Mlyny_202602" / "Ogolne_202602.csv", "%d.%m.%Y %H:%M")


def load_kpi() -> pd.DataFrame:
    return _read_raw(DANE / "K2KPI_202602.csv", "%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    for m in range(1, 5):
        df = load_mill(m)
        print(f"MW{m}: {df.shape}, {df.index.min()} -> {df.index.max()}, "
              f"braki%: {df.isna().mean().mean()*100:.1f}")
        print("  kolumny:", ", ".join(df.columns[:50]))
    og = load_ogolne(); kpi = load_kpi()
    print("Ogolne:", og.shape, "KPI:", kpi.shape)
