# Predykcja zapchania młyna węglowego — projekt na Metody Identyfikacji

Autorzy: **Jakub Wróblewski, Mateusz Smardzewski**
Politechnika Warszawska, Wydział Elektroniki i Technik Informacyjnych, Automatyka i Robotyka

Celem projektu jest wczesne ostrzeganie przed zapchaniem (dławieniem) młyna
węglowego w elektrociepłowni na podstawie minutowych danych procesowych DCS
(blok 1, luty 2026, młyny MW1–MW4). Projekt obejmuje analizę identyfikacyjną
obiektu (charakterystyka statyczna, ARX/RLS, odpowiedzi skokowe), definicję
zdarzenia, inżynierię cech oraz porównanie modeli predykcyjnych
(reguła fizyczna, LightGBM, MLP, LSTM).

## Wymagania

- Python ≥ 3.12 (testowane w WSL2 Ubuntu 24.04, 16 GB RAM hosta; GPU niepotrzebne)
- pakiety: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`,
  `openpyxl`, `scipy`, `torch` (CPU; tylko do eksperymentu NN), `joblib`

```bash
pip3 install --user pandas numpy scikit-learn lightgbm matplotlib openpyxl scipy
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
# na Ubuntu 24.04 (PEP 668) dodaj flagę: --break-system-packages
```

- do kompilacji raportu PDF: dowolna dystrybucja LaTeX (MiKTeX / TeX Live)
  z pakietami `babel-polish`, `listings`, `booktabs`

## Dane

Katalog `dane/` (nie wersjonowany):

```
dane/
├── Mlyny_202602/
│   ├── MW1_202602.csv ... MW4_202602.csv   # sygnały młynów, 1 min, 01–27.02.2026
│   └── Ogolne_202602.csv                   # sygnały ogólnokotłowe
├── K2KPI_202602.csv                        # KPI kotła
├── Punkty_blok_2.xlsx                      # lista punktów pomiarowych (KKS)
└── System_Screens.pptx                     # zrzuty ekranów DCS
```

Uwaga: pliki CSV mają kodowanie cp1250, **mieszane formaty czasu**
(`dd.mm.yyyy HH:MM` i ISO w tym samym pliku) oraz braki `N_A`/`NaN` —
wszystko obsługuje `src/loader.py`.

## Uruchomienie (kolejność)

```bash
cd mi_projekt

# 1. sanity check wczytywania danych
python3 src/loader.py

# 2. (opcjonalnie) przeglądy EDA i detekcja kandydatów zdarzeń
python3 src/eda_overview.py
python3 src/eda_events.py

# 3. etykiety zdarzeń (lista epizodów -> work/epizody_zapchania.csv)
python3 src/labels.py

# 4. zbiór uczący: cechy + etykiety (-> work/zbior_uczacy.pkl)
python3 src/features.py

# 5. trening + ocena (5-fold po dniach oraz leave-one-day-out)
python3 src/train.py
python3 src/train2.py

# 6. eksperyment z sieciami neuronowymi (MLP, LSTM; ~10 min na CPU)
python3 src/nn_experiment.py

# 7. analiza identyfikacyjna (charakterystyka statyczna, RLS, FOPDT)
python3 src/identyfikacja.py

# 8. model finalny + demo inferencji (-> work/model_zapchanie.joblib)
python3 src/predict.py --demo

# 9. ryciny do raportu (-> work/plots/raport/)
python3 src/plots_raport.py
```

Kompilacja raportu (z katalogu `raport/`):

```bash
pdflatex raport.tex && pdflatex raport.tex   # dwa przebiegi (spis treści, odnośniki)
```

## Struktura projektu

```
src/loader.py         # wczytywanie i ujednolicenie sygnałów (tagi KKS -> nazwy)
src/eda_overview.py   # miesięczne przeglądy sygnałów per młyn
src/eda_events.py     # detekcja kandydatów na zdarzenia + wykresy zbliżeń
src/labels.py         # sygnałowa definicja zapchania (D1/D2/D3) i epizody
src/features.py       # 39 cech + etykieta "zdarzenie w ciągu 60 min"
src/train.py          # LightGBM, walidacja 5-fold po dniach, ocena zdarzeniowa
src/train2.py         # walidacja leave-one-day-out, hybryda model+reguła
src/nn_experiment.py  # MLP i LSTM vs LightGBM (identyczna walidacja)
src/identyfikacja.py  # charakterystyka statyczna, RLS/ARX, odpowiedź skokowa FOPDT
src/predict.py        # model finalny + funkcja inferencji score_mlyna()
src/plots_raport.py   # ryciny do raportu
raport/raport.tex     # raport końcowy (LaTeX)
work/                 # artefakty: model, wyniki json/csv, wykresy
```

## Najważniejsze wyniki (luty 2026)

- 5 silnych epizodów zapchania (MW3 ×4, MW4 ×1); wszystkie poprzedzone
  ≥ 60 min narastającym deficytem wentylacji przy wysyconym regulatorze.
- LightGBM (leave-one-day-out): **5/5 wykrytych epizodów, średnio 30 min
  wyprzedzenia, 0,06 fałszywych serii / dobę / młyn** (próg 0,5, seria 3 min).
- Hybryda z regułą fizyczną wydłuża wyprzedzenie do ~51 min.
- Sieci neuronowe (MLP/LSTM) przy 5 zdarzeniach nie dają spójnej przewagi,
  a ich uczenie jest niestabilne — szczegóły w raporcie (rozdz. o NN).
- Identyfikacyjnie: zapchanie = spadek wzmocnienia statycznego (drożności)
  kanału kierownica→przepływ (MW3: 0,51→0,28; MW4: 0,92→0,15).

## Zastrzeżenia

Etykiety zdarzeń wyznaczono sygnałowo (brak dziennika zdarzeń) — wymagają
potwierdzenia przez obsługę bloku. Miesiąc danych i 5 zdarzeń to za mało na
produkcyjny model; wyniki traktować jako studium wykonalności.
