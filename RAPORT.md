# Predykcja zapchania młyna węglowego (kruszarki) — raport z analizy

Data analizy: 10.06.2026 • Dane: blok 1, luty 2026 (01–27.02, rozdzielczość 1 min)

## 1. Co jest w danych

| Plik | Zawartość | Wymiar |
|---|---|---|
| `dane/Mlyny_202602/MW1..MW4_202602.csv` | sygnały 4 zespołów młynowych (młyny wentylatorowe z podajnikami węgla) | 37 440 min × ~38 sygnałów |
| `dane/Mlyny_202602/Ogolne_202602.csv` | sygnały ogólnokotłowe (O₂, NH₃, temperatury spalin/powietrza) | 37 440 × 38 |
| `dane/K2KPI_202602.csv` | KPI kotła (para, CO, NOx, O₂) | 37 440 × 19 |
| `dane/Punkty_blok_2.xlsx` | lista punktów pomiarowych (tagi KKS) | — |
| `dane/System_Screens.pptx` | zrzuty ekranów DCS (topologia układu) | 12 slajdów |
| PDF-y + `efficiency estimates*.xlsx` | metodyka strat kotła (Siegert) — **wątek poboczny**, nie dotyczy młynów | — |

Kluczowe sygnały per młyn (ujednolicone w `src/loader.py`): moc silnika młyna,
prędkość i prąd podajnika węgla, pomiar/zadana/sterowanie UAR wentylacji młyna,
pozycja kierownicy wentylatora, temperatura mieszanki pyłowo-powietrznej (+zadana,
+sterowanie), klapy zimnego/gorącego powietrza, ciśnienia, temperatury łożysk i komór.

Pułapki techniczne: kodowanie cp1250, mieszane formaty czasu w obrębie plików
(`dd.mm.yyyy HH:MM` i ISO), braki jako `N_A`/`NaN`, ~3% braków.

## 2. Jak wygląda zapchanie w tych danych (EDA)

W lutym **wszystkie potwierdzone epizody wystąpiły na MW3 i MW4** (MW1/MW2 czyste).
Sekwencja jest powtarzalna i fizycznie spójna:

1. regulator wentylacji młyna stopniowo otwiera kierownicę aż do **wysycenia (100%)**,
2. mimo to **przepływ wentylacji spada** poniżej wartości zadanej (rosnący opór =
   zapychanie się młyna/pyłoprzewodów), nawet z 15 do 3 kNm³/h,
3. **moc młyna rośnie** (przeładowanie komory mielenia, np. 65 → 90 kW),
   **temperatura mieszanki spada** poniżej zadanej,
4. automatyka/operator **awaryjnie odcina podajnik węgla** (piłokształtne cięcia
   prędkości podajnika), młyn się oczyszcza i wraca do normy.

Prekursory (wysycanie sterowania, narastający deficyt przepływu) są widoczne
**60–120 min przed kulminacją** — jest fizyczna podstawa do wczesnego ostrzegania.

Zidentyfikowane epizody (definicja niżej): **5 silnych** — MW3: 05.02 ~12:00,
09.02 ~04:40, 11.02 ~09:40, 15.02 ~02:40; MW4: 19.02 ~18:15 — oraz kilka
mikro-zdarzeń (nagłe odcięcia/tripy bez fazy narastania, np. MW3 10.02 22:36).
Wykresy: `work/plots/zoom_*.png`, pełny przegląd: `work/plots/przeglad_MW*.png`.

## 3. Definicja zdarzenia (etykiety — `src/labels.py`)

Brak dziennika zdarzeń w danych ⇒ etykiety wyznaczone sygnałowo. Zdarzenie =
(w czasie pracy młyna, moc > 20 kW):

- **D1** deficyt wentylacji > 20% wartości zadanej przez ≥ 5 min przy sterowaniu ≥ 98%,
- **D2** awaryjne cięcie podajnika (spadek ≥ 25 pkt w ≤ 3 min) przy dalej pracującym młynie,
- **D3** moc młyna > kwantyl 99,5% przy deficycie > 10%.

Epizody sklejane przy przerwie < 90 min. Epizod **silny** = ≥ 3 min warunku D1
(ma fazę narastania ⇒ jest przewidywalny). Mikro-zdarzenia D2/D3 bez prekursorów
wyłączono z uczenia i oceny (nieprzewidywalne z 1-minutowych danych procesowych).

## 4. Przygotowanie danych pod model (`src/features.py`)

- wszystkie młyny sklejone w jeden zbiór (cechy niezależne od numeru młyna),
- 39 cech liczonych wyłącznie z przeszłości: poziomy, zmiany (15/30/60 min),
  średnie kroczące, w tym cechy fizyczne:
  - **deficyt** = (zadana − pomiar)/zadana wentylacji,
  - **drożność** = przepływ / wyjście regulatora (proxy oporu przepływu),
  - **moc_na_wegiel** = moc młyna / prędkość podajnika (energochłonność mielenia),
  - minuty wysycenia sterowania w ostatniej godzinie, niestabilność mocy (std 30 min),
- etykieta: `y(t) = 1` gdy silny epizod zacznie się w ciągu **60 min** (horyzont ostrzegania),
- wykluczenia: postoje, 60 min po rozruchu, czas trwania epizodu + 60 min rekonwalescencji.

Zbiór: 131 719 minut, 300 pozytywnych (0,23%), 5 epizodów.

## 5. Wybór modelu i wyniki (`src/train.py`, `src/train2.py`)

**Wybrany model: LightGBM (gradient boosting) + reguła fizyczna jako zabezpieczenie.**

Dlaczego nie sieć neuronowa (LSTM/TCN na RTX 2070 Super): mamy **5 zdarzeń
z jednego miesiąca** — to o 1–2 rzędy wielkości za mało na uczenie głębokie;
boosting na ręcznie zbudowanych cechach fizycznych jest przy tej liczbie zdarzeń
odporniejszy, interpretowalny (ważności cech) i trenuje się w sekundy na CPU.
GPU nie jest potrzebne na tym etapie (dane miesiąca to ~40 MB).

Walidacja: leave-one-day-out (27 foldów, grupowanie po dniach ⇒ bez wycieku czasowego).

| Wariant | Wykryte epizody | Śr. wyprzedzenie | Fałszywe serie / dobę / młyn |
|---|---|---|---|
| **LightGBM, próg 0,5** | **5/5** | **30 min** | **0,06** (6 serii / 27 dni / 4 młyny) |
| LightGBM, próg 0,9 | 3/5 | 30 min | 0,04 |
| reguła fizyczna* | 5/5 (wcześniej 5/6) | 49 min | 0,07 |
| hybryda (model LUB reguła) | 5/5 | **51 min** | 0,09 |

\* reguła: ≥20 min wysycenia sterowania w godzinie **i** średni deficyt 15 min > 8%.

Metryki minutowe (LODO): ROC-AUC 0,95, PR-AUC 0,36 (przy 0,23% klasy pozytywnej).
Alarm zgłaszany dopiero po 3 kolejnych minutach score ≥ próg (tłumi piki).

Najważniejsze cechy: pomiar przepływu wentylacji (poziom i średnia 15 min),
średnia moc młyna 60 min, zmiana odchyłki temperatury mieszanki (30 min),
liczba minut deficytu w 30 min, niestabilność mocy, drożność.

## 6. Artefakty

```
src/loader.py        # wczytywanie + ujednolicenie sygnałów
src/eda_overview.py  # przeglądy miesięczne
src/eda_events.py    # detekcja kandydatów + zbliżenia
src/labels.py        # definicja zdarzenia zapchania
src/features.py      # zbiór uczący (cechy + etykiety)
src/train.py         # CV 5-fold, ocena zdarzeniowa, baseline regułowy
src/train2.py        # walidacja leave-one-day-out, hybryda, diagnostyka per epizod
src/predict.py       # finalny model + inferencja score_mlyna(df)
work/model_zapchanie.joblib   # gotowy model (próg 0,5, seria 3 min)
work/epizody_zapchania.csv    # lista epizodów
work/plots/*.png              # przeglądy, zbliżenia, przebieg score
```

Uruchomienie od zera: `python3 src/features.py && python3 src/train2.py && python3 src/predict.py --demo`

## 7. Ograniczenia i co dalej

1. **Etykiety są heurystyczne** — najpilniejsze jest potwierdzenie epizodów
   z obsługą/dziennikiem zdarzeń DCS (czy 5 znalezionych epizodów to faktycznie
   zapchania, czy któreś przegapiliśmy).
2. **5 zdarzeń / 1 miesiąc / 2 młyny** — model na granicy istotności statystycznej.
   Potrzeba min. 6–12 miesięcy danych (sezonowość paliwa, różna wilgotność węgla).
   Przy >50 zdarzeniach warto przetestować modele sekwencyjne (TCN/LSTM — wtedy
   RTX 2070 Super się przyda).
3. Dane ogólnokotłowe i KPI nie weszły jeszcze do cech (obciążenie bloku,
   jakość węgla) — naturalne rozszerzenie.
4. Do wdrożenia online: liczenie cech w oknie kroczącym na strumieniu 1-min
   z DCS (model przelicza się w ms na CPU), alarm po 3 min score ≥ 0,5,
   równolegle reguła fizyczna jako niezależny watchdog (hybryda dała najdłuższe
   wyprzedzenie: ~51 min).
5. Terminologia: dane opisują **młyny wentylatorowe (MW1–4)** — przyjęto, że
   „kruszarka węgla" w zadaniu odnosi się do zespołu mielącego młyna. Jeśli
   chodziło o odrębną kruszarkę w ciągu nawęglania, w dostarczonych danych
   nie ma jej sygnałów.
