# -*- coding: utf-8 -*-
"""Eksperyment: sieci neuronowe vs LightGBM przy 5 zdarzeniach / 1 miesiącu danych.

Cel: empiryczna weryfikacja hipotezy, że przy tak małej liczbie zdarzeń
uczenie głębokie nie ma przewagi nad boostingiem na cechach fizycznych.

Warianty (identyczna walidacja GroupKFold po dniach, identyczna ocena):
  1. LightGBM na 39 cechach inżynierskich      (referencja)
  2. MLP (64-32) na tych samych 39 cechach     (czy architektura NN coś wnosi?)
  3. LSTM na surowych sekwencjach 60 min x 12  (czy NN sama "odkryje" cechy?)

Wyniki -> work/wyniki_nn.json + krzywe uczenia -> work/nn_learning_curves.csv
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_all_mills
from src.train import cross_val_scores, wygladz, ocena_zdarzeniowa
from src.features import HORYZONT_MIN, REKONWALESCENCJA_MIN, ROZRUCH_MIN
from src.labels import maski_zdarzen, epizody
from src.features import MIN_D1_SILNY

ROOT = Path(__file__).resolve().parent.parent
META = ["y", "mlyn", "epizod_id", "dzien"]
SEED = 42
N_SPLITS = 5
OKNO_LSTM = 60
SYGNALY_LSTM = ["went_pomiar", "went_zadana", "went_sterowanie", "kierownica",
                "moc_mlyna", "podajnik_predkosc", "podajnik_prad", "t_mieszanki",
                "t_mieszanki_zadana", "klapa_goraca", "p_mlyn", "p_przed_went"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- MLP na cechach
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cv_mlp(zb: pd.DataFrame, epoki=30, lr=1e-3):
    X = zb.drop(columns=META).values.astype(np.float32)
    y = zb["y"].values.astype(np.float32)
    groups = zb["dzien"].values
    oof = np.full(len(zb), np.nan)
    krzywe = []
    for fold, (tr, te) in enumerate(GroupKFold(N_SPLITS).split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        Xtr = torch.tensor(sc.transform(X[tr]), device=DEVICE)
        ytr = torch.tensor(y[tr], device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]), device=DEVICE)
        pos_w = torch.tensor(min((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), 30.0),
                             device=DEVICE)
        model = MLP(X.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        n = len(Xtr)
        for ep in range(epoki):
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 4096):
                b = perm[i:i + 4096]
                opt.zero_grad()
                loss = lossf(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            # AP na zbiorze testowym po każdej epoce (tylko do krzywej uczenia)
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xte)).cpu().numpy()
            krzywe.append({"model": "MLP", "fold": fold, "epoka": ep,
                           "loss_tr": tot / n,
                           "ap_te": float(average_precision_score(y[te], p))
                           if y[te].sum() > 0 else np.nan})
        oof[te] = p
    return oof, pd.DataFrame(krzywe)


# ------------------------------------------------------------- LSTM na sekwencjach
class LSTMNet(nn.Module):
    def __init__(self, n_sig, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_sig, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def zbuduj_sekwencje():
    """Okna 60 min surowych sygnałów, etykiety i wykluczenia jak w features.py."""
    mills = load_all_mills()
    Xs, ys, dnie, mlyny, epizody_id, czasy = [], [], [], [], [], []
    for m, df in mills.items():
        df = df.interpolate(limit=5)
        z = maski_zdarzen(df)
        ep = epizody(z["zdarzenie"])
        sila = [int(z.loc[r.start:r.koniec, "d1"].sum()) for _, r in ep.iterrows()]
        ep["silny"] = pd.Series(sila, index=ep.index) >= MIN_D1_SILNY

        y = pd.Series(0, index=df.index, dtype=int)
        wyklucz = ~z["praca"]
        start_pracy = z["praca"] & ~z["praca"].shift(1).fillna(False).astype(bool)
        for t in z.index[start_pracy]:
            wyklucz.loc[t:t + pd.Timedelta(minutes=ROZRUCH_MIN)] = True
        eid = pd.Series(-1, index=df.index, dtype=int)
        for i, r in ep.iterrows():
            if r.silny:
                y.loc[r.start - pd.Timedelta(minutes=HORYZONT_MIN):
                      r.start - pd.Timedelta(minutes=1)] = 1
                okno = (df.index >= r.start - pd.Timedelta(minutes=HORYZONT_MIN)) & (df.index < r.start)
                eid[okno] = i + m * 100
            else:
                wyklucz.loc[r.start - pd.Timedelta(minutes=HORYZONT_MIN):r.start] = True
            wyklucz.loc[r.start:r.koniec + pd.Timedelta(minutes=REKONWALESCENCJA_MIN)] = True

        S = df[SYGNALY_LSTM].ffill().fillna(0).values.astype(np.float32)
        ok = (~wyklucz).values
        for i in range(OKNO_LSTM, len(df)):
            if not ok[i]:
                continue
            Xs.append(S[i - OKNO_LSTM:i])
            ys.append(y.iloc[i])
            dnie.append(str(df.index[i].date()))
            mlyny.append(m)
            epizody_id.append(eid.iloc[i])
            czasy.append(df.index[i])
    return (np.stack(Xs), np.array(ys, dtype=np.float32), np.array(dnie),
            np.array(mlyny), np.array(epizody_id), pd.DatetimeIndex(czasy))


def cv_lstm(X, y, dnie, epoki=15, lr=1e-3, neg_na_poz=200):
    oof = np.full(len(y), np.nan)
    krzywe = []
    for fold, (tr, te) in enumerate(GroupKFold(N_SPLITS).split(X, y, dnie)):
        # normalizacja per sygnał wg foldu treningowego
        mu = X[tr].reshape(-1, X.shape[2]).mean(0)
        sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
        # podpróbkowanie negatywów (przyspieszenie; wszystkie pozytywy zostają)
        rng = np.random.default_rng(SEED + fold)
        poz = tr[y[tr] == 1]
        neg = tr[y[tr] == 0]
        neg = rng.choice(neg, size=min(len(neg), neg_na_poz * max(len(poz), 1)),
                         replace=False)
        trs = np.concatenate([poz, neg])
        Xtr = torch.tensor((X[trs] - mu) / sd, device=DEVICE)
        ytr = torch.tensor(y[trs], device=DEVICE)
        pos_w = torch.tensor(min(len(neg) / max(len(poz), 1), 30.0), device=DEVICE)
        model = LSTMNet(X.shape[2]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        n = len(Xtr)
        for ep in range(epoki):
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = lossf(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            model.eval()
            with torch.no_grad():
                ps = []
                for i in range(0, len(te), 4096):
                    xb = torch.tensor((X[te[i:i + 4096]] - mu) / sd, device=DEVICE)
                    ps.append(torch.sigmoid(model(xb)).cpu().numpy())
                p = np.concatenate(ps)
            krzywe.append({"model": "LSTM", "fold": fold, "epoka": ep,
                           "loss_tr": tot / n,
                           "ap_te": float(average_precision_score(y[te], p))
                           if y[te].sum() > 0 else np.nan})
            print(f"  LSTM fold {fold} epoka {ep}: loss {tot/n:.4f}, AP_te {krzywe[-1]['ap_te']:.4f}")
        oof[te] = p
    return oof, pd.DataFrame(krzywe)


def metryki(nazwa, zb_like, oof, progi=(0.3, 0.5, 0.7)):
    m = ~np.isnan(oof)
    wyn = {"model": nazwa,
           "pr_auc": float(average_precision_score(zb_like.y[m], oof[m])),
           "roc_auc": float(roc_auc_score(zb_like.y[m], oof[m])),
           "zdarzeniowo": []}
    for prog in progi:
        r = ocena_zdarzeniowa(zb_like, oof, prog)
        wyn["zdarzeniowo"].append(r)
    return wyn


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    wyniki = []
    zb = pd.read_pickle(ROOT / "work" / "zbior_uczacy.pkl")

    print("\n[1/3] LightGBM (referencja, 5-fold po dniach)...")
    oof_gbm, _ = cross_val_scores(zb, n_splits=N_SPLITS)
    oof_gbm = wygladz(zb, oof_gbm)
    wyniki.append(metryki("LightGBM", zb, oof_gbm))

    print("[2/3] MLP na tych samych cechach...")
    oof_mlp, k_mlp = cv_mlp(zb)
    oof_mlp = wygladz(zb, oof_mlp)
    wyniki.append(metryki("MLP_cechy", zb, oof_mlp))

    print("[3/3] LSTM na surowych sekwencjach 60 min...")
    X, y, dnie, mlyny, eid, czasy = zbuduj_sekwencje()
    print(f"  sekwencje: {X.shape}, pozytywne: {int(y.sum())}")
    oof_lstm, k_lstm = cv_lstm(X, y, dnie)
    zb_lstm = pd.DataFrame({"y": y.astype(int), "mlyn": mlyny,
                            "epizod_id": eid, "dzien": dnie}, index=czasy)
    oof_lstm = wygladz(zb_lstm, oof_lstm)
    wyniki.append(metryki("LSTM_sekwencje", zb_lstm, oof_lstm))

    pd.concat([k_mlp, k_lstm]).to_csv(ROOT / "work" / "nn_learning_curves.csv", index=False)
    np.save(ROOT / "work" / "oof_mlp.npy", oof_mlp)
    np.save(ROOT / "work" / "oof_lstm.npy", oof_lstm)
    zb_lstm.assign(score=oof_lstm).to_pickle(ROOT / "work" / "zb_lstm.pkl")
    with open(ROOT / "work" / "wyniki_nn.json", "w") as f:
        json.dump(wyniki, f, indent=2, default=str)

    print("\n=== PODSUMOWANIE ===")
    for w in wyniki:
        print(f"\n{w['model']}: PR-AUC={w['pr_auc']:.3f} ROC-AUC={w['roc_auc']:.3f}")
        for r in w["zdarzeniowo"]:
            print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
