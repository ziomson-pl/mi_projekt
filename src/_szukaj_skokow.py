# -*- coding: utf-8 -*-
"""Diagnostyka: które wartości zadane zmieniają się skokowo (z plateau)?"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_mill

for m in [3, 4]:
    df = load_mill(m)
    praca = df["moc_mlyna"] > 20
    for kol in ["went_zadana", "t_mieszanki_zadana"]:
        sp = df[kol]
        d = sp.diff().where(praca)
        for prog in [0.3, 0.5, 1.0, 2.0]:
            n_sk = int((d.abs() >= prog).sum())
            # ile z nich ma plateau 10 min przed i 20 min po
            czyste = 0
            for t in df.index[(d.abs() >= prog).fillna(False)]:
                i = df.index.get_loc(t)
                if i < 12 or i + 21 >= len(df):
                    continue
                if sp.iloc[i - 10:i].std() <= 0.05 and sp.iloc[i + 1:i + 21].std() <= 0.05:
                    czyste += 1
            print(f"MW{m} {kol}: |d|>={prog}: {n_sk} zmian, czystych skokow: {czyste}")
