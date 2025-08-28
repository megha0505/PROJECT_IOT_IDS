from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.tree import DecisionTreeClassifier


@dataclass
class FeatureSelector:
    k_chi2: int = 30
    rfe_final: int = 15
    dt_max_depth: int = 6

    def build_preprocessor(self, X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ])
        return pre, num_cols, cat_cols

    def select(self, Xtr, ytr, preproc: ColumnTransformer) -> np.ndarray:
        Xp = preproc.fit_transform(Xtr)
        k = min(self.k_chi2, Xp.shape[1])
        chi = SelectKBest(chi2, k=k).fit(Xp, ytr)
        mask1 = chi.get_support()
        base = Xp[:, mask1] if mask1.sum() > 0 else Xp
        final = min(self.rfe_final, int(mask1.sum()) if mask1.sum() > 0 else k)
        rfe = RFE(DecisionTreeClassifier(max_depth=self.dt_max_depth, random_state=42), n_features_to_select=final)
        rfe.fit(base, ytr)

        sel_mask = np.zeros(Xp.shape[1], dtype=bool)
        idx_base = np.where(mask1)[0] if mask1.sum() > 0 else np.arange(Xp.shape[1])
        sel_mask[idx_base[rfe.get_support()]] = True
        return sel_mask