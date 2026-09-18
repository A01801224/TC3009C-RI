import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


class SpaceshipPreprocessor(BaseEstimator, TransformerMixin):
    """
    Replica el feature engineering del Avance 1 (Secciones 3-7) como un
    objeto reusable. fit() aprende estadísticas SOLO de los datos de
    entrenamiento; transform() las aplica a cualquier dato nuevo.
    """

    def __init__(self):
        self.bins_cabin = [-np.inf, 299, 599, 899, 1199, 1499, np.inf]
        self.labels_cabin = ["0-299", "300-599", "600-899", "900-1199", "1200-1499", "1500+"]

    def fit(self, X, y=None):
        df = X.copy()
        df = df.drop(columns=["Name", "VIP"], errors="ignore")

        df["Deck"] = df["Cabin"].str.split("/").str[0]
        df["CabinNum"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce")
        df["Side"] = df["Cabin"].str.split("/").str[2]
        gastos_totales = df[SPEND_COLS].sum(axis=1)
        df.loc[df["CryoSleep"].isnull() & (gastos_totales > 0), "CryoSleep"] = False

        self.medianas_ = {col: df[col].median() for col in ["Age", "CabinNum"] + SPEND_COLS}
        self.modas_ = {col: df[col].mode()[0] for col in ["HomePlanet", "Destination", "CryoSleep", "Deck", "Side"]}

        df_imputado = self._imputar(df)
        df_codificado = self._codificar(df_imputado)
        self.cols_to_scale_ = ["Age", "group_size"] + SPEND_COLS
        self.scaler_ = RobustScaler()
        self.scaler_.fit(df_codificado[self.cols_to_scale_])

        self.columnas_finales_ = self._escalar(df_codificado.copy()).drop(columns=["Transported"], errors="ignore").columns
        return self

    def transform(self, X):
        df = X.copy()
        df = df.drop(columns=["Name", "VIP"], errors="ignore")

        df["Deck"] = df["Cabin"].str.split("/").str[0]
        df["CabinNum"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce")
        df["Side"] = df["Cabin"].str.split("/").str[2]
        gastos_totales = df[SPEND_COLS].sum(axis=1)
        df.loc[df["CryoSleep"].isnull() & (gastos_totales > 0), "CryoSleep"] = False

        df = self._imputar(df)
        df = self._codificar(df)
        df = self._escalar(df)

        target = df["Transported"] if "Transported" in df.columns else None
        df = df.reindex(columns=self.columnas_finales_, fill_value=0)
        if target is not None:
            df["Transported"] = target
        return df

    def _imputar(self, df):
        for col in SPEND_COLS:
            df.loc[(df["CryoSleep"] == True) & (df[col].isnull()), col] = 0.0
        for col in ["Age", "CabinNum"] + SPEND_COLS:
            df[col] = df[col].fillna(self.medianas_[col])
        for col in ["HomePlanet", "Destination", "CryoSleep", "Deck", "Side"]:
            df[col] = df[col].fillna(self.modas_[col])
        return df

    def _codificar(self, df):
        df["group"] = df["PassengerId"].str.split("_").str[0]
        df["group_size"] = df["group"].map(df["group"].value_counts())
        df.drop(columns=["Cabin", "PassengerId", "group"], inplace=True, errors="ignore")

        df["CabinNumBin"] = pd.cut(df["CabinNum"], bins=self.bins_cabin, labels=self.labels_cabin)
        df.drop(columns=["CabinNum"], inplace=True)

        df["HasSpent"] = (df[SPEND_COLS].sum(axis=1) > 0).astype(int)
        df["CryoSleep"] = df["CryoSleep"].astype(int)
        df["Side"] = df["Side"].map({"P": 0, "S": 1})
        if "Transported" in df.columns:
            df["Transported"] = df["Transported"].astype(int)

        df = pd.get_dummies(df, columns=["HomePlanet", "Destination", "Deck", "CabinNumBin"], drop_first=True, dtype=int)
        return df

    def _escalar(self, df):
        for col in SPEND_COLS:
            df[col] = np.log1p(df[col])
        df[self.cols_to_scale_] = self.scaler_.transform(df[self.cols_to_scale_])
        return df