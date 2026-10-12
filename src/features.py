"""Preprocessing pipeline shared by every model."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ], verbose_feature_names_out=False)
