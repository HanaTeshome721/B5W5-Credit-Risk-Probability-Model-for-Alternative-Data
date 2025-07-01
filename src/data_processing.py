import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from category_encoders.woe import WOEEncoder

from xverse.transformer import WOE as XverseWOE

# ---------------------
# Custom Transformers
# ---------------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col="CustomerId"):
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.group_col).agg({
            "Amount": ["sum", "mean", "std", "count"]
        }).reset_index()
        agg_df.columns = [self.group_col, "total_amount", "avg_amount", "std_amount", "txn_count"]
        return agg_df

class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col="TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df["txn_hour"] = df[self.time_col].dt.hour
        df["txn_day"] = df[self.time_col].dt.day
        df["txn_month"] = df[self.time_col].dt.month
        df["txn_year"] = df[self.time_col].dt.year
        return df[["CustomerId", "txn_hour", "txn_day", "txn_month", "txn_year"]]

class MergeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, Xs): return pd.concat(Xs, axis=1)

# ---------------------
# Main Processing Pipeline
# ---------------------

def build_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    # STEP 1: Aggregate Features
    agg_features = AggregateFeatures().fit_transform(raw_df)

    # STEP 2: Time-based Features
    time_features = TimeFeatures().fit_transform(raw_df)

    # STEP 3: Merge Features
    features_df = agg_features.merge(time_features, on="CustomerId", how="left")

    # STEP 4: Categorical Columns
    raw_df["ChannelId"] = raw_df["ChannelId"].fillna("Unknown")
    cat_df = raw_df[["CustomerId", "ChannelId", "ProductCategory"]].drop_duplicates(subset="CustomerId")
    
    # STEP 5: Merge categorical info
    full_df = features_df.merge(cat_df, on="CustomerId", how="left")

    # STEP 6: Handle Missing Values
    num_cols = ["total_amount", "avg_amount", "std_amount", "txn_count", "txn_hour", "txn_day", "txn_month", "txn_year"]
    cat_cols = ["ChannelId", "ProductCategory"]

    # Transformers
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X = full_df.drop(columns=["CustomerId"])
    X_transformed = preprocessor.fit_transform(X)

    # Reattach CustomerId if needed
    return X_transformed

# ---------------------
# WOE / IV (Optional)
# ---------------------

def apply_woe_iv(df, target_col="target"):
    # Example only — adjust as needed
    woe_model = WoE()
    woe_model.fit(df, df[target_col], ignore_columns=["CustomerId"])
    transformed = woe_model.transform(df)
    return transformed

def apply_xverse_woe(df, target_col="target"):
    model = XverseWOE()
    model.fit(df.drop(columns=[target_col]), df[target_col])
    transformed = model.transform(df.drop(columns=[target_col]))
    return transformed
