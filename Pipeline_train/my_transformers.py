import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Transformer xử lý Outlier theo IQR
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}
        self.medians_ = {}

    def fit(self, X, y=None):
        # Tính toán Q1, Q3, IQR và Mean dựa trên tập X truyền vào (thường là Train)
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            self.medians_[col] = X[col].median()
        return self

    def transform(self, X):
        X_copy = X.copy().astype(float)
        for col in self.columns:
            lower_bound, upper_bound = self.bounds_[col]
            median_val = self.medians_[col]
            mask = (X_copy[col] < lower_bound) | (X_copy[col] > upper_bound)
            X_copy.loc[mask, col] = median_val
        return X_copy

# Transformer loại bỏ các cột có độ tương quan cao
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_)