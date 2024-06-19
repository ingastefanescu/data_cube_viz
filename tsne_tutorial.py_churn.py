# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 19:37:47 2024

@author: Andrei
"""

#%% CHURN Loading
import pandas as pd

df = pd.read_csv("churn.csv")
df.head(3)

#%% PCA


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, random_state=13, test_size=0.25, shuffle=True
)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

pca.score(X_test)

#%% PCA Viz
import plotly.express as px
from plotly.offline import plot

fig = px.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], color=y_train)
fig.update_layout(
    title="PCA visualization of Customer Churn dataset",
    xaxis_title="First Principal Component",
    yaxis_title="Second Principal Component",
)

plot(fig, auto_open=True)
