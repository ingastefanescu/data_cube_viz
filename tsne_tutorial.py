# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 19:09:15 2024

@author: Andrei
"""


#%% Plotly 2D

from plotly.offline import plot
import plotly.graph_objs as go

fig = go.Figure(data=[go.Bar(y=[1, 3, 2])])
plot(fig, auto_open=True)

#%% Iris 3D

import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
plot(fig, auto_open=True)


#%% 3D classification 
import plotly.express as px
from sklearn.datasets import make_classification
from plotly.offline import plot
import plotly.graph_objs as go

X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=1500,
    n_informative=2,
    random_state=5,
    n_clusters_per_class=1,
)

fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y, opacity=0.8)
plot(fig, auto_open=True)

#%% PCA 6D -> 2D
 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca.score(X)

#%%  PCA Plot

fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y)
fig.update_layout(
    title="PCA visualization of Custom Classification dataset",
    xaxis_title="First Principal Component",
    yaxis_title="Second Principal Component",
)
plot(fig, auto_open=True)

#%% TSNE

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
tsne.kl_divergence_

#%% TSNE Viz

fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)
fig.update_layout(
    title="t-SNE visualization of Custom Classification dataset",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)

plot(fig, auto_open=True)

#%% Perplexity
import numpy as np

perplexity = np.arange(5,100, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, init="pca", perplexity=i)
    reduced = model.fit_transform(X)
    divergence.append(model.kl_divergence_)
fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
plot(fig, auto_open=True)
