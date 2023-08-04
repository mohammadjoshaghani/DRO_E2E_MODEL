# Master thesis repository
This repository hosts my master thesis codes and dataset.

![dro](assets/dro.svg)

## Brief Introduction:
This thesis discusses a data-driven decision making problem, in which the investors are seeking to create and protect their wealth by investing in financial institutions. Due to various factors such as economic and political conditions, financial institutions exhibit stochastic behavior, and effectively managing the risks and returns of assets requires advanced mathematical models.

Deep reinforcement learning is a framework that can uncover hidden patterns in data distributions and aids decision making problem. This study's objective is to enhance the performance of WaveCorr, a specific deep reinforcement learning model, with a distributionally robust end-to-end learning approach for portfolio optimization. Specifically, the distributionally robust end-to-end learning approach takes into account the risk of model and the risk of returns prediction which are both dismissed from WaveCorr by itself.

In this approach, the decision-making layer optimizes the portfolio by solving a minimax problem, assuming that the distribution of asset returns belongs to an ambiguity set centered around a nominal distribution. The model parameters are updated using implicit differentiable optimization layers.

To evaluate these models, data from 20 highest large capital companies listed on the Tehran Stock Exchange between 2009 and 2022 is collected. The results indicate that enhancing WaveCorr with the distributionally robust end-to-end learning approach improves the performance of portfolio optimization.


# Methodology

lets assume $\boldsymbol{x}_t \in \mathbb{R}^m$ be $m$ predictive features at time $t$, and we want to predict $n$ assents returns $\boldsymbol{\hat{y}}_t \in \mathbb{R}^n$. Let $\{\boldsymbol{x}_j \in \mathbb{R}^m : j = t-T,...,t-1\}$ be time series of predictive features, and $\{\boldsymbol{y}_j \in \mathbb{R}^n : j = t-T,...,t-1\}$ time series of assets returns for $T$ time steps. Let $g_\theta : \mathbb{R}^m \to \mathbb{R}^n$، be predictive model that maps $\boldsymbol{x}_t$ to $\mathbb{E}[\boldsymbol{\hat{y}_t}]$. We take $g_\theta$ is differentiable on $\theta$ and  $\boldsymbol{\hat{y}}_t \overset{\Delta}{=} g_\theta(\boldsymbol{x}_t)$.

<!-- ![eq](assets/equation.svg) -->

<!-- ![wave](assets/wavecorr.JPG) -->

<!-- ![corr](assets/corr0.JPG) -->




