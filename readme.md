# Master thesis repository
This repository hosts my master thesis codes and dataset.

![dro](assets/dro.svg)

---
The 8 pages summary of the approximately 108 page original thesis can be found from [this pdf](assets\MSc_sum.pdf).

---

## Brief Introduction:
This thesis discusses a data-driven decision making problem, in which the investors are
seeking to create and protect their wealth by investing in financial institutions. Due to various
factors such as economic and political conditions, financial institutions exhibit stochastic
behavior, requiring advanced mathematical models for effectively managing risks and returns
of assets.

Deep reinforcement learning is a framework that can uncover hidden patterns in the data
distribution and aid decision-making problems. The objective of this study is to enhance
the performance of WaveCorr, a specific deep reinforcement learning
model, with a distributionally robust end-to-end learning approach for portfolio optimization.
Specifically, the distributionally robust end-to-end learning approach takes into account the
risk of the model and the risk of return predictions, both of which are not considered by
WaveCorr alone.

In this approach, the decision-making layer optimizes the portfolio by solving a mini-
max problem, assuming that the distribution of asset returns belongs to an ambiguity set
centered around a nominal distribution. The model parameters are updated using implicit
differentiable optimization layers.

To evaluate these models, data from the 20 highest large capital companies listed on
the Tehran Stock Exchange between 2009 and 2022 is collected. The results indicate that
WaveCorr with the distributionally robust end-to-end learning approach improves the per-
formance of portfolio optimization.

---

Use the package manager pip to install dependencies:

```bash
pip install -r requirements.txt
```
---

You can run the experiment with:

```bash
python src/main.py
```
---

The assets' return is in the directory `dataset/`.


