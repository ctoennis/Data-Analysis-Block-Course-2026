# Answer Key

## Part 1 — Threshold \(T>37.5^\circ\mathrm{C}\)

Positive: patients **6, 7, 8, 9, 10**.

|                         | Actually infected | Not infected |
| ----------------------- | ----------------: | -----------: |
| **Classified positive** |            TP = 4 |       FP = 1 |
| **Classified negative** |            FN = 1 |       TN = 4 |

---

## Part 2 — Sensitivity

$$
\text{Sensitivity}
=\frac{TP}{TP+FN}
=\frac{4}{5}
=\boxed{0.80=80\%}
$$

Thus, **80% of the infected patients are correctly identified**.

---

## Part 3 — Signal and background efficiency

Signal efficiency:

$$
\epsilon_S=\frac{4}{5}=\boxed{80\%}
$$

Background efficiency:

$$
\epsilon_B=\frac{1}{5}=\boxed{20\%}
$$

Therefore, the background rejection is

$$
1-\epsilon_B=\boxed{80\%}.
$$

In this example, **sensitivity = signal efficiency**.

---

## Part 4 — Gini index

### All 10 patients

There are 5 signal and 5 background patients:

$$
p_S=p_B=0.5
$$

$$
G=1-(0.5)^2-(0.5)^2
=\boxed{0.5}
$$

### Positive group

There are 4 signal and 1 background:

$$
p_S=0.8,\qquad p_B=0.2
$$

$$
G=1-(0.8)^2-(0.2)^2
=\boxed{0.32}
$$

### Negative group

There are 1 signal and 4 background:

$$
p_S=0.2,\qquad p_B=0.8
$$

$$
G=1-(0.2)^2-(0.8)^2
=\boxed{0.32}
$$

A **lower Gini index means the group is more pure**, i.e. it contains predominantly one class.

---

## Part 5 — Threshold study

|                  Threshold | Signal efficiency | Background efficiency | Signal rejected | Background rejected |
| -------------------------: | ----------------: | --------------------: | --------------: | ------------------: |
| \(T>36.5^\circ\mathrm{C}\) |              100% |                   80% |              0% |                 20% |
| \(T>37.0^\circ\mathrm{C}\) |              100% |                   60% |              0% |                 40% |
| \(T>37.5^\circ\mathrm{C}\) |               80% |                   20% |             20% |                 80% |
| \(T>38.0^\circ\mathrm{C}\) |               40% |                   20% |             60% |                 80% |
| \(T>38.5^\circ\mathrm{C}\) |               20% |                    0% |             80% |                100% |

As the threshold increases, **both signal efficiency and background efficiency generally decrease**. A higher threshold rejects more patients, including both background and, eventually, signal.

There is therefore a **trade-off between signal retention and background rejection**. The appropriate threshold depends on the consequences of false positives and false negatives.

---

## Part 6 — Interpretation

The points on a signal-efficiency vs. background-efficiency plot illustrate the **threshold trade-off**.

* Lower threshold → more signal retained, but more background accepted.
* Higher threshold → less background accepted, but more signal lost.
* There is no threshold that simultaneously maximizes signal efficiency and minimizes background efficiency for this dataset.

The example illustrates the central idea of **binary classification**: changing the decision threshold changes the balance between correctly identifying the desired class and rejecting the unwanted class.

