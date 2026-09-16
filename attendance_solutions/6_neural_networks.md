# Teacher Answer Key: Regularization of Neural Networks

## Part 1 — Predictions and classification

Using the threshold \(\hat y\geq0.5\):

* Class 1: images **1, 3, 5, 8**
* Class 0: images **2, 4, 6, 7, 9, 10**
* Incorrect: **image 10 only**

Accuracy:

$$
\frac{9}{10}=\boxed{90\%}
$$

Accuracy alone does not capture prediction confidence, so it cannot fully characterize model performance or overtraining.

---

## Part 2 — Binary cross-entropy

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

| Case | \(y\) | \(\hat y\) |       BCE |
| :--- | ----: | ---------: | --------: |
| A    |     1 |       0.90 | **0.105** |
| B    |     1 |       0.51 | **0.673** |
| C    |     0 |       0.10 | **0.105** |
| D    |     0 |       0.90 | **2.303** |

* Lowest loss: **A and C**
* Highest loss: **D**
* Confidently wrong predictions receive a very large penalty.
* BCE incorporates **prediction confidence**, unlike accuracy.

---

## Part 3 — Average BCE

Approximate individual losses:

| Example | \(y\) | \(\hat y\) |   BCE |
| :-----: | ----: | ---------: | ----: |
|    1    |     1 |       0.90 | 0.105 |
|    2    |     0 |       0.10 | 0.105 |
|    3    |     1 |       0.80 | 0.223 |
|    4    |     0 |       0.30 | 0.357 |
|    5    |     1 |       0.60 | 0.511 |

Average:

$$
\frac{0.105+0.105+0.223+0.357+0.511}{5}
\approx\boxed{0.260}
$$

Example 5 contributes the most because its prediction is only moderately confident despite being correct.

---

## Part 4 — Training versus validation

Validation BCE reaches its minimum:

$$
\boxed{0.31\text{ at epoch 6}}
$$

After epoch 6:

* Training BCE continues decreasing.
* Validation BCE increases.
* The model continues fitting the training data but becomes worse on unseen data.

This is evidence of **overtraining/overfitting**.

A reasonable stopping point is approximately:

$$
\boxed{\text{Epoch 6}}
$$

---

## Part 5 — What is overtraining?

Expected answers:

* Training loss can decrease while validation loss increases because the network begins fitting training-specific details.
* The network may learn noise or accidental patterns rather than general features.
* Very low training BCE does not guarantee good generalization.
* Training should be stopped around the point of minimum validation loss.
* Continuing training would likely increase the generalization gap and worsen performance on unseen data.

---

## Part 6 — Generalization gap

$$
\text{Gap}=L_{\text{validation}}-L_{\text{training}}
$$

| Epoch | Training BCE | Validation BCE |      Gap |
| ----: | -----------: | -------------: | -------: |
|     3 |         0.43 |           0.47 | **0.04** |
|     5 |         0.27 |           0.34 | **0.07** |
|     7 |         0.14 |           0.32 | **0.18** |
|    10 |         0.02 |           0.50 | **0.48** |

The gap grows substantially as overtraining occurs.

A large positive gap indicates that the model performs much better on training data than validation data, which is evidence of poor generalization.

---

## Part 7 — Recognizing overtraining from graphs

* Training curve: continually decreasing curve.
* Validation curve: initially decreasing, then increasing.
* Overtraining begins around the point where validation BCE reaches its minimum and starts rising.
* Stop training based on validation performance rather than training loss alone.

---

## Part 8 — Dropout

Dropout randomly disables neurons during training.

Purpose:

* Prevent excessive dependence on particular neurons.
* Encourage more distributed/robust representations.
* Reduce overfitting.

Too little dropout may have little regularization effect.

Too much dropout can make learning difficult and cause underfitting.

---

## Part 9 — Dropout calculation

20 neurons with:

$$
p=0.3
$$

Expected neurons dropped:

$$
20(0.3)=\boxed{6}
$$

Expected active neurons:

$$
20-6=\boxed{14}
$$

A new random set of neurons is normally dropped at the next training step.

With \(p=0.7\):

$$
20(0.7)=\boxed{14}
$$

would be dropped on average, leaving approximately **6 active**.

---

## Part 10 — Comparing networks

At epoch 8:

### Network A

$$
L_{\text{train}}=0.02,\qquad
L_{\text{validation}}=0.48
$$

$$
\text{Gap}=0.48-0.02=\boxed{0.46}
$$

### Network B

$$
L_{\text{train}}=0.25,\qquad
L_{\text{validation}}=0.35
$$

$$
\text{Gap}=0.35-0.25=\boxed{0.10}
$$

|                    | Network A | Network B |
| :----------------- | --------: | --------: |
| Training BCE       |  **0.02** |      0.25 |
| Validation BCE     |      0.48 |  **0.35** |
| Generalization gap |      0.46 |  **0.10** |

Network A has the lower training loss but substantially worse validation performance and a much larger gap.

Network B's higher training loss is consistent with the regularizing effect of dropout.

---

## Part 11 — Early stopping

* Early stopping terminates training when validation performance stops improving.
* In Part 4, stop around **epoch 6**.
* Continuing beyond epoch 6 increases validation BCE.
* Dropout changes how the network learns; early stopping changes **when training stops**.
* Both techniques can be used together.

---

## Part 12 — Interpreting different models

### Model A

$$
0.62-0.60=\boxed{0.02}
$$

Both losses are relatively high → **underfitting**.

### Model B

$$
0.23-0.20=\boxed{0.03}
$$

Low and similar losses → **good generalization** in this simplified example.

### Model C

$$
0.45-0.01=\boxed{0.44}
$$

Very large gap → **strong evidence of overfitting**.

Model C's extremely low training BCE is misleading because its validation loss is much higher.

---

## Part 13 — Practical regularization problem

| Model | Dropout | Training BCE | Validation BCE | Generalization gap |
| :---- | ------: | -----------: | -------------: | -----------------: |
| A     |     0.0 |         0.03 |           0.45 |           **0.42** |
| B     |     0.2 |         0.12 |           0.28 |           **0.16** |
| C     |     0.5 |         0.25 |           0.30 |           **0.05** |
| D     |     0.8 |         0.50 |           0.49 |          **−0.01** |

Interpretation:

* **A:** strong evidence of overfitting.
* **B/C:** regularization reduces the gap.
* **D:** very high dropout makes the training problem substantially harder and may indicate underfitting.
* Regularization involves balancing fitting the training data with maintaining good generalization.

---

## Part 14 — Final challenge

### Early training

Both training and validation BCE decrease:

$$
L_{\text{train}}\downarrow,
\qquad
L_{\text{validation}}\downarrow
$$

The network is learning useful patterns.

### Later training

$$
L_{\text{train}}\downarrow,
\qquad
L_{\text{validation}}\uparrow
$$

The network is increasingly fitting training-specific information.

### Generalization gap

The gap generally increases:

$$
L_{\text{validation}}-L_{\text{training}}\uparrow
$$

### Remedies

**Early stopping:** stop around the minimum validation loss.

**Dropout:** randomly disable neurons during training to reduce reliance on specific internal representations.

Dropout can increase training BCE while improving validation performance. This is acceptable because the objective is **generalization**, not the lowest possible training loss.

---

# Summary answers

| Statement                               | Answer                                 |
| :-------------------------------------- | :------------------------------------- |
| BCE compares true labels with predicted | **probabilities**                      |
| During overtraining, training loss      | **decreases**                          |
| During overtraining, validation loss    | **increases**                          |
| Generalization gap                      | **Validation loss − Training loss**    |
| Large positive gap can indicate         | **overfitting**                        |
| Dropout randomly                        | **disables/drops neurons**             |
| Dropout is a form of                    | **regularization**                     |
| Early stopping prevents                 | **continued overtraining**             |
| Final objective                         | **Good generalization to unseen data** |

## Central teaching point

The key pattern students should recognize is:

$$
\boxed{
\underbrace{L_{\text{training}}\downarrow}_{\text{model keeps fitting training data}}
\qquad
\underbrace{L_{\text{validation}}\uparrow}_{\text{generalization gets worse}}
}
$$

This divergence is the central signal of **overtraining/overfitting** in this exercise.

Dropout and early stopping are two different tools that can help address it.

