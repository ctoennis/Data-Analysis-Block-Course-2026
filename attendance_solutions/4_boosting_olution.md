# Answer Key: AdaBoost and Gradient Boosting for Heart Disease Diagnosis

## Part 1 — Decision stumps

### Stump A — Age

Rule:

$$
\text{Age}>47 \Rightarrow +1
$$

Patients predicted positive: **4, 5, 6, 7, 9**

| Patient | Actual | Prediction | Correct? |
| :-----: | :----: | :--------: | :------: |
|    1    |   -1   |     -1     |     x    |
|    2    |   +1   |     -1     |     o    |
|    3    |   +1   |     -1     |     o    |
|    4    |   +1   |     +1     |     x    |
|    5    |   +1   |     +1     |     x    |
|    6    |   +1   |     +1     |     x    |
|    7    |   -1   |     +1     |     o    |
|    8    |   -1   |     -1     |     x    |
|    9    |   +1   |     +1     |     x    |
|    10   |   -1   |     -1     |     x    |

There are **3 errors**, so

$$
\boxed{\epsilon_A=0.30}
$$

> **Correction:** The original exercise's implied error for Stump A should therefore be 0.30, not 0.10.

### Stump B — Blocked arteries

Rule:

$$
\text{Blocked arteries = Yes} \Rightarrow +1
$$

Patients 3, 5, 6, and 9 are predicted positive.

Patients 2 and 4 are infected but have no blocked arteries, so they are incorrectly classified.

Thus:

$$
\boxed{\epsilon_B=\frac{2}{10}=0.20}
$$

### Stump C — Chest pain

Rule:

$$
\text{Chest pain = Yes} \Rightarrow +1
$$

Patients 2, 4, and 5 are correctly classified as positive.

Patients 3, 6, and 9 have heart disease but no chest pain, so they are incorrectly classified.

Thus:

$$
\boxed{\epsilon_C=\frac{3}{10}=0.30}
$$

### Summary

| Stump | Feature          | Errors | Error |
| :---- | :--------------- | -----: | ----: |
| A     | Age              |      3 |  0.30 |
| B     | Blocked arteries |      2 |  0.20 |
| C     | Chest pain       |      3 |  0.30 |

Therefore, among these three candidate stumps, **Stump B is selected** because it has the smallest error.

---

# Part 2 — Which stump should AdaBoost choose?

Initially,

$$
w_i=0.1
$$

for every patient.

Because all weights are equal, the weighted error is simply the fraction of incorrect predictions.

Therefore:

$$
\boxed{\epsilon_A=0.30}
$$

$$
\boxed{\epsilon_B=0.20}
$$

$$
\boxed{\epsilon_C=0.30}
$$

AdaBoost selects:

$$
\boxed{\text{Stump B: Blocked arteries}}
$$

because it has the lowest weighted error.

---

# Part 3 — AdaBoost gives the stump a weight

For the selected stump,

$$
\epsilon=0.20.
$$

The stump weight is

$$
\alpha=
\frac12
\ln\left(\frac{1-\epsilon}{\epsilon}\right).
$$

Therefore,

$$
\alpha=
\frac12
\ln\left(\frac{0.8}{0.2}\right)
$$

$$
=
\frac12\ln(4)
$$

$$
\boxed{\alpha\approx0.693}
$$

### Interpretation

The value is positive because the stump performs better than random guessing.

A **larger $\alpha$** means that the weak learner has a smaller error and therefore receives more influence in the final ensemble.

If

$$
\epsilon=0.5,
$$

then

$$
\alpha=
\frac12\ln(1)=0.
$$

So a learner that performs no better than random guessing receives **zero weight**.

---

# Part 4 — Updating the patient weights

The first stump is:

> Predict heart disease if blocked arteries = Yes.

It makes errors for:

* Patient 2
* Patient 4

The other eight patients are correctly classified.

We have

$$
\alpha=0.693.
$$

For a correctly classified patient:

$$
w_i'=0.1e^{-0.693}
\approx0.1(0.5)
=\boxed{0.05}.
$$

For an incorrectly classified patient:

$$
w_i'=0.1e^{0.693}
\approx0.1(2)
=\boxed{0.20}.
$$

Thus the unnormalized weights are:

| Patient | Correct? | Unnormalized weight |
| :-----: | :------: | ------------------: |
|    1    |     x    |                0.05 |
|    2    |     o    |                0.20 |
|    3    |     x    |                0.05 |
|    4    |     o    |                0.20 |
|    5    |     x    |                0.05 |
|    6    |     x    |                0.05 |
|    7    |     x    |                0.05 |
|    8    |     x    |                0.05 |
|    9    |     x    |                0.05 |
|    10   |     x    |                0.05 |

The sum is

$$
8(0.05)+2(0.20)
=0.40+0.40
=0.80.
$$

We normalize by dividing every weight by 0.80.

Therefore:

* Correctly classified patients:

$$
\frac{0.05}{0.80}
=\boxed{0.0625}
$$

* Incorrectly classified patients:

$$
\frac{0.20}{0.80}
=\boxed{0.25}
$$

### Final weights

| Patient | New weight |
| :-----: | ---------: |
|    1    |     0.0625 |
|    2    | **0.2500** |
|    3    |     0.0625 |
|    4    | **0.2500** |
|    5    |     0.0625 |
|    6    |     0.0625 |
|    7    |     0.0625 |
|    8    |     0.0625 |
|    9    |     0.0625 |
|    10   |     0.0625 |

The weights sum to 1.

### Interpretation

Patients **2 and 4** now have much larger weights because the first stump failed to identify them.

The next stump therefore has a strong incentive to classify these patients correctly.

This is the central idea behind AdaBoost:

> **Make difficult examples more important to subsequent learners.**

---

# Part 5 — The second stump

## Stump D — Chest pain

Rule:

$$
\text{Chest pain = Yes}\Rightarrow+1
$$

It makes errors for patients:

* Patient 3
* Patient 6
* Patient 9

Each has weight 0.0625.

Therefore:

$$
\epsilon_D
=
0.0625+0.0625+0.0625
$$

$$
\boxed{\epsilon_D=0.1875}
$$

---

## Stump E — Weight

Rule:

$$
\text{Weight}>77.5\text{ kg}\Rightarrow+1
$$

Predictions:

* Patients 3, 4, 5, 6, 9, 10 → positive
* Patients 1, 2, 7, 8 → negative

Errors occur for:

* Patient 2: infected, but predicted negative
* Patient 8: not infected, but predicted positive
* Patient 10: not infected, but predicted positive

Their weights are:

$$
w_2=0.25,\qquad
w_8=0.0625,\qquad
w_{10}=0.0625.
$$

Thus:

$$
\epsilon_E
=
0.25+0.0625+0.0625
$$

$$
\boxed{\epsilon_E=0.375}
$$

### Which stump is selected?

$$
\epsilon_D=0.1875
$$

versus

$$
\epsilon_E=0.375.
$$

Therefore AdaBoost selects:

$$
\boxed{\text{Stump D}}
$$

### Important observation

Stump D makes **three mistakes**, while Stump E also makes three mistakes.

However, AdaBoost does **not** treat all patients equally anymore.

Stump E makes a mistake on patient 2, who has a very large weight of 0.25.

Therefore its weighted error is much larger.

This demonstrates why AdaBoost uses **weighted error rather than simply counting mistakes**.

---

# Part 6 — Combining the stumps

Suppose three stumps make the following predictions:

| Stump   | Prediction | Weight |
| :------ | :--------: | -----: |
| $h_1$ |     +1     |    0.8 |
| $h_2$ |     -1     |    0.4 |
| $h_3$ |     +1     |    0.3 |

The combined score is

$$
F(x)
=
(0.8)(+1)
+
(0.4)(-1)
+
(0.3)(+1).
$$

Therefore,

$$
F(x)=0.8-0.4+0.3
$$

$$
\boxed{F(x)=0.7}
$$

Since the score is positive,

$$
\boxed{\hat y=+1}
$$

so the final model predicts **heart disease**.

Stump $h_1$ has the largest influence because it has the largest $\alpha$.

---

# Part 7 — Understanding AdaBoost

### 1. Why does AdaBoost focus on difficult patients?

Because incorrectly classified patients receive **larger weights**.

This makes their classification errors more important when training the next weak learner.

### 2. What happens if a patient is repeatedly misclassified?

Its weight can become increasingly important to subsequent learners.

The ensemble therefore increasingly focuses on difficult examples.

### 3. Why use decision stumps?

A stump is a very simple and weak classifier.

For example:

> "Does the patient have blocked arteries?"

It may not be accurate enough by itself, but many weak learners can be combined into a much stronger classifier.

### 4. Why can many simple stumps be powerful?

Each stump can capture a different aspect of the data.

AdaBoost combines them using their individual weights, allowing the ensemble to make a more sophisticated overall decision.

---

# Part 8 — From AdaBoost to Gradient Boosting

The simplified example uses

$$
F_0(x)=0.5.
$$

The residual is

$$
r_i=y_i-F_0(x_i).
$$

### Residuals

| Patient | Actual $y$ | Prediction $F_0$ | Residual $r=y-F_0$ |
| :-----: | -----------: | -----------------: | -------------------: |
|    1    |          0.0 |                0.5 |                 -0.5 |
|    2    |          1.0 |                0.5 |                 +0.5 |
|    3    |          1.0 |                0.5 |                 +0.5 |
|    4    |          0.0 |                0.5 |                 -0.5 |
|    5    |          1.0 |                0.5 |                 +0.5 |

Therefore:

$$
\boxed{
r=(-0.5,+0.5,+0.5,-0.5,+0.5)
}
$$

Patients 2, 3, and 5 have positive residuals, meaning the model is **underestimating** their target.

Patients 1 and 4 have negative residuals, meaning the model is **overestimating** their target.

The next tree therefore tries to learn these residuals — in other words, it tries to learn how the current model should be **corrected**.

---

# Part 9 — Gradient boosting with a decision stump

The second stump predicts:

$$
\text{Age}>50
\Rightarrow r=+0.3
$$

otherwise:

$$
r=-0.2.
$$

The new model is

$$
F_1(x)=F_0(x)+\eta h_1(x).
$$

Given

$$
F_0=0.5
$$

and

$$
\eta=1,
$$

### Patient aged 60

Since

$$
60>50,
$$

the stump predicts

$$
+0.3.
$$

Therefore:

$$
F_1=0.5+0.3
$$

$$
\boxed{F_1=0.8}
$$

### Patient aged 40

Since

$$
40<50,
$$

the stump predicts

$$
-0.2.
$$

Therefore:

$$
F_1=0.5-0.2
$$

$$
\boxed{F_1=0.3}
$$

The second tree is a **correction** because it does not replace the first model. It adds information that attempts to reduce the existing prediction errors.

With additional trees:

$$
F_2(x)=F_1(x)+\eta h_2(x)
$$

$$
F_3(x)=F_2(x)+\eta h_3(x)
$$

and so on.

The model gradually becomes more sophisticated.

---

# Part 10 — AdaBoost vs. Gradient Boosting

|                                 | AdaBoost                                           | Gradient Boosting                                                            |
| :------------------------------ | :------------------------------------------------- | :--------------------------------------------------------------------------- |
| Basic building block            | Weak learners such as stumps                       | Usually shallow decision trees                                               |
| What happens after an error?    | Misclassified examples receive larger weights      | The next learner focuses on the current model's residuals/negative gradients |
| How is the next learner guided? | By the updated example weights                     | By the gradient of the loss function                                         |
| How are learners combined?      | Weighted sum/vote using $\alpha$                 | Sequentially added to the existing model                                     |
| Main idea                       | Focus more strongly on difficult training examples | Add models that correct the current model's errors                           |

---

# Final questions

### 1. What is the difference between a stump and an ensemble of stumps?

A **stump** is a single, very simple decision rule based on one split.

An **ensemble of stumps** combines many such rules to produce a more powerful classifier.

---

### 2. Why does AdaBoost change training-example weights?

To make incorrectly classified examples more important to subsequent learners.

The next stump therefore concentrates more on the cases that previous stumps struggled with.

---

### 3. How does $\alpha$ determine a stump's influence?

$$
\alpha=
\frac12\ln\left(\frac{1-\epsilon}{\epsilon}\right)
$$

A stump with a smaller error $\epsilon$ receives a larger $\alpha$, giving it greater influence in the final prediction.

---

### 4. How does gradient boosting use errors?

For a given loss function, gradient boosting determines the direction in which the current predictions should be changed to reduce the loss.

A new tree is trained to approximate this **correction**, and the correction is added to the existing model.

---

### 5. What is the main similarity between AdaBoost and gradient boosting?

Both methods build a strong model **sequentially from multiple weak learners**.

Each new learner attempts to improve the ensemble produced by the previous learners.

---

### 6. What is the main difference?

The key difference is **how the next learner is determined**:

* **AdaBoost:** changes the weights of training examples, emphasizing examples that were previously misclassified.
* **Gradient boosting:** fits the next learner to the **negative gradient of the loss** (often introduced as residuals for squared-error regression).

---

# Key numerical results

For quick reference:

$$
\boxed{\epsilon_A=0.30}
$$

$$
\boxed{\epsilon_B=0.20}
$$

$$
\boxed{\epsilon_C=0.30}
$$

First selected stump:

$$
\boxed{\text{Blocked arteries}}
$$

First stump weight:

$$
\boxed{\alpha\approx0.693}
$$

Updated weights:

$$
\boxed{
(0.0625,\;0.25,\;0.0625,\;0.25,\;0.0625,\;0.0625,\;0.0625,\;0.0625,\;0.0625,\;0.0625)
}
$$

Second-stump errors:

$$
\boxed{\epsilon_D=0.1875}
$$

$$
\boxed{\epsilon_E=0.375}
$$

Therefore, the second selected stump is:

$$
\boxed{\text{Chest pain}}
$$

Gradient-boosting residuals:

$$
\boxed{(-0.5,\;+0.5,\;+0.5,\;-0.5,\;+0.5)}
$$

Example gradient-boosting updates:

$$
\boxed{F_1(60)=0.8}
$$

$$
\boxed{F_1(40)=0.3}
$$

