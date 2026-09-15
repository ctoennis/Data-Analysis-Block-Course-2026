Here is the exercise in clean Markdown:

 Exercise: AdaBoost for Heart Disease Detection

# Exercise: AdaBoost for Heart Disease Detection

 A hospital wants to develop a machine-learning model to predict whether a patient has **heart disease**.

 For simplicity, we will use only two features:

 - $X_1$: Age — classified as **Young** (\< 50) or **Old** (≥ 50)
- $X_2$: Chest pain — **0 = No chest pain**, **1 = Chest pain**

 The target is:

 - $y = +1$: Heart disease
- $y = -1$: No heart disease

 We have the following six patients:

 | Patient | Age | Chest pain | True class $y$ |
| --- | --- | --- | --- |
| A | Young | No | -1 |
| B | Young | Yes | +1 |
| C | Old | No | +1 |
| D | Old | Yes | +1 |
| E | Young | No | -1 |
| F | Old | No | -1 |

We want to use **AdaBoost** with simple decision stumps as weak learners.

---

 ## Part 1 — Initial weights

 AdaBoost initially assigns every patient the same weight:

 $$
w_i = \frac{1}{6}
$$

 1. Write down the initial weight for each patient.
2. Consider the following first decision stump:
    $$
   h_1(x) =
      \begin{cases}
      +1 & \text{if Age = Old}\\
      -1 & \text{if Age = Young}
      \end{cases}
   $$
    Calculate the prediction of $h_1$ for every patient.
3. Which patients are misclassified?
4. Calculate the weighted classification error:
    $$
   \epsilon_1 =
      \sum_{i:h_1(x_i)\neq y_i} w_i
   $$

---

 ## Part 2 — Calculate the weak learner's importance

 AdaBoost calculates the importance of a weak learner using:

 $$
\alpha_1 =
\frac{1}{2}
\ln\left(
\frac{1-\epsilon_1}{\epsilon_1}
\right)
$$

 5. Calculate $\alpha_1$.
6. What does a **large positive $\alpha$** mean in AdaBoost?

---

 ## Part 3 — Update the patient weights

 The weights are updated according to:

 $$
w_i' =
w_i e^{-\alpha_1 y_i h_1(x_i)}
$$

 Notice that:

 - If the patient is **correctly classified**:
   $$
  y_i h_1(x_i) = +1
  $$
- If the patient is **misclassified**:
   $$
  y_i h_1(x_i) = -1
  $$

 Therefore:

 $$
w_i' =
\begin{cases}
w_i e^{-\alpha_1} & \text{correct}\\
w_i e^{+\alpha_1} & \text{incorrect}
\end{cases}
$$

 7. Calculate the new **unnormalized** weight for every patient.
8. Normalize the weights so that:
    $$
   \sum_i w_i = 1
   $$
9. Which patients now receive the **largest weights**?
10. Why does this make intuitive sense?

---

 ## Part 4 — Second weak learner

 Now consider a second decision stump:

 $$
h_2(x) =
\begin{cases}
+1 & \text{if Chest pain = Yes}\\
-1 & \text{if Chest pain = No}
\end{cases}
$$

 11. Calculate the prediction of $h_2$ for every patient.
12. Using the **updated weights from Part 3**, calculate the weighted error:

 $$
\epsilon_2 =
   \sum_{i:h_2(x_i)\neq y_i} w_i
$$

 13. Calculate its importance:

 $$
\alpha_2 =
   \frac{1}{2}
   \ln\left(
   \frac{1-\epsilon_2}{\epsilon_2}
   \right)
$$

---

 ## Part 5 — The final AdaBoost classifier

 After two rounds, AdaBoost combines the weak learners:

 $$
H(x) =
\operatorname{sign}
\left[
\alpha_1h_1(x) +
\alpha_2h_2(x)
\right]
$$

 Consider a new patient:

 > **Patient G:** Old, with chest pain.

 14. What does $h_1$ predict for Patient G?
15. What does $h_2$ predict?
16. Calculate:

 $$
\alpha_1h_1(x_G) +
   \alpha_2h_2(x_G)
$$

 17. What is the final AdaBoost prediction?
18. Would the model classify this patient as having heart disease?

---

 ## Discussion Questions

 19. Why does AdaBoost increase the weights of incorrectly classified patients?
20. Suppose one patient has an unusual combination of characteristics and is repeatedly misclassified. What will happen to that patient's weight over successive AdaBoost iterations?
21. Why might this behavior be problematic if the unusual patient is actually an **outlier** or contains an incorrect medical record?
22. In a real heart-disease application, would you prefer a model that minimizes **overall error**, or might it be more important to minimize **false negatives** (patients with heart disease incorrectly classified as healthy)? Explain.

---

 ## Bonus Challenge

 Instead of using only two weak learners, imagine that AdaBoost trains **100 decision stumps**.

 Explain qualitatively what happens to:

 - The weights of difficult patients
- The influence of easy-to-classify patients
- The complexity of the final classifier
- The risk of overfitting
