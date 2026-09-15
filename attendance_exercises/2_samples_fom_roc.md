# Exercise: Receiver Operating Characteristic (ROC) and AUC

A Receiver Operating Characteristic (ROC) curve visualizes the performance of a binary classification model across all possible decision thresholds. It plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity).

---

## 1. The Dataset

Consider a small dataset of 6 patients. A machine learning model outputs a predicted risk score (a probability from 0.0 to 1.0) indicating how likely a patient is to have a disease. The actual true status is also known (1 = Diseased, 0 = Healthy).

| Patient | Predicted Score ($y$) | Actual Status ($Y$) |
| :---: | :---: | :---: |
| **A** | 0.9 | 1 |
| **B** | 0.8 | 0 |
| **C** | 0.6 | 1 |
| **D** | 0.5 | 0 |
| **E** | 0.4 | 1 |
| **F** | 0.1 | 0 |

---

## 2. Mathematical Foundations

To calculate the coordinates for the ROC curve, use the following formulas:

*   **True Positive Rate (TPR) / Sensitivity:**
    $$TPR = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{All Actual Positives}}$$

*   **False Positive Rate (FPR) / 1 - Specificity:**
    $$FPR = \frac{FP}{FP + TN} = \frac{\text{False Positives}}{\text{All Actual Negatives}}$$

*   **Total Actual Positives ($P$):** 3 (Patients A, C, E)
*   **Total Actual Negatives ($N$):** 3 (Patients B, D, F)

---

## 3. Step-by-Step Example (Threshold = 0.5)

To classify patients, we use a rule: if $\text{Score} \ge \text{Threshold}$, predict Positive (1); otherwise, predict Negative (0).

Let's test a **Threshold of 0.5**:
*   **Predicted Positive ($\ge 0.5$):** A (0.9), B (0.8), C (0.6), D (0.5)
*   **Predicted Negative ($< 0.5$):** E (0.4), F (0.1)

Evaluating these predictions against actual outcomes gives:
*   **TP (True Positive):** A, C $\rightarrow$ **2**
*   **FP (False Positive):** B, D $\rightarrow$ **2**
*   **FN (False Negatives):** E $\rightarrow$ **1**
*   **TN (True Negatives):** F $\rightarrow$ **1**

Now calculate the rates:
*   $TPR = \frac{2}{3} \approx 0.67$
*   $FPR = \frac{2}{3} \approx 0.67$

---

## 4. Complete Threshold Table

By repeating this evaluation for every unique score boundary, we get the coordinates $(FPR, TPR)$ required to plot the curve:

| Threshold | TP | FP | FN | TN | TPR ($y$-axis) | FPR ($x$-axis) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **> 1.0** (No one positive) | 0 | 0 | 3 | 3 | 0.00 | 0.00 |
| **0.9** | 1 | 0 | 2 | 3 | 0.33 | 0.00 |
| **0.8** | 1 | 1 | 2 | 2 | 0.33 | 0.33 |
| **0.6** | 2 | 1 | 1 | 2 | 0.67 | 0.33 |
| **0.5** | 2 | 2 | 1 | 1 | 0.67 | 0.67 |
| **0.4** | 3 | 2 | 0 | 1 | 1.00 | 0.67 |
| **0.1** | 3 | 3 | 0 | 0 | 1.00 | 1.00 |

---

## 5. Visualization & AUC Challenge

1.  **Plotting:** Set up a graph with the FPR on the $x$-axis ($0$ to $1$) and the TPR on the $y$-axis ($0$ to $1$). Plot the coordinates from the table and connect them sequentially starting at $(0,0)$ up to $(1,1)$.
2.  **Calculate AUC:** Find the Area Under the Curve (AUC) by calculating the geometric area of the rectangles or trapezoids created beneath your plotted line.


---

# Short conceptual questions

## Question 1


