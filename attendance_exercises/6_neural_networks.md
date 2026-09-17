# Exercise: Regularization of Neural Networks

## Learning objectives

By the end of this exercise, you should be able to:

* Explain what **overtraining (overfitting)** is.
* Distinguish between **training loss** and **validation loss**.
* Calculate **binary cross-entropy (BCE)**.
* Use BCE to monitor whether a neural network is overtraining.
* Calculate and interpret the **generalization gap**.
* Explain how **dropout layers** can reduce overfitting.
* Understand why a model with very low training loss is not necessarily a good model.

---

# Scenario

A neural network is being trained to determine whether a medical image contains a particular condition.

The network performs **binary classification**:

$$
y =
\begin{cases}
1 & \text{condition present}\\
0 & \text{condition absent}
\end{cases}
$$

For each image, the neural network produces a probability:

$$
0\leq\hat y\leq1.
$$

For example,

$$
\hat y=0.90
$$

means that the network predicts a 90% probability that the condition is present.

We will use a simplified dataset to investigate what happens as the neural network is trained for more and more epochs.

---

# Part 1 — Predictions and classification

Suppose the network makes the following predictions for 10 images.

| Image | True label $y$ | Predicted probability $\hat y$ |
| :---: | :--------------: | -------------------------------: |
|   1   |         1        |                             0.90 |
|   2   |         0        |                             0.10 |
|   3   |         1        |                             0.80 |
|   4   |         0        |                             0.30 |
|   5   |         1        |                             0.60 |
|   6   |         0        |                             0.20 |
|   7   |         0        |                             0.40 |
|   8   |         1        |                             0.75 |
|   9   |         0        |                             0.15 |
|   10  |         1        |                             0.45 |

Use the decision rule:

$$
\hat y\geq0.5
\Rightarrow \text{class 1}
$$

$$
\hat y<0.5
\Rightarrow \text{class 0}.
$$

### Questions

1. Which images are classified as class 1?
2. Which images are classified as class 0?
3. Which images are classified incorrectly?
4. Calculate the classification accuracy.
5. Is accuracy enough to determine whether the network is overtraining? Explain.

---

# Part 2 — Binary cross-entropy

Accuracy does not take the **confidence** of the network into account.

Consider two predictions for an example whose true class is 1:

$$
\hat y=0.51
$$

and

$$
\hat y=0.99.
$$

Both predictions result in class 1, but the second prediction is much more confident.

Binary cross-entropy measures the quality of the predicted probabilities.

For one example:

$$
\boxed{
L_{\mathrm{BCE}}
=
-\left[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
\right]
}
$$

For $y=1\):

$$
L_{\mathrm{BCE}}=-\log(\hat y)
$$

For $y=0\):

$$
L_{\mathrm{BCE}}=-\log(1-\hat y).
$$

### Questions

Calculate the BCE for each case.

### Case A

$$
y=1,\qquad \hat y=0.9
$$

### Case B

$$
y=1,\qquad \hat y=0.51
$$

### Case C

$$
y=0,\qquad \hat y=0.1
$$

### Case D

$$
y=0,\qquad \hat y=0.9
$$

Use natural logarithms.

Then answer:

1. Which prediction has the lowest BCE?
2. Which prediction has the highest BCE?
3. Why is a confidently wrong prediction penalized strongly?
4. Why can BCE provide information that accuracy does not?

---

# Part 3 — Average binary cross-entropy

The BCE for a dataset is calculated as the mean of the individual losses:

$$
\boxed{
L_{\mathrm{BCE}}
=
\frac{1}{N}
\sum_{i=1}^{N}L_i
}
$$

Consider the following predictions:

| Example | $y\) | $\hat y$ |
| :-----: | :---: | ---------: |
|    1    |   1   |       0.90 |
|    2    |   0   |       0.10 |
|    3    |   1   |       0.80 |
|    4    |   0   |       0.30 |
|    5    |   1   |       0.60 |

### Questions

1. Calculate the BCE for each example.
2. Calculate the average BCE.
3. Which example contributes the most to the total loss?
4. Why does BCE penalize the prediction in example 5 more than the prediction in example 1, even though both are classified correctly?

---

# Part 4 — Training versus validation data

When training a neural network, the available data is normally divided into different sets.

### Training set

The network uses these examples to update its weights.

### Validation set

These examples are not used to update the weights. They are used to monitor how well the model generalizes to unseen data during development.

Suppose we train a neural network for 10 epochs and obtain:

| Epoch | Training BCE | Validation BCE |
| ----: | -----------: | -------------: |
|     1 |         0.69 |           0.70 |
|     2 |         0.55 |           0.58 |
|     3 |         0.43 |           0.47 |
|     4 |         0.34 |           0.39 |
|     5 |         0.27 |           0.34 |
|     6 |         0.20 |           0.31 |
|     7 |         0.14 |           0.32 |
|     8 |         0.09 |           0.36 |
|     9 |         0.05 |           0.42 |
|    10 |         0.02 |           0.50 |

### Questions

1. What happens to the training BCE as training progresses?
2. What happens to the validation BCE?
3. At which epoch is the validation BCE lowest?
4. After which epoch does the validation BCE start increasing?
5. What happens to the training BCE after this point?
6. Is the network still learning the training data?
7. Is it improving on unseen data?
8. What does this tell you about overtraining?

---

# Part 5 — What is overtraining?

A neural network is **overtraining/overfitting** when it starts to learn details that are specific to the training data rather than patterns that generalize well to new data.

A typical pattern is:

$$
\boxed{
L_{\text{training}}\downarrow
}
$$

while

$$
\boxed{
L_{\text{validation}}\uparrow
}
$$

In other words, the network continues getting better on the examples it has seen while getting worse on examples it has not seen.

### Questions

1. Why can training BCE continue to decrease while validation BCE increases?
2. What might the network be learning during this stage?
3. Why is a very small training BCE not necessarily desirable?
4. Which epoch in Part 4 would be a reasonable point to stop training?
5. What would happen if training continued to epoch 20 or 50?

---

# Part 6 — Generalization gap

The **generalization gap** measures the difference between the training loss and validation loss.

For BCE, define:

$$
\boxed{
\text{Generalization gap}
=
L_{\text{validation}}
-
L_{\text{training}}
}
$$

Consider:

| Epoch | Training BCE | Validation BCE |
| ----: | -----------: | -------------: |
|     3 |         0.43 |           0.47 |
|     5 |         0.27 |           0.34 |
|     7 |         0.14 |           0.32 |
|    10 |         0.02 |           0.50 |

### Questions

Calculate the generalization gap at each epoch.

Complete:

| Epoch | Training BCE | Validation BCE | Generalization gap |
| ----: | -----------: | -------------: | -----------------: |
|     3 |         0.43 |           0.47 |                    |
|     5 |         0.27 |           0.34 |                    |
|     7 |         0.14 |           0.32 |                    |
|    10 |         0.02 |           0.50 |                    |

Then answer:

1. How does the gap change as training progresses?
2. At which epoch is the gap largest?
3. What does a large positive gap suggest?
4. Why does the gap increase during overtraining?
5. Can a model have a low training BCE and a large generalization gap?

---

# Part 7 — Recognizing overtraining from graphs

Imagine that the following graph shows the training and validation BCE.

```text
BCE
 ^
 |\
 | \
 |  \
 |   \ Training
 |    \________________
 |
 |\
 | \
 |  \____
 |       \__
 |          \___
 |              \__
 |                 /
 |                /
 |               /
 |              /
 |             / Validation
 +----------------------------> Epoch
```

### Questions

1. Which curve represents training BCE?
2. Which curve represents validation BCE?
3. What happens to training BCE throughout training?
4. What happens to validation BCE initially?
5. What happens to validation BCE later?
6. At what point does overtraining appear to begin?
7. How could you use this graph to decide when to stop training?

---

# Part 8 — Introducing dropout

One way of reducing overfitting is **dropout**.

A dropout layer randomly disables a fraction of the neurons in a neural network during training.

For example, suppose a layer has six neurons:

```text
Before dropout:

x  x  x  x  x  x

After dropout:

x  o  x  o  x  x
```

The neurons represented by `o` are temporarily disabled during that training step.

Suppose the dropout rate is:

$$
p=0.5.
$$

Approximately 50% of the neurons are randomly disabled during each training step.

> During inference/testing, dropout is not applied in the same way; the full network is used with the appropriate scaling handled by the neural-network framework.

### Questions

1. What does a dropout layer do?
2. Why does randomly removing neurons make the network less dependent on particular neurons?
3. How could this reduce overfitting?
4. Why are different neurons randomly dropped during different training steps?
5. What might happen if the dropout rate is too low?
6. What might happen if the dropout rate is too high?

---

# Part 9 — Dropout calculation

Suppose a hidden layer contains 20 neurons.

The dropout rate is:

$$
p=0.3.
$$

### Questions

1. How many neurons are expected to be dropped during a training step?
2. Approximately how many remain active?
3. Will exactly the same neurons be dropped in the next training step?
4. What would happen if the dropout rate were increased to 0.7?
5. Why might an extremely high dropout rate cause underfitting?

---

# Part 10 — Comparing networks

Two neural networks are trained on the same dataset.

### Network A — No dropout

| Epoch | Training BCE | Validation BCE |
| ----: | -----------: | -------------: |
|     1 |         0.68 |           0.69 |
|     2 |         0.50 |           0.53 |
|     3 |         0.35 |           0.40 |
|     4 |         0.23 |           0.34 |
|     5 |         0.14 |           0.32 |
|     6 |         0.08 |           0.35 |
|     7 |         0.04 |           0.41 |
|     8 |         0.02 |           0.48 |

### Network B — With dropout

| Epoch | Training BCE | Validation BCE |
| ----: | -----------: | -------------: |
|     1 |         0.70 |           0.71 |
|     2 |         0.59 |           0.61 |
|     3 |         0.49 |           0.51 |
|     4 |         0.41 |           0.44 |
|     5 |         0.35 |           0.39 |
|     6 |         0.30 |           0.36 |
|     7 |         0.27 |           0.35 |
|     8 |         0.25 |           0.35 |

### Questions

1. Which network has the lower training BCE at epoch 8?
2. Which network has the lower validation BCE at epoch 8?
3. Calculate the generalization gap for both networks at epoch 8.
4. Which network shows stronger evidence of overfitting?
5. Why might the dropout network have a higher training BCE?
6. Why can a higher training BCE nevertheless be associated with better validation performance?
7. What does this example demonstrate about the purpose of regularization?

---

# Part 11 — Early stopping

Another method for controlling overtraining is **early stopping**.

Suppose validation BCE reaches its minimum at epoch 6:

$$
L_{\text{validation}}=0.31.
$$

After this point, validation BCE begins to increase.

### Questions

1. What is the basic idea behind early stopping?
2. At which epoch would you stop training in the example from Part 4?
3. Why would continuing to epoch 10 be undesirable?
4. How is early stopping different from dropout?
5. Could dropout and early stopping be used together?

---

# Part 12 — Interpreting different models

Consider three models:

### Model A

$$
L_{\text{training}}=0.60
$$

$$
L_{\text{validation}}=0.62
$$

### Model B

$$
L_{\text{training}}=0.20
$$

$$
L_{\text{validation}}=0.23
$$

### Model C

$$
L_{\text{training}}=0.01
$$

$$
L_{\text{validation}}=0.45
$$

### Questions

For each model:

1. Calculate the generalization gap.
2. Which model appears to be underfitting?
3. Which model appears to generalize well?
4. Which model shows strong evidence of overfitting?
5. Why is Model C's very low training BCE misleading?

---

# Part 13 — A practical regularization problem

You train a neural network and obtain the following results:

| Model | Dropout rate | Training BCE | Validation BCE |
| :---- | -----------: | -----------: | -------------: |
| A     |          0.0 |         0.03 |           0.45 |
| B     |          0.2 |         0.12 |           0.28 |
| C     |          0.5 |         0.25 |           0.30 |
| D     |          0.8 |         0.50 |           0.49 |

### Questions

1. Calculate the generalization gap for each model.
2. What does Model A's result suggest?
3. What does Model D's result suggest?
4. Why can too much dropout be harmful?
5. Explain why regularization involves finding a balance between fitting the training data and generalizing to new data.

---

# Part 14 — Final challenge

Imagine that you are training a neural network.

Initially:

$$
L_{\text{training}}\approx
L_{\text{validation}}.
$$

After several epochs:

$$
L_{\text{training}}\downarrow
$$

and

$$
L_{\text{validation}}\downarrow.
$$

Eventually:

$$
L_{\text{training}}\downarrow
$$

but

$$
L_{\text{validation}}\uparrow.
$$

### Questions

1. What is happening during the first phase of training?
2. What happens when the validation loss begins to increase?
3. Why is this considered evidence of overtraining?
4. How would the generalization gap change?
5. How could early stopping help?
6. How could dropout help?
7. Why might dropout cause training BCE to increase?
8. Why can this still be a desirable result?
9. What should ultimately matter when choosing a model: the lowest training BCE or good performance on unseen data? Explain.

---

# Summary

Complete the following statements.

1. **Binary cross-entropy** measures the difference between the true binary labels and the predicted __________.

2. A neural network is likely to be **overtraining** when training loss continues to __________ while validation loss begins to __________.

3. The generalization gap is:

$$
\boxed{
\text{Validation loss}-
\text{Training loss}
}
$$

4. A large positive generalization gap can be evidence of __________.

5. **Dropout** randomly __________ neurons during training.

6. Dropout is a form of __________ used to reduce overfitting.

7. **Early stopping** can prevent a model from training beyond the point where validation performance begins to __________.

8. The ultimate goal of training is not to memorize the training set, but to __________ well to unseen data.

---

# Key idea

A well-trained neural network is not necessarily the network with the smallest training BCE.

The central goal is:

$$
\boxed{
\text{Good performance on unseen data}
}
$$

Regularization techniques such as **dropout** and **early stopping** help prevent the neural network from learning the training data too specifically.

The characteristic warning sign of overtraining is:

$$
\boxed{
\text{Training BCE decreases}
\quad\text{while}\quad
\text{Validation BCE increases}
}
$$

This is accompanied by an increasing **generalization gap**.

