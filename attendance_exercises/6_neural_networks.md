Here’s a self-contained exercise designed to have students compute **Adam (Adaptive Moment Estimation)** by hand, including the first\- and second-moment estimates, bias correction, and parameter update.

 Exercise: Adaptive Moment Estimation (Adam) Step by Step

# Exercise: Adaptive Moment Estimation (Adam)

 In this exercise, you will apply the **Adaptive Moment Estimation (Adam)** optimization algorithm to a simple concrete problem. You will calculate each step of Adam by hand and observe how the parameter changes over several iterations.

 ## Learning objectives

 By the end of this exercise, you should be able to:

 - Explain the role of the first and second moments in Adam.
- Compute the moving averages of gradients and squared gradients.
- Apply bias correction.
- Compute an Adam parameter update.
- Explain why Adam's update is different from ordinary gradient descent.

---

 ## 1\. The problem

 Suppose we want to minimize the following function:

 $$
f(x) = (x-3)^2
$$

 The minimum occurs at

 $$
x^* = 3.
$$

 The derivative is

 $$
\frac{df}{dx} = 2(x-3).
$$

 We will use **Adam** to find the minimum.

 Assume the following Adam hyperparameters:

 $$
\alpha = 0.1
$$

 $$
\beta_1 = 0.9
$$

 $$
\beta_2 = 0.999
$$

 $$
\epsilon = 10^{-8}.
$$

 We start with

 $$
x_0 = 0.
$$

 Initially, Adam's first- and second-moment estimates are both zero:

 $$
m_0 = 0,
\qquad
v_0 = 0.
$$

 Recall the Adam update rules:

 ### Step 1: Compute the gradient

 $$
g_t = \nabla f(x_{t-1})
$$

 ### Step 2: Update the first moment

 $$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

 ### Step 3: Update the second moment

 $$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

 ### Step 4: Correct the bias

 $$
\hat m_t = \frac{m_t}{1-\beta_1^t}
$$

 $$
\hat v_t = \frac{v_t}{1-\beta_2^t}
$$

 ### Step 5: Update the parameter

 $$
x_t =
x_{t-1}
-
\alpha
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}.
$$

---

 # 2\. Iteration 1 — Guided calculation

 We begin with

 $$
x_0=0.
$$

 ### Question 1: Compute the gradient

 Calculate

 $$
g_1 = 2(x_0-3).
$$

 **Your answer:**

 $$
g_1 = \boxed{\phantom{000}}
$$

---

 ### Question 2: Compute the first moment

 Using

 $$
m_1 = 0.9m_0 + 0.1g_1,
$$

 calculate $m_1$.

 **Your answer:**

 $$
m_1 = \boxed{\phantom{000}}
$$

---

 ### Question 3: Compute the second moment

 Using

 $$
v_1 = 0.999v_0 + 0.001g_1^2,
$$

 calculate $v_1$.

 **Your answer:**

 $$
v_1 = \boxed{\phantom{000}}
$$

---

 ### Question 4: Apply bias correction

 Because this is the first iteration,

 $$
\hat m_1 =
\frac{m_1}{1-0.9^1}.
$$

 Calculate $\hat m_1$.

 Then calculate

 $$
\hat v_1 =
\frac{v_1}{1-0.999^1}.
$$

 **Your answers:**

 $$
\hat m_1 = \boxed{\phantom{000}}
$$

 $$
\hat v_1 = \boxed{\phantom{000}}
$$

---

 ### Question 5: Update $x$

 Finally, calculate

 $$
x_1 =
x_0 -
0.1
\frac{\hat m_1}
{\sqrt{\hat v_1}+10^{-8}}.
$$

 **Your answer:**

 $$
x_1 \approx \boxed{\phantom{000}}
$$

---

 # 3\. Iteration 2 — Work it out yourself

 You should now have a value for $x_1$.

 Repeat the same five steps.

 ### Step 1: Gradient

 $$
g_2 = 2(x_1-3)
$$

 $$
g_2 = \boxed{\phantom{000}}
$$

 ### Step 2: First moment

 $$
m_2 = 0.9m_1+0.1g_2
$$

 $$
m_2 = \boxed{\phantom{000}}
$$

 ### Step 3: Second moment

 $$
v_2 = 0.999v_1+0.001g_2^2
$$

 $$
v_2 = \boxed{\phantom{000}}
$$

 ### Step 4: Bias correction

 $$
\hat m_2 =
\frac{m_2}{1-0.9^2}
$$

 $$
\hat v_2 =
\frac{v_2}{1-0.999^2}
$$

 Calculate:

 $$
\hat m_2 = \boxed{\phantom{000}}
$$

 $$
\hat v_2 = \boxed{\phantom{000}}
$$

 ### Step 5: Parameter update

 $$
x_2 =
x_1 -
0.1
\frac{\hat m_2}
{\sqrt{\hat v_2}+10^{-8}}
$$

 Therefore,

 $$
x_2 = \boxed{\phantom{000}}
$$

---

 # 4\. Iteration 3

 Now perform the calculation for a third iteration without the intermediate hints.

 Complete the following table.

 | Quantity | Iteration 3 |
| --- | --- |
| $x_2$ |  |
| $g_3$ |  |
| $m_3$ |  |
| $v_3$ |  |
| $\hat m_3$ |  |
| $\hat v_3$ |  |
| $x_3$ |  |

Use:

 $$
g_3 = 2(x_2-3)
$$

 $$
m_3 = 0.9m_2+0.1g_3
$$

 $$
v_3 = 0.999v_2+0.001g_3^2
$$

 $$
\hat m_3=\frac{m_3}{1-0.9^3}
$$

 $$
\hat v_3=\frac{v_3}{1-0.999^3}
$$

 and

 $$
x_3 =
x_2 -
0.1
\frac{\hat m_3}
{\sqrt{\hat v_3}+10^{-8}}.
$$

---

 # 5\. Understanding what Adam is doing

 Answer the following conceptual questions.

 ### Question 6

 What does $m_t$ represent?

 Choose the best description:

 - [ ] The average of the parameters seen so far.
- [ ] An exponentially weighted moving average of the gradients.
- [ ] An exponentially weighted moving average of the squared parameters.
- [ ] The learning rate.

 Explain your answer in one or two sentences.

---

 ### Question 7

 What does $v_t$ represent?

 - [ ] An exponentially weighted moving average of the gradients.
- [ ] An exponentially weighted moving average of the squared gradients.
- [ ] The current value of the objective function.
- [ ] The accumulated parameter updates.

---

 ### Question 8

 Why does Adam use $g_t^2$ when calculating $v_t$?

 Consider what happens if a gradient is large in magnitude. How would this affect $v_t$ and, consequently, the size of the parameter update?

---

 ### Question 9

 Why are the bias-correction terms

 $$
1-\beta_1^t
$$

 and

 $$
1-\beta_2^t
$$

 necessary?

 **Hint:** Think about the fact that

 $$
m_0=v_0=0.
$$

---


