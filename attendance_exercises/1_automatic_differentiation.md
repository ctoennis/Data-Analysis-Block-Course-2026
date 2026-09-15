# Forward and Reverse Automatic Differentiation by Hand

## Learning goals

After this exercise, you should be able to

- explain what a **dual number** is and why dual numbers are useful for forward-mode automatic differentiation,
- propagate values and derivatives through a computation using dual numbers,
- compute directional derivatives and Jacobian columns with forward mode,
- explain the idea of a **computational graph** and an **adjoint** in reverse-mode automatic differentiation,
- propagate adjoints backwards through a computation graph,
- compute gradients and Jacobian rows with reverse mode,
- explain the practical difference between forward and reverse AD.

The calculations in this sheet are designed to be done **by hand**. No programming is required.

---

# Part I — Forward-mode AD with dual numbers

## 1. What are dual numbers?

A dual number $\hat{x}$ has the form

$$
\hat{x} = x + \dot{x}\varepsilon
$$

where $x,\dot{x}\in\mathbb{R}$ and $\varepsilon$ is a formal symbol with the special property

$$
\varepsilon^2 = 0
\qquad
\varepsilon \neq 0
$$

If we then evaluate a differentiable function $f$ using the usual arithmetic rules together (think Taylor series) with $\varepsilon^2=0$, we obtain

$$
f(x+\dot{x}\varepsilon)
= f(x) + f'(x)\dot{x}\,\varepsilon
$$

For the special choice $\dot{x}=1$

$$
f(x+\varepsilon)=f(x)+f'(x)\varepsilon
$$

so the coefficient of $\varepsilon$ is directly $f'(x)$.

---

## 2. Arithmetic with dual numbers

Let

$$
\hat{a}=a+\dot{a}\varepsilon,
\qquad
\hat{b}=b+\dot{b}\varepsilon.
$$

Then

### Addition

$$
\hat{a}+\hat{b}
=(a+b)+(\dot{a}+\dot{b})\varepsilon.
$$

### Multiplication

$$
\hat{a}\hat{b}
=(a+\dot{a}\varepsilon)(b+\dot{b}\varepsilon).
$$

Expanding the product gives

$$
ab+(a\dot{b}+\dot{a}b)\varepsilon
$$

which is exactly the product rule.

### Some useful elementary functions

$$
\exp(a+\dot{a}\varepsilon)
= e^a+\dot{a}\,e^a\varepsilon,
$$

$$
\sin(a+\dot{a}\varepsilon)
= \sin(a)+\dot{a}\,\cos(a)\varepsilon,
$$

$$
\log(a+a'\varepsilon)
= \log(a)+\frac{\dot{a}}{a}\varepsilon,
\qquad a>0,
$$

and

$$
(a+\dot{a}\varepsilon)^2
= a^2+2a\,\dot{a}\varepsilon.
$$

---

## 3. Very simple 1D example

Consider

$$
f(x)=x^2+3x
$$

and compute $f(2)$ and $f'(2)$ using dual numbers.

The dual $\hat{x}_0$ is (with tangent $\dot{x}_0=1$)

$$
\hat{x}_0=2+\varepsilon.
$$

Then

$$
\hat{x}_0^2=(2+\varepsilon)^2
=4+4\varepsilon,
$$

and

$$
3\hat{x}_0=6+3\varepsilon.
$$

Therefore

$$
f(\hat{x}_0)
=10+7\varepsilon.
$$

Hence

$$
f(2)=10
$$
$$
f'(2)=7.
$$

The important point is that the value and the derivative were propagated at the same time.

---

## 4. More than one input

Suppose now that

$$
g:\mathbb{R}^n\to\mathbb{R}^m.
$$

We can seed every input with its own tangent component:

$$
\hat{\mathbf{x}}
=\mathbf{x}+\dot{\mathbf{x}}\varepsilon.
$$

Forward AD then gives

$$
\boxed{
 g(\mathbf{x}+\dot{\mathbf{x}}\varepsilon)
 = g(\mathbf{x}) + J_g(\mathbf{x})\,\dot{\mathbf{x}}\,\varepsilon
}
$$

where $J_g$ is the Jacobian matrix.

Therefore, one forward-mode sweep computes one Jacobian-vector product

$$
J_g(\mathbf{x})\dot{\mathbf{x}}.
$$

If we choose $\dot{\mathbf{x}}$ to be a standard basis vector, for example

$$
\mathbf{e}_1=(1,0,0)^T,
$$

then the tangent part is the first column of the Jacobian.

---

# Forward-mode exercises

## Exercise 1 — Scalar input, scalar output

Consider

$$
f(x)=\log\left(xe^x+3\right).
$$

Evaluate the function and its derivative at

$$
x=1
$$

using dual-number propagation.

---

## Exercise 2 — Vector input, scalar output

Consider

$$
g(x,y,z)=xy+\sin z+y^2
$$

at the point

$$
(x,y,z)=(1,2,0).
$$

a.) First calculate the ordinary function value $g(1,2,0)$.

b.) Use forward-mode AD with the three seeds

$$
\mathbf{e}_x=(1,0,0),
\qquad
\mathbf{e}_y=(0,1,0),
\qquad
\mathbf{e}_z=(0,0,1)
$$

to determine the full gradient

$$
\nabla g(1,2,0).
$$


c.) Now use only one forward sweep with seed direction

$$
\dot{\mathbf{x}}=(1,-1,2)^T
$$

to compute the directional derivative

$$
D_{\dot{\mathbf{x}}}g
=
\nabla g^T\dot{\mathbf{x}}.
$$

---

## Exercise 3 — Vector input, vector output

Consider the vector-valued function

$$
\mathbf{h}(x,y,z)
=
\begin{pmatrix}
 h_1(x,y,z)\\
 h_2(x,y,z)
\end{pmatrix}
=
\begin{pmatrix}
 xy+z\\
 x^2+\sin y-z^2
\end{pmatrix}.
$$

Evaluate it at

$$
(x,y,z)=(1,0,1).
$$

a.) Calculate $\mathbf h(1,0,1)$.

b.) Use the three input basis directions

$$
\mathbf e_x,
\qquad
\mathbf e_y,
\qquad
\mathbf e_z
$$

and dual numbers to compute the full Jacobian

$$
J_{\mathbf h}
=
\begin{pmatrix}
\frac{\partial h_1}{\partial x} &
\frac{\partial h_1}{\partial y} &
\frac{\partial h_1}{\partial z}\\[4pt]
\frac{\partial h_2}{\partial x} &
\frac{\partial h_2}{\partial y} &
\frac{\partial h_2}{\partial z}
\end{pmatrix}.
$$

Remember: each forward sweep gives one column of the Jacobian.

c.) Use a single forward sweep with

$$
\mathbf v=(1,2,-1)^T
$$

to compute

$$
J_{\mathbf h}\dot{\mathbf{x}}
$$

---

# Part II — Reverse-mode AD / Backward AD

## 1. The basic idea

Forward mode propagates derivatives together with the values from input to output.

Reverse mode proceeds differently:

1. Perform an ordinary forward pass and store the intermediate values.
2. Start from the output.
3. Propagate sensitivities backwards through the computational graph.

For an intermediate variable $v$, define its adjoint as

$$
\boxed{
\bar v = \frac{\partial L}{\partial v}
}
$$

where $L$ is the final scalar output whose derivative we want (the bar notation is common in reverse-mode AD).

The output is initialized with

$$
\bar L
=
\frac{\partial L}{\partial L}
=1.
$$

Then the chain rule is applied locally, one operation at a time, in reverse order.

---

## 2. Local backward rules

Suppose an intermediate variable $c$ is computed from earlier variables.

### Addition

If

$$
c=a+b,
$$

then

$$
\bar a \mathrel{+}= \bar c,
\qquad
\bar b \mathrel{+}= \bar c.
$$

with $\mathrel{+}=$ the programming operator meaning $\bar a + \bar c$ replaces the initial $\bar a$-assignment: if one variable influences the output through several paths, all contributions to its adjoint must be added.

### Multiplication

If

$$
c=ab,
$$

then

$$
\bar a \mathrel{+}= \bar c\,b,
\qquad
\bar b \mathrel{+}= \bar c\,a.
$$

### Some useful elementary functions

$$
c = e^a \qquad \Rightarrow \qquad \bar a \mathrel{+}= \bar c\,e^a
$$
$$
c=\sin a \qquad \Rightarrow \qquad \bar a \mathrel{+}= \bar c\cos a.
$$
$$
c=\log a \qquad \Rightarrow \qquad \bar a \mathrel{+}= \bar c\frac{1}{a}.
$$
$$
c=a^2 \qquad \Rightarrow \qquad \bar a \mathrel{+}= \bar c\,2a.
$$

---

## 3. Very simple 1D reverse-mode example

Consider

$$
f(x)=(x+1)^2
$$

at $x=2$.

Break the function into elementary operations:

$$
v_1=x+1,
$$

$$
v_2=v_1^2,
$$

$$
f=v_2.
$$

### Forward pass

At $x=2$,

$$
v_1=3,
\qquad
v_2=9.
$$

### Backward pass

Start with

$$
\bar v_2=1
$$
which is the condition $\frac{\partial f}{\partial v_2} \overset{!}{=} 1$.

Since

$$
v_2=v_1^2
$$

we get

$$
\bar v_1
=
\bar v_2\,2v_1
=1\cdot 6
=6
$$

and since

$$
v_1=x+1
$$

we obtain

$$
\bar x=\bar v_1=6
$$

Hence

$$
f'(2)=6.
$$

In one dimension this may look more complicated than ordinary differentiation. Its advantage becomes clear when a function has many inputs but only one scalar output.

---

## 4. Reverse mode for several inputs

For

$$
f:\mathbb R^n\to\mathbb R,
$$

one reverse sweep gives

$$
\nabla f
$$

with derivatives with respect to all inputs at once.

This is why reverse mode is particularly useful in machine learning: a neural network may have millions of parameters, but the loss is usually a single scalar.

For a vector-valued function

$$
\mathbf f:\mathbb R^n\to\mathbb R^m,
$$

we must first choose an output seed $\mathbf w\in\mathbb R^m$. Reverse mode then computes

$$
\boxed{
J_{\mathbf f}^T\mathbf w
}.
$$

This is called a vector-Jacobian product or, equivalently under row-vector notation, $\mathbf w^TJ$.

To reconstruct the complete Jacobian, use one reverse sweep per output basis vector.

---

# Reverse-mode exercises

Use exactly the same three functions as in the forward-mode section.

## Exercise 1 — Scalar input, scalar output

For

$$
f(x)=\log(xe^x+3)
$$

at $x=1$:

1. Write the function as a sequence of elementary intermediate variables.
2. Perform the forward pass and record all intermediate values.
3. Set the output adjoint equal to $1$.
4. Propagate all adjoints backwards.
5. Determine $\bar x=f'(1)$.

---

## Exercise 2 — Three inputs, scalar output

For

$$
g(x,y,z)=xy+\sin z+y^2
$$

at

$$
(x,y,z)=(1,2,0),
$$

use the intermediate variables

$$
a=xy,
\qquad
b=\sin z,
\qquad
c=y^2,
$$

$$
d=a+b+c
$$

---

## Exercise 3 — Three inputs, two outputs

For

$$
\mathbf h(x,y,z) = \begin{pmatrix} xy+z\\ x^2+\sin y-z^2\end{pmatrix}
$$

at

$$
(x,y,z)=(1,0,1),
$$

reverse mode needs an output seed because the output is not scalar.

a.) First output. Use

$$
\mathbf w_1 = \begin{pmatrix}1\\0\end{pmatrix}.
$$

That is, propagate backwards only from $h_1$.

Compute

$$
J_{\mathbf h}^T\mathbf w_1.
$$

b.) Second output. Use

$$
\mathbf w_2 = \begin{pmatrix}0\\1\end{pmatrix}
$$

Compute

$$
J_{\mathbf h}^T\mathbf w_2
$$

c.) Full Jacobian. Use the two results to reconstruct the complete Jacobian.

d.) General output seed. Without calculating the Jacobian from scratch, use

$$
\mathbf w = \begin{pmatrix}3\\-1\end{pmatrix}
$$

to calculate

$$
J_{\mathbf h}^T\mathbf w
$$

---

# Short conceptual questions

## Question 1

Why does setting $\varepsilon^2=0$ cause the coefficient of $\varepsilon$ to behave like a derivative?

## Question 2

For a function

$$
f:\mathbb R^{100}\to\mathbb R,
$$

how many basis-direction forward sweeps would be needed to obtain the full gradient? How many reverse sweeps?

## Question 3

For a function

$$
f:\mathbb R\to\mathbb R^{100},
$$

which mode would naturally be more efficient for constructing the full Jacobian?

## Question 4

What is an adjoint $\bar v$?

## Question 5

Why do reverse-mode updates use accumulation such as

$$
\bar x\mathrel{+}=\cdots
$$

instead of simply assigning one value to $\bar x$?