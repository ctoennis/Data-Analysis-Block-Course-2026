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

# Part I - Forward-mode AD with dual numbers

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

## Exercise 1 - Scalar input, scalar output

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

## Exercise 2 - Vector input, scalar output

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

## Exercise 3 - Vector input, vector output

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

# Solutions - Forward mode

## Solution 1

We start with

$$
\hat{x}=1+\varepsilon.
$$

### Step 1: exponential

$$
a=e^{\hat{x}}
=e^{1+\varepsilon}
=e+e\varepsilon.
$$

### Step 2: multiplication

$$
b=\hat{x}a
=(1+\varepsilon)(e+e\varepsilon).
$$

Expanding and dropping $\varepsilon^2$,

$$
b=e+2e\varepsilon.
$$

### Step 3: addition

$$
c=b+3
=(e+3)+2e\varepsilon.
$$

### Step 4: logarithm

Using

$$
\log(a+b\varepsilon)
=\log a+\frac{b}{a}\varepsilon,
$$

we obtain

$$
f(\hat{x})
=
\log(e+3)
+
\frac{2e}{e+3}\varepsilon.
$$

Therefore

$$
\boxed{f(1)=\log(e+3)}
$$

and

$$
\boxed{
f'(1)=\frac{2e}{e+3}
}.
$$

---

## Solution 2

The function is

$$
g(x,y,z)=xy+\sin z+y^2.
$$

At $(1,2,0)$,

$$
g(1,2,0)=1\cdot2+\sin 0+2^2=6.
$$

### Seed in the $x$-direction

Use

$$
\hat{x}=1+\varepsilon,
\qquad
\hat{y}=2,
\qquad
\hat{z}=0.
$$

Then

$$
\hat{x}\hat{y}
=(1+\varepsilon)2
=2+2\varepsilon,
$$

$$
\sin\hat{z}=0,
$$

and

$$
\hat{y}^2=4.
$$

Thus

$$
g=6+2\varepsilon.
$$

Therefore

$$
\frac{\partial g}{\partial x}=2.
$$

### Seed in the $y$-direction

Use

$$
\hat{x}=1,
\qquad
\hat{y}=2+\varepsilon,
\qquad
\hat{z}=0.
$$

Then

$$
\hat{x}\hat{y}=2+\varepsilon,
$$

and

$$
\hat{y}^2
=(2+\varepsilon)^2
=4+4\varepsilon.
$$

Hence

$$
g=6+5\varepsilon,
$$

so

$$
\frac{\partial g}{\partial y}=5.
$$

### Seed in the $z$-direction

Use

$$
\hat{z}=\varepsilon.
$$

Since

$$
\sin(\varepsilon)
=\sin 0+\cos 0\,\varepsilon
=\varepsilon,
$$

we get

$$
g=6+\varepsilon.
$$

Therefore

$$
\frac{\partial g}{\partial z}=1.
$$

The gradient is

$$
\boxed{
\nabla g(1,2,0)
=
\begin{pmatrix}
2\\5\\1
\end{pmatrix}
}.
$$

### Directional derivative

For

$$
\mathbf v=(1,-1,2)^T,
$$

seed all inputs at once:

$$
\hat{x}=1+\varepsilon,
\qquad
\hat{y}=2-\varepsilon,
\qquad
\hat{z}=2\varepsilon.
$$

Forward propagation gives tangent

$$
2(1)+5(-1)+1(2)=-1.
$$

Thus

$$
\boxed{D_{\mathbf v}g=-1}.
$$

---

## Solution 3

At $(1,0,1)$,

$$
h_1=1\cdot0+1=1,
$$

$$
h_2=1^2+\sin 0-1^2=0.
$$

Therefore

$$
\boxed{
\mathbf h(1,0,1)
=
\begin{pmatrix}1\\0\end{pmatrix}
}.
$$

### Seed $\mathbf e_x$

Use

$$
\hat{x}=1+\varepsilon,
\qquad
\hat{y}=0,
\qquad
\hat{z}=1.
$$

For the first component,

$$
h_1=(1+\varepsilon)0+1=1+0\varepsilon.
$$

For the second component,

$$
h_2=(1+\varepsilon)^2+\sin 0-1
=2\varepsilon.
$$

So the first Jacobian column is

$$
\begin{pmatrix}0\\2\end{pmatrix}.
$$

### Seed $\mathbf e_y$

Use

$$
\hat{y}=\varepsilon.
$$

Then

$$
h_1=1+\varepsilon,
$$

and

$$
h_2=\sin(\varepsilon)=\varepsilon.
$$

So the second Jacobian column is

$$
\begin{pmatrix}1\\1\end{pmatrix}.
$$

### Seed $\mathbf e_z$

Use

$$
\hat{z}=1+\varepsilon.
$$

Then

$$
h_1=1+\varepsilon,
$$

while

$$
h_2
=1-(1+\varepsilon)^2
=-2\varepsilon.
$$

So the third Jacobian column is

$$
\begin{pmatrix}1\\-2\end{pmatrix}.
$$

Therefore

$$
\boxed{
J_{\mathbf h}(1,0,1)
=
\begin{pmatrix}
0 & 1 & 1\\
2 & 1 & -2
\end{pmatrix}
}.
$$

For

$$
\mathbf v=(1,2,-1)^T,
$$

one forward sweep gives

$$
J_{\mathbf h}\mathbf v
=
\begin{pmatrix}
0 & 1 & 1\\
2 & 1 & -2
\end{pmatrix}
\begin{pmatrix}
1\\2\\-1
\end{pmatrix}
=
\begin{pmatrix}
1\\6
\end{pmatrix}.
$$

Thus

$$
\boxed{
J_{\mathbf h}\mathbf v
=
\begin{pmatrix}1\\6\end{pmatrix}
}.
$$

---

# Part II - Reverse-mode AD / Backward AD

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

## Exercise 1 - Scalar input, scalar output

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

Suggested computational graph:

$$
x
\longrightarrow
v_1=e^x,
$$

$$
(x,v_1)
\longrightarrow
v_2=xv_1,
$$

$$
v_2
\longrightarrow
v_3=v_2+3,
$$

$$
v_3
\longrightarrow
v_4=\log v_3=f.
$$

Pay attention to the fact that $x$ affects the result through two paths: directly through $v_2=xv_1$, and indirectly through $v_1=e^x$.

---

## Exercise 2 - Three inputs, scalar output

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
d=a+b,
\qquad
g=d+c.
$$

---

## Exercise 3 - Three inputs, two outputs

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

# Solutions - Reverse mode

## Solution 1

Define

$$
v_1=e^x,
\qquad
v_2=xv_1,
\qquad
v_3=v_2+3,
\qquad
v_4=\log v_3.
$$

The final output is

$$
f=v_4.
$$

### Forward pass

At $x=1$,

$$
v_1=e,
$$

$$
v_2=e,
$$

$$
v_3=e+3,
$$

$$
v_4=\log(e+3).
$$

### Backward pass

Initialize

$$
\bar v_4=1.
$$

Because

$$
v_4=\log v_3,
$$

$$
\bar v_3
=
\bar v_4\frac{1}{v_3}
=
\frac{1}{e+3}.
$$

Because

$$
v_3=v_2+3,
$$

$$
\bar v_2
=
\bar v_3
=
\frac{1}{e+3}.
$$

Now

$$
v_2=xv_1.
$$

This gives one direct contribution to $x$:

$$
\bar x_{\text{direct}}
=
\bar v_2 v_1
=
\frac{e}{e+3}.
$$

It also gives

$$
\bar v_1
=
\bar v_2 x
=
\frac{1}{e+3}.
$$

Finally,

$$
v_1=e^x,
$$

so the indirect contribution to $x$ is

$$
\bar x_{\text{via }v_1}
=
\bar v_1 e^x
=
\frac{e}{e+3}.
$$

The two paths must be added:

$$
\bar x
=
\frac{e}{e+3}
+
\frac{e}{e+3}.
$$

Therefore

$$
\boxed{
f'(1)=\bar x=\frac{2e}{e+3}
}.
$$

This is exactly the same derivative obtained with dual numbers.

---

## Solution 2

Use

$$
a=xy,
\qquad
b=\sin z,
\qquad
c=y^2,
\qquad
d=a+b+c
$$

### Forward pass

At $(1,2,0)$,

$$
a=2,
\qquad
b=0,
\qquad
c=4,
\qquad
d=6
$$

### Backward pass

Start with

$$
\bar d=1.
$$

Since

$$
d=a+b+c,
$$

we obtain

$$
\bar a=1,
\qquad
\bar b=1,
\qquad
\bar c=1
$$


From

$$
a=xy,
$$

we get

$$
\bar x
\mathrel{+}=
\bar a\,y
=1\cdot2
=2,
$$

and

$$
\bar y
\mathrel{+}=
\bar a\,x
=1\cdot1
=1.
$$

From

$$
b=\sin z,
$$

we obtain

$$
\bar z
\mathrel{+}=
\bar b\cos z
=1\cdot1
=1.
$$

From

$$
c=y^2,
$$

we obtain another contribution to $y$:

$$
\bar y
\mathrel{+}=
\bar c\,2y
=1\cdot4
=4.
$$

Therefore

$$
\bar y=1+4=5.
$$

The final gradient is

$$
\boxed{
\nabla g(1,2,0)
=
\begin{pmatrix}
2\\5\\1
\end{pmatrix}
}.
$$

The adjoint of $y$ gets two contributions because $y$ influences the output through both

$$
xy
$$

and

$$
y^2.
$$

Reverse mode must accumulate both paths.

---

## Solution 3

The two outputs are

$$
h_1=xy+z,
$$

and

$$
h_2=x^2+\sin y-z^2.
$$

At $(1,0,1)$,

$$
\mathbf h(1,0,1)
=
\begin{pmatrix}1\\0\end{pmatrix}.
$$

### Reverse sweep from $h_1$

Use output seed

$$
\mathbf w_1
=
\begin{pmatrix}1\\0\end{pmatrix}.
$$

Only $h_1$ contributes.

Since

$$
h_1=xy+z,
$$

we have

$$
\frac{\partial h_1}{\partial x}=y=0,
$$

$$
\frac{\partial h_1}{\partial y}=x=1,
$$

$$
\frac{\partial h_1}{\partial z}=1.
$$

Thus

$$
\boxed{
J_{\mathbf h}^T\mathbf w_1
=
\begin{pmatrix}
0\\1\\1
\end{pmatrix}
}.
$$

This is the first **row** of the Jacobian, written as a column vector.

### Reverse sweep from $h_2$

Use

$$
\mathbf w_2
=
\begin{pmatrix}0\\1\end{pmatrix}.
$$

Since

$$
h_2=x^2+\sin y-z^2,
$$

we obtain

$$
\frac{\partial h_2}{\partial x}=2x=2,
$$

$$
\frac{\partial h_2}{\partial y}=\cos y=1,
$$

$$
\frac{\partial h_2}{\partial z}=-2z=-2.
$$

Therefore

$$
\boxed{
J_{\mathbf h}^T\mathbf w_2
=
\begin{pmatrix}
2\\1\\-2
\end{pmatrix}
}.
$$

Putting the two rows together gives

$$
\boxed{
J_{\mathbf h}(1,0,1)
=
\begin{pmatrix}
0 & 1 & 1\\
2 & 1 & -2
\end{pmatrix}
}.
$$

### General output seed

Let

$$
\mathbf w
=
\begin{pmatrix}3\\-1\end{pmatrix}.
$$

Then

$$
J_{\mathbf h}^T\mathbf w
=
3\nabla h_1-\nabla h_2.
$$

Therefore

$$
J_{\mathbf h}^T\mathbf w
=
3
\begin{pmatrix}0\\1\\1\end{pmatrix}
-
\begin{pmatrix}2\\1\\-2\end{pmatrix}
=
\begin{pmatrix}-2\\2\\5\end{pmatrix}.
$$

Hence

$$
\boxed{
J_{\mathbf h}^T\mathbf w
=
\begin{pmatrix}-2\\2\\5\end{pmatrix}
}.
$$

---

# Part III - Forward mode versus reverse mode

The most important distinction is not that one method is "better" than the other. They compute different Jacobian products efficiently.

For

$$
f:\mathbb R^n\to\mathbb R^m
$$

with Jacobian

$$
J\in\mathbb R^{m\times n},
$$

we have:

| Mode | Seed | One sweep computes | Full Jacobian requires |
|---|---|---|---|
| Forward mode | input direction $\mathbf v\in\mathbb R^n$ | $J\mathbf v$ | roughly $n$ basis sweeps |
| Reverse mode | output direction $\mathbf w\in\mathbb R^m$ | $J^T\mathbf w$ | roughly $m$ basis sweeps |

This gives an important rule of thumb:

- **Few inputs, many outputs:** forward mode is often attractive.
- **Many inputs, few outputs:** reverse mode is often attractive.
- **Many parameters, one scalar loss:** reverse mode is especially attractive.

That last case is exactly the situation encountered when training neural networks.

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

---

# Solutions - Conceptual questions

## Solution 1

When arithmetic is expanded with

$$
\varepsilon^2=0,
$$

all terms of second and higher order in $\varepsilon$ vanish. What remains is exactly the first-order term of the function expansion:

$$
f(x+\dot x\varepsilon)
=f(x)+f'(x)\dot x\varepsilon.
$$

Therefore the coefficient of $\varepsilon$ follows the chain rule automatically.

## Solution 2

For

$$
f:\mathbb R^{100}\to\mathbb R,
$$

the full gradient has 100 input derivatives.

Forward mode needs 100 basis-direction sweeps.

Reverse mode needs only one reverse sweep because the output is scalar.

## Solution 3

For

$$
f:\mathbb R\to\mathbb R^{100},
$$

there is only one input direction. One forward sweep gives the complete single Jacobian column, i.e. all 100 output derivatives with respect to the one input.

Forward mode is therefore the natural choice.

## Solution 4

The adjoint of an intermediate variable $v$ is

$$
\bar v=\frac{\partial L}{\partial v},
$$

where $L$ is the final scalar output being differentiated.

It measures how sensitive the final output is to a change in that intermediate variable.

## Solution 5

A variable may influence the output through several different paths in the computational graph. Each path contributes to the total derivative. Reverse mode must therefore add all contributions to the variable's adjoint.

---

# Optional exam-style summary

A concise answer to "What are dual numbers and how are they used in forward-mode AD?" could be:

> A dual number has the form $a+b\varepsilon$ with $\varepsilon^2=0$. In forward-mode AD, the primal input is augmented with a tangent, $x+\dot x\varepsilon$. Evaluating the function with dual-number arithmetic produces $f(x)+J_f(x)\dot x\varepsilon$, so the coefficient of $\varepsilon$ is the propagated directional derivative.

A concise answer to "What is the role of a computational graph in reverse-mode AD?" could be:

> A computational graph decomposes a function into elementary operations. A forward pass stores intermediate values. A backward pass starts with output adjoint 1 and applies local derivatives in reverse order, accumulating adjoints for all parent nodes. For a scalar output, one reverse pass yields derivatives with respect to all inputs.
