# Forward and Reverse Automatic Differentiation by Hand — Solutions

# Part I — Forward-mode AD with dual numbers

## Solution 1.1 — Scalar input, scalar output

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

## Solution 1.2 — Vector input, scalar output

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

Forward propagation gives

$$
\hat x\hat y=2+\varepsilon,\qquad
\sin\hat z=2\varepsilon,\qquad
\hat y^2=4-4\varepsilon.
$$

Adding the three terms gives $g=6-\varepsilon$. The tangent is therefore $-1$, in agreement with $2(1)+5(-1)+1(2)=-1$.

Thus

$$
\boxed{D_{\mathbf v}g=-1}.
$$

---

## Solution 1.3 — Vector input, vector output

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

seed the inputs as $\hat x=1+\varepsilon$, $\hat y=2\varepsilon$, and $\hat z=1-\varepsilon$. One forward sweep gives

$$
\hat h_1=(1+\varepsilon)2\varepsilon+(1-\varepsilon)=1+\varepsilon,
$$

$$
\hat h_2=(1+\varepsilon)^2+\sin(2\varepsilon)-(1-\varepsilon)^2=6\varepsilon.
$$

The tangent vector is $(1,6)^T$, agreeing with multiplication of the Jacobian by $\mathbf v$.

Thus

$$
\boxed{
J_{\mathbf h}\mathbf v
=
\begin{pmatrix}1\\6\end{pmatrix}
}.
$$

---

# Part II — Reverse-mode AD / Backward AD

## Solution 2.1 — Scalar input, scalar output

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

Initialize all adjoints to zero before setting the output seed.

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

## Solution 2.2 — Three inputs, scalar output

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

The scalar output is $g=d$.

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

Initialize all adjoints to zero before setting the output seed.

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

This agrees with the gradient obtained in Exercise 1.2.

---

## Solution 2.3 — Three inputs, two outputs

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

# Solutions — Conceptual questions

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

## Forward mode versus reverse mode

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
| Forward mode | input direction $\mathbf v\in\mathbb R^n$ | $J\mathbf v$ | $n$ basis sweeps |
| Reverse mode | output direction $\mathbf w\in\mathbb R^m$ | $J^T\mathbf w$ | $m$ basis sweeps |

This gives an important rule of thumb:

- **Few inputs, many outputs:** forward mode is often attractive.
- **Many inputs, few outputs:** reverse mode is often attractive.
- **Many parameters, one scalar loss:** reverse mode is especially attractive.

That last case is exactly the situation encountered when training neural networks.

A reverse sweep requires a preceding forward pass to obtain the intermediate values. The sweep counts above refer to one seed direction per sweep.

### Key concepts

| Concept | Meaning in this exercise |
| --- | --- |
| **Automatic differentiation (AD)** | Computing derivatives by applying the chain rule to elementary operations. |
| **Loss** | A scalar measure of how poorly a model fits its target. |
| **Dual number** | $x+\dot x\varepsilon$, where $\varepsilon\ne0$ and $\varepsilon^2=0$. |
| **Input seed** | The initial tangent or input direction chosen for a forward sweep. |
| **Directional derivative** | Rate of change along $\mathbf x+t\mathbf v$; for a scalar output, $\nabla g^T\mathbf v$. |
| **Jacobian matrix** | Matrix of first partial derivatives, with outputs as rows and inputs as columns. |
| **Forward mode / JVP** | Propagates values and tangents to compute $J\mathbf v$. |
| **Sweep / pass** | One traversal of the calculation in a given direction. |
| **Computational graph** | Nodes and directed dependencies representing a calculation. |
| **Adjoint** | $\bar v=\partial L/\partial v$: sensitivity of the chosen scalar output $L$ to $v$. |
| **Output seed** | Initial output adjoints; $1$ for a scalar output, or weights $\mathbf w$ for several outputs. |
| **Reverse mode / VJP** | Propagates adjoints backwards to compute $J^T\mathbf w$, the transpose of $\mathbf w^TJ$. |
| **Accumulation** | Adding all contributions to an adjoint when a variable affects the output through several paths. |
