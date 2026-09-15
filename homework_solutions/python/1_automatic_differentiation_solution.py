import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 1 - Automatic differentiation

    In this exercise session, we will revisit the concepts of automatic differentiation (forward and
    backward AD) on simple functions, draw computation graphs. Lastly, we will try fitting a compound
    function of a gaussian and exponential using Minuit2 and see how we can boost its performance using
    AD.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Our functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Functions:
    """)
    return


@app.cell
def _(np):
    def f(x):
        # Original function 1 with a single argument
        return x**2 + np.sin(x)

    def F(x, y):
        # Original function 2 with two arguments
        return x**3 + np.sin(x)*y**2 + np.log(y)

    return F, f


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Values:
    """)
    return


@app.cell
def _(np):
    x0 = np.pi
    y0 = np.exp(1)
    return x0, y0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Todo
    Write the analytic expressions for the derivatives

    a) $f'(x) = \frac{d}{dx}f(x)$ and

    b) $F^x(x,y) = \frac{\partial}{\partial x}F(x,y)$ / $F^y(x,y) = \frac{\partial}{\partial y}F(x,y)$
    """)
    return


@app.cell
def _():
    def f1(x):
        return # TODO: write the analytical derivative f'(x) for f(x) defined above

    def Fx(x, y):
        return # TODO: write the analytical derivative F^x(x,y) for F(x,y) defined above

    def Fy(x, y):
        return # TODO: write the analytical derivative F^y(x,y) for F(x,y) defined above

    return (f1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Forward AD
    In this section, you will calculate the derivative of the functions $f(x)$ and $F(x,y)$ using dual numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Packages for dual numbers and forward AD:
    """)
    return


@app.cell
def _():
    import num_dual
    import math

    return (num_dual,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Playing with $f(x)$ / checking $f'(x)$
    Defining a dual number $(x,\dot{x})$ with $x=x_0$ and $\dot{x}=1$
    """)
    return


@app.cell
def _(num_dual, x0):
    x_dual = num_dual.Dual64(x0, 1.0)
    return (x_dual,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access real- and non-real part of the dual number
    """)
    return


@app.cell
def _(x_dual):
    x_real = x_dual.value
    print(x_real)
    return


@app.cell
def _(x_dual):
    x_nonreal = x_dual.first_derivative
    print(x_nonreal)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    Write an expression to check if the derivative from dual matches your analytical
    expression.
    """)
    return


@app.cell
def _(f, f1, f1_dual, num_dual, x0, x_dual):
    # Write here


    # Simplest Solution
    f_dual = f(x_dual)
    print("dual:")
    print(f"f(x_0)   = {f_dual.value}")
    print(f"f'(x_0)  = {f_dual.first_derivative}")
    print()
    print("analytic:")
    print(f"f(x_0)   = {f(x0)}")
    print(f"f'(x_0)  = {f1(x0)}")

    # Alternative Solution
    def f_dual(x):
        x_dual = num_dual.Dual64(x,1)
        return f(x_dual).first_derivative
    f1_dual(x0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Playing with F(x) / gradients
    - Write expressions to obtain $F^x(x_0,y_0)$ and $F^y(x_0,y_0)$ and compare with your analytic expressions
    - What does $F^{xy}(x_0,y_0)$ correspond to?
    """)
    return


@app.cell
def _(F, num_dual, x0, x_dual, y0):
    y_dual = num_dual.Dual64(y0, 1)

    # Write here

    # Simplest Solution
    Fx_dual = F(x_dual,y0)
    Fy_dual = F(x0,y_dual)
    Fxy_dual = F(x_dual,y_dual)

    print(f"F(x0,y0)     = {Fx_dual.value}")
    print(f"F^x(x0,y0)   = {Fx_dual.first_derivative}")
    print(f"F^y(x0,y0)   = {Fy_dual.first_derivative}")
    print(f"F^xy(x0,y0)  = {Fxy_dual.first_derivative}")

    # Alternative Solution
    def F_dual(x,y,dx,dy):
        x_dual = num_dual.Dual64(x,dx)
        y_dual = num_dual.Dual64(y,dy)
        return F(x_dual,y_dual).first_derivative

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    Draw the relevant computation graphs summarizing the procedure. You can also use online tools like excalidraw for diagrams.

    Use 'Export Image' to save your diagram and insert it in this document.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Backward AD
    For backward AD, we will use the JAX package in python (which also can do forward AD). To have the suport of basic functions, we redefine the functions $f$ and $F$ as well as the constants $x_0$ and $y_0$.

    (Jax needs the functions in its native language jax.numpy and cannot "read" standard numpy functions. This is due to functions like $\sin(x)$, $e^x$, ... not being native to the python language in contrast to Julia for example)
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    # Redefine functions:
    def f_jax(x):
        # Original function 1 with a single argument
        return x**2 + jnp.sin(x)

    def F_jax(x, y):
        # Original function 2 with two arguments
        return x**3 + jnp.sin(x)*y**2 + jnp.log(y)

    x0_jax = jnp.pi
    y0_jax = jnp.exp(1)
    return F_jax, f_jax, jax, jnp, x0_jax, y0_jax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pullback Functions / The Vector-Jacobian Product

    We can use jax.vjp to get the pullback function for gradient calculations.

    ### Todo
    Check that the derivative calculated using the pullback-function matches your prediction.
    """)
    return


@app.cell
def _(f_jax, jax, x0_jax):
    fx0, dfx0 = jax.vjp(f_jax, x0_jax) # function at x0 and pullback function for a weight, dfx0(1.0) gives gradient

    # Write here


    # Simplest solution
    print(f"f(x0)   = {fx0}")
    print(f"df(x0)  = {dfx0(1.0)}")
    return


@app.cell
def _(F_jax, jax, x0_jax, y0_jax):
    Fx0y0, dFx0y0 = jax.vjp(F_jax,x0_jax,y0_jax)

    # Write here

    # Simplest solution
    print(f"F(x0,y0)   = {Fx0y0}")
    print(f"df(x0,y0)  = {dFx0y0(1.0)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    Draw the relevant computation graphs summarizing the procedure. You can use online
    tools like excalidraw for diagrams and drop them here
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 Minuit2 Fitting Example
    In this example we generate a signal with background and use Minuit2 to minimize the parameters of our model function. Minuit2 uses central finite difference per default to obtain the best fitting parameters. By using faster differentiation methods like the ones from above, we can significantly fasten up the process of the parameter fitting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Histogram Function
    Defines a histogram-plotter to visualize fitting results.
    """)
    return


@app.cell
def _(np):
    import matplotlib.pyplot as plt

    def get_histogram(data, datasupport, nbins, plottitle):
        counts, edges = np.histogram(data, bins=nbins, range=datasupport)  # bin data in histogram
        centers = 0.5 * (edges[:-1] + edges[1:])                           # calculate bin-center
        bin_width = (datasupport[1] - datasupport[0]) / nbins              # width of a bin
        errors = np.sqrt(counts)                                           # Poisson-error of counts in a bin

        # Draw histogram:
        fig, ax = plt.subplots()

        ax.errorbar(centers, counts, yerr=errors, fmt="o",label="Data")
        ax.set_title(plottitle)
        ax.legend()

        return {
            "figure": fig,
            "axes": ax,
            "edges": edges,
            "centers": centers,
            "counts": counts,
            "errors": errors,
            "bin_width": bin_width,
            "n_events": counts.sum(),
        }

    return get_histogram, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sample Generator
    """)
    return


@app.cell
def _(np):
    from scipy.integrate import quad

    def sample_inversion(pdf, n, support, nbins=1000,):
        normalization = quad(pdf, support[0], support[1])[0]         # calculates area under PDF between supports
        grid = np.linspace(support[0], support[1], nbins)            # calculates a data-grid

        cdf = np.array([quad(pdf, support[0], x)[0] for x in grid])  # calculates all cummulative distribution function (CDF) for the data-grid

        samples = []

        while len(samples) < n:
            u = normalization * np.random.rand()                     # draw a random CDF value
            idx = np.searchsorted(cdf, u) - 1                        # find first calculated CDF smaller then u 

            # put the sample into the bin (position inside of bin gets randomized)
            x_left = grid[idx]
            x_right = grid[idx + 1]

            x = x_left + np.random.rand() * (x_right - x_left)

            samples.append(x)

        return np.array(samples)

    return (sample_inversion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    Write a function `signal_plus_background` which takes two inputs
    - x: input variable
    - pars: array (look at `gen_pars` below)
      - `mu`: mean of Gaussian signal
      - `sigma`: $\sigma$ of Gaussian signal
      - `tau`: parameter of the exponential background function
      - `a`: signal fraction
    """)
    return


app._unparsable_cell(
    r"""
    def signal_plus_background(x, pars):

        signal = 
        background = 

        return a * signal + (1-a) * background
    """,
    name="_"
)


@app.cell
def _(np):
    # Solution
    def signal_plus_background(x, pars):
        mu, sigma, tau, a = pars

        signal = 1 / np.sqrt(2*np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2*sigma**2))
        background = tau * np.exp(-tau*x)

        return a * signal + (1-a) * background

    return (signal_plus_background,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Parameters
    """)
    return


@app.cell
def _():
    gen_pars = [0.7, 0.06, 8.0, 0.5276] # mu, sigma, tau, a

    x_range = [0.0, 1.5] # Sample range
    return gen_pars, x_range


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate a Sample
    Simulates a data sample of 50,000 data points based on the input parameters `gen_pars` in the range `x_range` used to fit later
    """)
    return


@app.cell
def _(gen_pars, sample_inversion, signal_plus_background, x_range):
    data_sample = sample_inversion(
        lambda x: signal_plus_background(x, gen_pars),
        50000,
        x_range,
    )
    return (data_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    Plot the function to see how the distribution looks like
    """)
    return


@app.cell
def _(data_sample, get_histogram, plt, x_range):
    # Write here


    # Solution
    histogram = get_histogram(data_sample, x_range, 100, "Sample")

    fig = histogram["figure"]
    plt.xlabel("x")
    plt.ylabel("y")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    Define a negative log-likelihood function `nll`
    $$
    NLL = -\sum_{x\,\in\,\text{data}} \log(f(x))
    $$
    that takes the following input
    - `model`: a model function
    - `model_pars`: the fit parameters used in the model
    - `data`: the data the model is fit on

    in case the model-value of $f(x)>0$, add a rediculously large number (e.g. `1e10`) instead
    """)
    return


@app.cell
def _(np):
    # Write here


    # Solution
    def nll(model, model_pars, data):
        minus_sum_log = 0.0

        for x in data:
            value = model(x, model_pars)

            if value > 0:
                minus_sum_log -= np.log(value)
            else:
                minus_sum_log += 1e10

        return minus_sum_log

    return (nll,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Objective
    The objective we want the function to meet
    """)
    return


@app.cell
def _(data_sample, nll, signal_plus_background):
    n_sample_to_fit = 10000

    def objective(pars):
        return nll(signal_plus_background, pars, data_sample[0:n_sample_to_fit])

    return n_sample_to_fit, objective


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameter Fit
    Now we run our Minuit object and minimize it and have a look at the summary table. Give initial parameters different (but reasonable) to the original parameters.
    """)
    return


@app.cell
def _(objective):
    from iminuit import Minuit

    mu=1.0
    sigma=0.4
    tau=10.0
    a=0.9

    start_pars = mu, sigma, tau, a

    m = Minuit(objective, start_pars)

    m.migrad()
    return Minuit, start_pars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Todo
    We now calculate the time elapsed in executing the function using the time-method. We will
    compare three things:
    1. Time for fitting by Minuit without derivative function passed
    2. Time for fitting by Minuit with ForwardDiff derivative

    Please try repeating the tests by changing the number of sample points used to fit and see if it has an
    impact

    To do that, we need a function that measures the time elapsed for a particular process that also takes the gradient of the objective function as an input
    """)
    return


@app.cell
def _(Minuit):
    import time

    def get_pars_with_time(objective, start_pars, gradient=None):

        t0 = time.perf_counter()

        m = Minuit(objective,start_pars,grad=gradient)

        m.migrad()

        t = time.perf_counter() - t0

        return m, t

    return (get_pars_with_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to differentiate the objective function, we have to make it Jax-readable as well:
    """)
    return


@app.cell
def _(data_sample, jnp, n_sample_to_fit):
    # Write here


    # Solution
    def signal_plus_background_jax(x, pars):
        mu, sigma, tau, a = pars

        signal = 1 / jnp.sqrt(2*jnp.pi * sigma**2) * jnp.exp(-(x - mu)**2 / (2*sigma**2))
        background = tau * jnp.exp(-tau*x)

        return a * signal + (1-a) * background

    def nll_jax(model, model_pars, data):
        values = model(data, model_pars)
        values = jnp.clip(values, 1e-300)

        return -jnp.sum(jnp.log(values))

    def objective_jax(pars):
        return nll_jax(signal_plus_background_jax, pars, data_sample[0:n_sample_to_fit])

    return (objective_jax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we need a function which computes the gradients using AD. We will use `jax.grad()` in our
    example and define a local function which calculates the derivative for the objective defined above.
    """)
    return


@app.cell
def _(jax, objective_jax):
    def forward_gradient_function(pars):
        return jax.grad(objective_jax)(pars)

    return (forward_gradient_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Benchmark Time
    To get a benchmark for the elapsed time, we run `get_pars_with_time` without using a gradient (and thus there is no need to use `objective_jax`)

    ###(Todo)
    (Test with `objective_jax` instead, what do you see?)
    """)
    return


@app.cell
def _(get_pars_with_time, objective, start_pars):
    m_noderiv, t_noderiv = get_pars_with_time(objective, start_pars)
    return


@app.cell
def _(
    forward_gradient_function,
    get_pars_with_time,
    objective_jax,
    start_pars,
):
    m_withderiv, t_withderiv = get_pars_with_time(objective_jax, start_pars, gradient=forward_gradient_function,)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualize Code
    Write a visualization for all your Minuit results
    """)
    return


@app.cell
def _():
    # Write here
    return


if __name__ == "__main__":
    app.run()
