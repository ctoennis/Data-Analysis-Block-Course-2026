import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import uproot
    import pandas as pd
    import math
    import matplotlib.pyplot as plt

    return math, mo, plt, uproot


@app.cell
def _(uproot):
    dfbkg = None
    dfsig = None

    with uproot.open("background.root") as file:

        tbkg = file["t"]
        dfbkg = tbkg.arrays(library="pd")

    with uproot.open("signal.root") as file:

        tsig = file["t"]
        dfsig = tsig.arrays(library="pd")
    return dfbkg, dfsig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Exercise 1
    We would like to calculate the Gini Index and $\frac{S}{\sqrt{S+B}}$ when applying a selection on the Lc\_PT variable.

       a) Discuss how you would do this for a given cut value, and the signal and background samples at hand.

       b) Do you expect to get the same result when measuring the number on actual data?

       c) Write a generic function that computes any figure of merit, given a selection, signal and background samples. The function documentation is already given below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    fom_for_cut(sdf::DataFrame, bdf::DataFrame, var::AbstractString, cut::Real, fom::Function; tf::Function=identity)

    Compute a figure of merit (FoM) for a given cut value on a specified variable.

    Arguments
    - `sdf::DataFrame`: DataFrame containing the signal sample.
    - `bdf::DataFrame`: DataFrame containing the background sample.
    - `var::AbstractString`: Name of the variable (column) to cut on.
    - `cut::Real`: Threshold value; events with `tf(row[var]) > cut` are selected.
    - `fom::Function`: A function `fom(s, b)` that computes the figure of merit from the number of signal (`s`) and background (`b`) events.
    - `op::Function=>(a, b)`: Comparison operator (e.g. `>`, `<`, `>=`, `<=`). The default is `>`.
    - `tf::Function=identity`: An optional transformation applied to the variable before cutting (e.g., `log10`).
    - `nsig::Real`: Use this optional number of signal events, instead of the number of events in sdf.
    - `nbkg::Real`: Use this optional number of background events, instead of the number of events in bdf.

    Returns
    - The computed figure of merit for the given cut.
    """)
    return


@app.function
def fom_for_cut(
    sdf,
    bdf,
    var,
    cut,
    fom,
    op=lambda x, y: x > y,
    tf=lambda x: x,
    nsig=None,
    nbkg=None,
):
    """
    Compute a figure of merit (FoM) for a given cut value.

    Parameters
    ----------
    sdf : pandas.DataFrame
        DataFrame containing the signal sample.
    bdf : pandas.DataFrame
        DataFrame containing the background sample.
    var : str
        Name of the variable (column) to cut on.
    cut : float
        Threshold value. Events satisfying ``op(tf(value), cut)``
        are selected.
    fom : callable
        Function ``fom(s, b)`` that computes the figure of merit
        from the selected signal and background event counts.
    op : callable, optional
        Comparison function, e.g. ``operator.gt`` or ``operator.lt``.
        Defaults to ``>``.
    tf : callable, optional
        Transformation applied to the variable before cutting.
        Defaults to the identity function.
    nsig : float, optional
        Total number of signal events to use instead of ``len(sdf)``.
    nbkg : float, optional
        Total number of background events to use instead of ``len(bdf)``.

    Returns
    -------
    float
        The computed figure of merit.
    """
    if var not in sdf.columns:
        raise ValueError(f"Variable '{var}' not found in signal DataFrame.")

    if var not in bdf.columns:
        raise ValueError(f"Variable '{var}' not found in background DataFrame.")

    # Apply transformation and cut
    signal_selected = op(sdf[var].map(tf), cut)
    background_selected = op(bdf[var].map(tf), cut)

    # Number of selected events
    n_signal_selected = signal_selected.sum()
    n_background_selected = background_selected.sum()

    # Total event counts (or user-supplied normalization)
    total_signal = len(sdf) if nsig is None else nsig
    total_background = len(bdf) if nbkg is None else nbkg

    # Scale selected events if alternative total counts are supplied
    s = n_signal_selected / len(sdf) * total_signal
    b = n_background_selected / len(bdf) * total_background

    return fom(s, b)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2
    a) Use the function to scan cut values and produce a plot of the Gini Index and $\frac{S}{\sqrt{S+B}}$. Think about a reasonable range and transformation of the variable. Hint: normalize to plot both curves in the same pad.

    b) Plot the ROC curve, and add a marker for the point where the figures of merit are maximal. Hint: you will need signal and background efficiencies for the ROC curve. They can be obtained from the generic function we wrote earlier.

    c) Discuss the results.
    """)
    return


@app.cell
def _(mo):
    sig_slider = mo.ui.slider(
        start=0.0,
        stop=100000.0,
        step=10001,
        value=10001,
        label="Signal",
    )
    bg_slider = mo.ui.slider(
        start=0.0,
        stop=100000.0,
        step=10001,
        value=10001,
        label="Signal",
    )
    return bg_slider, sig_slider


@app.cell
def _(bg_slider, sig_slider):
    nsig= sig_slider.value
    nbkg= bg_slider.value
    return nbkg, nsig


@app.cell
def _(mo, nsig, sig_slider):
    mo.vstack([
        sig_slider,
        mo.md(f"Current sig: **{nsig:.2f}**"),
    ])
    return


@app.cell
def _(bg_slider, mo, nbkg):
    mo.vstack([
        bg_slider,
        mo.md(f"Current bg: **{nbkg:.2f}**"),
    ])
    return


@app.cell
def _(dfbkg, dfsig, math, nbkg, nsig):
    cut_values = [3.2+x*0.8/120 for x in range(120)]

    # Initialize storage
    ginis = []
    significances = []
    sig_effs = []
    bkg_effs = []

    # Loop through cut values
    for cut in cut_values:
        ginis.append(fom_for_cut(dfsig,dfbkg,"Lc_PT",cut,lambda s , b : s/math.sqrt(s+b), tf=lambda x : math.log10(x), nsig = nsig, nbkg = nbkg))
        significances.append(fom_for_cut(dfsig,dfbkg,"Lc_PT",cut,lambda s , b : 2*s*b/(s+b)**2, tf=lambda x : math.log10(x), nsig = nsig, nbkg = nbkg))
        sig_effs.append(fom_for_cut(dfsig,dfbkg,"Lc_PT",cut,lambda s , b : s, tf=lambda x : math.log10(x), nsig = nsig, nbkg = nbkg))
        bkg_effs.append(fom_for_cut(dfsig,dfbkg,"Lc_PT",cut,lambda s , b : b, tf=lambda x : math.log10(x), nsig = nsig, nbkg = nbkg))

    best_index_gini = ginis.index(max(ginis))                                                                       
    best_index_significances = significances.index(max(significances))
    best_cut_gini = cut_values[best_index_gini]                                                           
    best_cut_significances = cut_values[best_index_significances]
    print("Best cut gini: " + str(best_cut_gini))
    print("Best cut best_cut_significances: " + str(best_cut_significances))
    return cut_values, ginis, significances


@app.cell
def _(cut_values, ginis, plt):
    plt.plot(cut_values, ginis, marker='o')

    # Add labels and title
    plt.xlabel("Cut values")
    plt.ylabel("Gini indices")
    plt.title("Gini indices")
    return


@app.cell
def _(cut_values, plt, significances):
    plt.plot(cut_values, significances, marker='o')

    # Add labels and title
    plt.xlabel("Cut values")
    plt.ylabel("significances")
    plt.title("significances")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 3
    We will apply the previously found best cut-values to the test dataset, and fit the signal in the variable "Lc_M" together. Then we discuss the results.
    """)
    return


@app.cell
def _(uproot):
    dftest = None

    with uproot.open("test.root") as file:

        ttest = file["t"]
        dtest = ttest.arrays(library="pd")
    return


if __name__ == "__main__":
    app.run()
