import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import uproot
    import pandas as pd

    return mo, uproot


@app.cell
def _(uproot):
    with uproot.open("background.root") as file:

        tbkg = file["t"]
        dfbkg = tbkg.arrays(library="pd")

    with uproot.open("signal.root") as file:

        tsig = file["t"]
        dfsig = tsig.arrays(library="pd")
    return


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

    # Arguments
    - `sdf::DataFrame`: DataFrame containing the signal sample.
    - `bdf::DataFrame`: DataFrame containing the background sample.
    - `var::AbstractString`: Name of the variable (column) to cut on.
    - `cut::Real`: Threshold value; events with `tf(row[var]) > cut` are selected.
    - `fom::Function`: A function `fom(s, b)` that computes the figure of merit from the number of signal (`s`) and background (`b`) events.
    - `op::Function=>(a, b)`: Comparison operator (e.g. `>`, `<`, `>=`, `<=`). The default is `>`.
    - `tf::Function=identity`: An optional transformation applied to the variable before cutting (e.g., `log10`).
    - `nsig::Real`: Use this optional number of signal events, instead of the number of events in sdf.
    - `nbkg::Real`: Use this optional number of background events, instead of the number of events in bdf.

    # Returns
    - `Float64`: The computed figure of merit for the given cut.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2
    a) Use the function to scan cut values and produce a plot of the Gini Index and $\frac{S}{\sqrt{S+B}}$. Think about a reasonable range and transformation of the variable. Hint: normalize to plot both curves in the same pad.

    b) Plot the ROC curve, and add a marker for the point where the figures of merit are maximal. Hint: you will need signal and background efficiencies for the ROC curve. They can be obtained from the generic function we wrote earlier.

    c) Discuss the results.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 3
We will apply the previously found best cut-values to the test dataset, and fit the signal in the variable "Lc_M" together. Then we discuss the results.
    """)
    return

@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
