import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import uproot
    import pandas as pd
    import matplotlib.pyplot as plt

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boosting techniques for decision trees
In this exercise we're using the same datasets as in Exercise 1B, and we're going to explore the full dataset to see if we can get a better discrimination of signal and background.
    """)
    return

@app.cell
def _(uproot):
    dfbkg = None
    dfsig = None
    dftest = None

    with uproot.open("background.root") as file:

        tbkg = file["t"]
        dfbkg = tbkg.arrays(library="pd")

    with uproot.open("signal.root") as file:

        tsig = file["t"]
        dfsig = tsig.arrays(library="pd")

    with uproot.open("test.root") as file:

        ttest = file["t"]
        dftest = ttest.arrays(library="pd")

    return dfbkg, dfsig, dftest

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pre-processing of data

Before training a ML algorithm, it makes sense to take a closer look at the features we'd like to use for training.

Let's check the extrema of the data.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Not all variables are in the same domain.

In the case of the `pi_OWNPVIPCHI2` variable, there is a region, where the algorithm would see only signal (between ~1 and ~4).

For instance, when training BDTs with AdaBoost, this variable would maximise the Gini index below values of 4, since there is only signal.

In the case of the samples used in this exercise, there were different pre-processing workflows, and the variable domains are not the same for all data.

We are now skipping an intermediate step in which we would plot the distributions and decide on transformations of variables.

In our case, there are a few variables where it's easier to look at the distribution with a log transform:
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By skipping the intermediate step, the transformation and trimming of variables can be done in one go.

We can also do the splitting of signal and background samples into training and validation datasets in the same step.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1
Let's start with something boring but very common:
Implement the transformations (in our case only log-transoformations) and trim the signal and background samples to a common support.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Training BDTs with AdaBoost
We are ready to train a BDT now.

We start with a wrapper function that helps with using dataframes, and gives us flexibility to choose the features and the number of trees to train.

Implement the function to train an `AdaBoostStumpClassifier`. You can find rudimentary documentation in the Live Docs and https://docs.juliahub.com/DecisionTree/pEDeB/0.12.4/
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: A closer look at the results
We have trained and applied the BDT.
Since it may still feel like a black box, let's look at the first three trees in detail, and compute the output ourselves.

Do you get the same result if you only train 3 trees, and compare the result of `predict_proba` to the manual 3rd iteration?
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4: GradientBDTs
Training a gradient BDT is similar to the AdaBoost example. However, you have learned in the lecture that the boosting works in a different way. We will explore some of the most important differences together, to get an idea why GBDTs are more popular and performant for most classification tasks.
    """)
    return

if __name__ == "__main__":
    app.run()
