import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import uproot
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    return AdaBoostClassifier, DecisionTreeClassifier, mo, np, pd, plt, uproot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Heart disease example

    chest pain, circulation, blocked arteries, weight, heart disease target

    make stumps

    guess weights

    see classification
    """)
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


@app.cell
def _(dfbkg):
    max_values = dfbkg.max()
    print(max_values)
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


@app.cell
def _():
    log_transform = ["K_OWNPVIPCHI2",
    "pi_PT",
    "Xi_PT",
    "Lc_OWNPVVDRHO",
    "Lc_OWNPVVDZ",
    "pi_OWNPVIPCHI2",
    "Lc_OWNPVFDCHI2",
    "Xi_OWNPVIP",
    "K_PT",
    "Lc_PT"
    ]
    return (log_transform,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We need now a function to plit the dataample into training and validation sets:
    """)
    return


@app.function
def split_dataframe(df, fraction, random_state=None):
    """
    Randomly split a DataFrame into two parts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    fraction : float
        Fraction of entries to put in the first DataFrame.
        Must be between 0 and 1.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    df1, df2 : pd.DataFrame
        The two resulting DataFrames.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    df_shuffled = df.sample(frac=1, random_state=random_state)

    split_index = int(len(df) * fraction)

    df1 = df_shuffled.iloc[:split_index]
    df2 = df_shuffled.iloc[split_index:]

    return df1, df2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1
    Let's start with something boring but very common:
    Implement the transformations (in our case only log-transoformations) and trim the signal and background samples to a common support.
    """)
    return


@app.cell
def _(np):
    def log_transformer(df, columns):
        df = df.copy()

        for column in columns:
            df[column] = np.log(df[column])

        return df

    return (log_transformer,)


@app.cell
def _(pd):
    def trim_to_common_support(dataframes, columns):
        """
        Trim multiple DataFrames so that every specified column
        has the same min/max range across all DataFrames.
        """
        common_min = {
            col: max(df[col].min() for df in dataframes)
            for col in columns
        }

        common_max = {
            col: min(df[col].max() for df in dataframes)
            for col in columns
        }

        for col in columns:
            if common_min[col] > common_max[col]:
                raise ValueError(f"No common support for column '{col}'.")

        trimmed_dataframes = []

        for df in dataframes:
            mask = pd.Series(True, index=df.index)

            for col in columns:
                mask &= (
                    (df[col] >= common_min[col]) &
                    (df[col] <= common_max[col])
                )

            trimmed_dataframes.append(df.loc[mask].copy())

        return trimmed_dataframes

    return (trim_to_common_support,)


@app.cell
def _(
    dfbkg,
    dfsig,
    dftest,
    log_transform,
    log_transformer,
    trim_to_common_support,
):
    dftestlog = log_transformer(dftest,log_transform)
    dfsiglog = log_transformer(dfsig,log_transform)
    dfbkglog = log_transformer(dfbkg,log_transform)

    feature_names = dfsiglog.columns.tolist()

    dftestlogtrim, dfsiglogtrim, dfbkglogtrim = trim_to_common_support([dftestlog,dfsiglog,dfbkglog],feature_names)

    dfsiglogtrim["label"] = 1.0
    dfbkglogtrim["label"] = 0.0
    return dfbkglogtrim, dfsiglogtrim, feature_names


@app.cell
def _(dfsig, plt):
    plt.hist(dfsig["Lc_PT"], bins=30)
    return


@app.cell
def _(dfbkglogtrim, dfsiglogtrim):
    dfsig_val, dfsig_train = split_dataframe(dfsiglogtrim,0.8)
    dfbkg_val, dfbkg_train = split_dataframe(dfbkglogtrim,0.8)
    return dfbkg_train, dfbkg_val, dfsig_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Training BDTs with AdaBoost
    We are ready to train a BDT now.

    We start with a wrapper function that helps with using dataframes, and gives us flexibility to choose the features and the number of trees to train.

    Implement the function to train an `AdaBoostStumpClassifier`. You can find rudimentary documentation in the Live Docs and https://docs.juliahub.com/DecisionTree/pEDeB/0.12.4/
    """)
    return


@app.cell
def _(AdaBoostClassifier, DecisionTreeClassifier):
    def train_adaboost_stump(df, features):
        """
        Train an AdaBoost classifier using decision stumps.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data.
        features : list of str
            List of column names to use as features.

        Returns
        -------
        model : AdaBoostClassifier
            Trained AdaBoost model.
        """

        X = df[features]
        y = df["label"]

        # Decision stump
        stump = DecisionTreeClassifier(max_depth=1)

        # AdaBoost using decision stumps
        model = AdaBoostClassifier(
            estimator=stump,
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )

        model.fit(X, y)

        return model

    return (train_adaboost_stump,)


@app.cell
def _(dfbkg_train, dfsig_train, pd):
    df_train = pd.concat([dfsig_train, dfbkg_train], ignore_index=True)
    return (df_train,)


@app.cell
def _(df_train, feature_names, train_adaboost_stump):
    adabdt_tf = train_adaboost_stump(df_train, feature_names)
    return (adabdt_tf,)


@app.cell
def _(adabdt_tf, dfbkg_train, dfbkg_val, dfsig_train, np, plt):
    X_sig_train = dfsig_train.drop(columns=["label"],errors="ignore")
    X_sig_val = dfbkg_val.drop(columns=["label"],errors="ignore")
    X_bkg_train = dfbkg_train.drop(columns=["label"],errors="ignore")
    X_bkg_val = dfbkg_val.drop(columns=["label"],errors="ignore")

    p_sig_train = adabdt_tf.predict_proba(X_sig_train)[:, 1]
    p_bkg_train = adabdt_tf.predict_proba(X_bkg_train)[:, 1]

    bins = np.linspace(0, 0.7, 50)

    plt.hist([p_sig_train,p_bkg_train], bins, alpha=0.5)

    plt.show()
    return X_bkg_val, X_sig_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: A closer look at the results
    We have trained and applied the BDT.
    Since it may still feel like a black box, let's look at the first three trees in detail, and compute the output ourselves.

    Do you get the same result if you only train 3 trees, and compare the result of `predict_proba` to the manual 3rd iteration?
    """)
    return


@app.cell
def _(np):
    def adaboost_response(model, X, n_trees=None):
        """
        Calculate the AdaBoost decision response using the first n_trees.

        Parameters
        ----------
        model : sklearn.ensemble.AdaBoostClassifier
            A fitted AdaBoostClassifier.
        X : array-like
            Input samples.
        n_trees : int, optional
            Number of trees/estimators to use. If None, use all trees.

        Returns
        -------
        response : np.ndarray
            AdaBoost decision response for each sample.
        """
        estimators = model.estimators_
        estimator_weights = model.estimator_weights_

        if n_trees is None:
            n_trees = len(estimators)

        n_trees = min(n_trees, len(estimators))

        # AdaBoost decision function:
        # sum(alpha_i * h_i(x)) / sum(alpha_i)
        response = np.zeros(len(X), dtype=float)
        weight_sum = 0.0

        for estimator, weight in zip(
            estimators[:n_trees],
            estimator_weights[:n_trees]
        ):
            response += weight * estimator.predict(X)
            weight_sum += weight

        if weight_sum != 0:
            response /= weight_sum

        return response

    return (adaboost_response,)


@app.cell
def _(X_bkg_val, X_sig_val, adabdt_tf, adaboost_response):
    a1 = adaboost_response(adabdt_tf, X_sig_val,n_trees=3)
    a2 = adaboost_response(adabdt_tf, X_bkg_val,n_trees=3)
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
    ## Exercise 4: GradientBDTs
    Training a gradient BDT is similar to the AdaBoost example. However, you have learned in the lecture that the boosting works in a different way. We will explore some of the most important differences together, to get an idea why GBDTs are more popular and performant for most classification tasks.
    """)
    return


@app.cell
def _(GradientBoostingClassifier):
    def train_gradient_bdt(
        df,
        target_column,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42
    ):
        """
        Train a Gradient Boosted Decision Tree (GBDT) on a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing features and target.
        target_column : str
            Name of the target column.
        n_estimators : int
            Number of boosting trees.
        learning_rate : float
            Contribution of each tree.
        max_depth : int
            Maximum depth of each decision tree.
        min_samples_split : int
            Minimum samples required to split a node.
        min_samples_leaf : int
            Minimum samples required in a leaf.
        subsample : float
            Fraction of samples used for fitting each tree.
        random_state : int
            Random seed.

        Returns
        -------
        model : GradientBoostingClassifier
            Trained GBDT.
        X : pandas.DataFrame
            Training features.
        y : pandas.Series
            Target values.
        """

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Create the GBDT
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state
        )

        # Train
        model.fit(X, y)

        return model, X, y

    return


@app.cell
def _(GradientBoostingClassifier, log_loss, np, plt, train_test_split):
    def train_gradient_bdt_explicit(
        df,
        target_column,
        test_size=0.3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42,
        plot_loss=True
    ):
        """
        Train a Gradient Boosted Decision Tree while explicitly monitoring
        the log-loss after every boosting iteration.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing features and target.

        target_column : str
            Name of the target column.

        test_size : float
            Fraction of data used for validation.

        n_estimators : int
            Maximum number of boosting trees.

        learning_rate : float
            Learning rate of the boosting algorithm.

        max_depth : int
            Maximum depth of each decision tree.

        min_samples_split : int
            Minimum number of samples required to split a node.

        min_samples_leaf : int
            Minimum number of samples required in a leaf.

        subsample : float
            Fraction of events used for each boosting iteration.

        random_state : int
            Random seed.

        plot_loss : bool
            If True, plot training and validation loss.

        Returns
        -------
        model : GradientBoostingClassifier
            Trained model.

        history : dict
            Training history containing loss after every tree.

        X_train, X_val, y_train, y_val
            Training and validation datasets.
        """

        # ---------------------------------------------------------
        # 1. Split features and target
        # ---------------------------------------------------------

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # ---------------------------------------------------------
        # 2. Create Gradient BDT
        # ---------------------------------------------------------

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state
        )

        # ---------------------------------------------------------
        # 3. Train
        # ---------------------------------------------------------

        model.fit(X_train, y_train)

        # ---------------------------------------------------------
        # 4. Calculate loss after every tree
        # ---------------------------------------------------------

        train_loss = []
        val_loss = []

        for train_proba, val_proba in zip(
            model.staged_predict_proba(X_train),
            model.staged_predict_proba(X_val)
        ):

            train_loss.append(
                log_loss(y_train, train_proba)
            )

            val_loss.append(
                log_loss(y_val, val_proba)
            )

        history = {
            "train_loss": np.array(train_loss),
            "val_loss": np.array(val_loss),
            "n_trees": np.arange(1, n_estimators + 1)
        }

        # ---------------------------------------------------------
        # 5. Find best iteration
        # ---------------------------------------------------------

        best_iteration = np.argmin(val_loss) + 1
        best_val_loss = val_loss[best_iteration - 1]

        print(f"Best number of trees: {best_iteration}")
        print(f"Best validation loss: {best_val_loss:.5f}")

        # ---------------------------------------------------------
        # 6. Plot loss
        # ---------------------------------------------------------

        if plot_loss:

            plt.figure(figsize=(8, 6))

            plt.plot(
                history["n_trees"],
                history["train_loss"],
                label="Training",
                linewidth=2
            )

            plt.plot(
                history["n_trees"],
                history["val_loss"],
                label="Validation",
                linewidth=2
            )

            plt.axvline(
                best_iteration,
                color="black",
                linestyle="--",
                alpha=0.6,
                label=f"Best = {best_iteration} trees"
            )

            plt.xlabel("Number of trees")
            plt.ylabel("Log loss")
            plt.title("Gradient BDT training")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return model, history, X_train, X_val, y_train, y_val

    return


if __name__ == "__main__":
    app.run()
