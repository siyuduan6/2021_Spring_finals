import pandas as pd
import seaborn as sns
from statsmodels.api import OLS
import numpy as np
import matplotlib.pyplot as plt


# GV_rate = the number of GV cases / the population of the region /1000)
# the number of case per 1000 people
# edu = the percentage of people over 25 years old with high school or higher degree
# pov = the percentage of people who are under poverty level
# un = the percentage of people over 16 years old who are unemployed
def zip_split(file):
    """
    Split the zip code from address.
    :param file: dataframe containing variables
    :return: file with new column:zip

    >>> file1 = pd.DataFrame({"Geographic Area Name": ["ZCTA5 35004","ZCTA5 35005","ZCTA5 35006"]})
    >>> print(zip_split(file1))
      Geographic Area Name    zip
    0          ZCTA5 35004  35004
    1          ZCTA5 35005  35005
    2          ZCTA5 35006  35006

    """
    file["zip"] = file["Geographic Area Name"].str.split(" ", expand=True)[1].astype("str")
    return file

def linear(X, y):
    """
    Multi-Linear Regression
    :param X: x variables
    :param y: y variable
    :return: linear regression model
    >>> db = pd.read_csv("Variables.csv")# doctest:+ELLIPSIS
    >>> pov = db["poverty"] / 100# doctest:+ELLIPSIS
    >>> un = db["unemployment"] / 100# doctest:+ELLIPSIS
    >>> edu = db["High school or higher"] / 100# doctest:+ELLIPSIS
    >>> xv1 = pd.DataFrame([edu, pov, un]).T# doctest:+ELLIPSIS
    >>> yv = pd.DataFrame(np.log10(db["rate"]))# doctest:+ELLIPSIS
    >>> linear(xv1, yv) # doctest:+ELLIPSIS
                                     OLS Regression Result...
    <statsmodels.regression.linear_model.RegressionResultsWrapper object at ...>

    """

    ols = OLS(y,X).fit()
    print(ols.summary())
    return ols

def plot_correlation(x):
    """
    Plot correlation matrix
    :param x: dataframe containing x and y variables
    :return: correlation matrix
    >>> db = pd.read_csv("Variables.csv")
    >>> pov = db["poverty"] / 100
    >>> un = db["unemployment"] / 100
    >>> edu = db["High school or higher"] / 100
    >>> edu2 = db["Bachelor or higher"] / 100
    >>> xv1 = pd.DataFrame([edu, pov, un]).T
    >>> yv = pd.DataFrame(np.log10(db["rate"]))
    >>> dba = xv1.merge(yv, left_index=True, right_index=True)
    >>> plot_correlation(dba)
                           High school or higher   poverty  unemployment      rate
    High school or higher               1.000000 -0.699980     -0.557392 -0.314174
    poverty                            -0.699980  1.000000      0.625390  0.441839
    unemployment                       -0.557392  0.625390      1.000000  0.291978
    rate                               -0.314174  0.441839      0.291978  1.000000
    """
    corr = x.corr(method="spearman")
    plt.matshow(corr)
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    return corr



if __name__ == '__main__':
    # Multi Linear Regression
    db = pd.read_csv("Variables.csv")
    pov = db["poverty"] / 100
    un = db["unemployment"] / 100
    edu = db["High school or higher"] / 100
    edu2 = db["Bachelor or higher"] / 100
    xv1 = pd.DataFrame([edu, pov, un]).T
    xv2 = pd.DataFrame([edu2, pov, un]).T
    xv3 = pd.DataFrame([edu, edu2, pov, un]).T
    yv = pd.DataFrame(np.log10(db["rate"]))

    model1 = linear(xv1, yv)
    model2 = linear(xv2, yv)
    dba = xv3.merge(yv, left_index=True, right_index=True)
    # Plot correlation metrix and pairwise metrix
    plot_correlation(dba)
    sns.pairplot(dba)

    # Further Analysis
    # import other variables age and sex ratio
    age_file = pd.read_csv("age-2017.csv", header=1, low_memory=False)
    age_new = zip_split(age_file)[
        ["zip", "Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over",
         "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over"]].rename(
        columns={"Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over": "age",
                 "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over": "sex"})
    age_new["zip"] = age_new["zip"].astype("float64")
    db5 = db.merge(age_new, on="zip")
    db5["age"] = db5["age"].astype("float64")
    db5["sex"] = db5["sex"].astype("float64")
    describe = db5.describe()
    print(describe)
    db = db5
    un = db["unemployment"] / 100
    pov = db["poverty"] / 100
    edu2 = db["Bachelor or higher"] / 100
    age = db["age"].astype("float64") / 100
    sex = db["sex"].astype("float64") / 100
    xv = pd.DataFrame([edu2, pov, un, sex, age]).T
    yv = pd.DataFrame(np.log10(db["rate"]))
    model3 = linear(xv, yv)

    dba2 = xv.merge(yv, right_index=True, left_index=True)
    plot_correlation(dba2)
    sns.pairplot(dba2)



