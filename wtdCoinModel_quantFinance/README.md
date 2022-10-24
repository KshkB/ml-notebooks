# Weighted coin model

Open the Jupyter notebook `tutorial_notebook.ipynb`, fill in according to directives provided, uncomment and run the code cells. Doing so generates a buy or sell prediction *for today*.

## Important preliminary information

### The API

The python scripts in this repository call data from the IEX Cloud API. In order to do this you will need to obtain valid API tokens. Once these are obtained, store them in `secrets.py` by filling in: 

- `API_TOKEN`;
- `API_TOKEN_SANDBOX`.

The `API_TOKEN_SANDBOX` token operates in the sandbox environment, so data called here is *not* real world data. The `API_TOKEN` token calls real world data.

**Note.** *Sandbox tokens are free*.

### Default settings 

By default, code in this package is set to run in the sandbox environment. To use real world data there are two code changes to make: in `dataprep.py` and `predict.py`.

In `dataprep.py`:

- comment out the two lines following `# SANDBOX TESTING`;
- in the class object `dataprep.PriceData` replace `token = API_TOKEN_SANDBOX` with `token = API_TOKEN`.

In `predict.py`:

- comment out the two lines following `# SANDBOX TESTING`.

### Required packages

In order to run the python scripts in this repository you will need the libraries:

- `scikit-learn`;
- `iexfinance`, pip install link [here](https://pypi.org/project/iexfinance/);
- `pandas` and `numpy`

# Methodology

Retrospectively, given price history, it is easy to determine whether we should have bought or sold a unit of a security $x$ over an interval $I$. If $p_0, p_1$ refer to the price of a security $x$ at the start and end of $I$ respectively, then:

- if $p_0 < p_1$, buy at $p_0$;
- if $p_0> p_1$, sell at $p_0$.

Now at the start of the interval $I$, there is a probabilty for each of the two scenarios above to eventuate. Let $B$ be the probability $p_0 < p_1$ and $S$ the probability $p_0 > p_1$. The combined probability is $B + S = 1$. 

Imagine therefore that at the start of any interval $I$ that we flip a weighted coin to decide whether to buy or sell a unit of $x$. The sides of this coin are `BUY` and `SELL` with weightings $B$ and $S$ respectively.

## Parameters for prediction

### Weighted coin probabilities

Maximum likelihood is the means by which we can decide whether to buy or sell at the outset of an interval. Any time interval $I$ is determined by its size, which is a specified number of days. Given a historical time *period*, $\mathcal T$, such as a number of months or years, it can be divided into intervals

$$\mathcal{T} = \bigcup_j T_j$$

where $T_j$ is an inteval $I$ comprised of a fixed number of days. As this is historical we know at the start of each $T_j$ whether or not to buy or sell. Counting the number of times to buy and the number of times to sell yields the weighted coin probabilities $B$ and $S$ *with respect to* $(\mathcal{T}, I)$.

### In machine learning 

With an extra parameter it is possible to use historical data to form training data with labels. These can then be used to train a machine learning model to make predictions based on maximum likelihood. Recall that the parameters we have currently are the time period $\mathcal T$ and generic interval $I$. This extra parameter, denoted $\delta$, is refered to as *depth*. It represents the following thesis: *knowing* $(\delta-1)$*-many day closing price movements influences the price movement on the* $\delta$*-th day*.

## Implementation

### Training and labels

A time period $\mathcal T$ is specified by starting and ending dates, $d_s, d_e$. Between these dates generate a list of dates $d_0, d_1, \ldots$ where $d_j - d_{j-1} = |I|$ for all $j$. That is, the difference in days between successive dates differ by the number of days defining the generic interval $I$. With the depth parameter now, partition this list of dates into almost totally overlapping sequences of $\delta$-many dates, yielding a list of lists $(d_0, d_1,\ldots, d_{\delta-1}), (d_1, d_2,\ldots, d_{\delta}), \ldots$ up until the last date. For machine learning purposes, each sequence here will define a feature vector $v$ and feature vector label $y(v)$ in the following way. 

For a sequence $(d_{j}, \ldots, d_{j + \delta})$ let $(p_j, \ldots, p_{j + \delta})$ denote the closing prices, so $p_j$ is the closing price of the security at the end of date $d_j$. With the prices then we can form the feature vector 

$$v_{j}=\left(p_{j+1}/p_j - 1, \ldots, p_{j+\delta-1}/p_{j+\delta-2} - 1\right)$$

with label

$$y(v_{j})=\frac{p_{j+\delta}/p_{j+\delta-1} - 1}{\|p_{j+\delta}/p_{j+\delta-1} - 1\|}.$$

The vector $v_j$ comprises price rate of change between successive dates. The label $y(v_j)$ is the rate of change on the last day, normalized to be valued in $\mathbb Z_2 = \{\pm 1\}$. If $y(v_j) = 1$ then *buy* on date $d_{j + \delta - 1}$; if $y(v_j) = -1$ then *sell* on date $d_{j + \delta - 1}$.

### Testing

With the training data comprising the features and labels have been formed and a model trained, it is applied to direct real world action as follows. At present we imagine to be at the starting date of an interval $I$. That is, the *present date* is the last date of a sequence of dates of length $\delta$. With $d_P$ the date at present, we want to know whether to buy or sell a unit of the security of interest, $x$. And so, over the period of $|I|$ days $d_P$ to $d_{P+|I|}$, will the price appreciate or depreciate?

Pass into the model a sequence $(d_{P-\delta}, d_{P-\delta + 1}, \ldots, d_P)$. The price is known for each of these dates. Based on this test feature, the model will generate a label $y\in \mathbb Z_2 = \{\pm 1\}$. 

This label is a signal to buy or sell at present. The rest is up to you!
