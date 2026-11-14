# Python Data Analysis Tutorial: From Raw Data to Executive Insights

This tutorial is a complete, hands-on walkthrough of exploratory data analysis
(EDA) and data preparation in Python. It covers the full pipeline — collecting
data, cleaning it, exploring it, visualizing it, querying it with SQL, and
turning the numbers into an executive-ready story. Every code snippet here is
real, verified code drawn from an end-to-end IBM Data Analyst capstone that
analyzed the 2023 Stack Overflow Developer Survey.

It is written for two goals:

1.  **Technical fluency** — so you can reliably write Python and SQL for data
    cleaning, anomaly detection, and feature engineering.
2.  **AI fluency** — so you can use generative AI to generate and audit those
    scripts, and to translate statistical output into a business narrative.

A single dataset — the Stack Overflow Developer Survey (65,437 raw responses,
later cleaned to 18,845) — runs through every section so you can follow the same
data from raw file to final recommendation.

## Table of contents

- [The data analysis workflow](#the-data-analysis-workflow)
- [Environment setup](#environment-setup)
- [Collecting data](#collecting-data)
- [Loading data](#loading-data)
- [Inspecting and understanding the data](#inspecting-and-understanding-the-data)
- [Statistical concepts and formulas](#statistical-concepts-and-formulas)
- [Cleaning and preparing the data](#cleaning-and-preparing-the-data)
- [Exploratory data analysis (EDA)](#exploratory-data-analysis-eda)
- [Data visualization](#data-visualization)
- [SQL for data analysis](#sql-for-data-analysis)
- [Communicating findings to executives](#communicating-findings-to-executives)
- [Using generative AI to automate the pipeline](#using-generative-ai-to-automate-the-pipeline)
- [End-to-end example script](#end-to-end-example-script)
- [See also](#see-also)

## The data analysis workflow

A reliable data analysis project follows a loop, not a straight line. The
industry-standard framing is **CRISP-DM** (Cross-Industry Standard Process for
Data Mining), and it maps cleanly onto the tools in this tutorial:

1.  **Business understanding** — what question are you answering, and for whom?
2.  **Data collection** — gather data from files, APIs, databases, or web
    scraping.
3.  **Data preparation** — clean, impute, standardize, and engineer features.
4.  **Exploratory analysis** — summarize distributions, find outliers and
    correlations.
5.  **Modeling / analysis** — aggregations, comparisons, and (if needed)
    statistics or ML.
6.  **Evaluation** — do the results answer the business question?
7.  **Communication** — dashboards, charts, and an executive narrative.

The rest of this tutorial walks through stages 2 through 7 in order. Treat the
sections as a reusable pipeline template: copy the code, swap in your own
dataset, and adjust the column names.

**The running example.** Throughout, we answer a business question a hiring or
product team might ask: *"What technologies do developers use today, what do
they want next, and who are they?"* The final answer becomes three dashboard
tabs (current usage, future trends, demographics) plus a report.

## Environment setup

You can run all of this in plain Visual Studio Code — no Jupyter or Anaconda
required. The notebooks this tutorial is based on use the standard data
libraries: `pandas`, `matplotlib`, `seaborn`, `numpy`, `requests`,
`beautifulsoup4`, and `openpyxl`.

### Installing libraries from inside VS Code

Jupyter's `!pip install ...` magic only works inside a notebook kernel. In a
plain `.py` script or a VS Code "Run Python File", use `subprocess` so packages
install into the same interpreter that runs your code:

```python
import importlib.util
import subprocess
import sys

def install(package, import_name=None):
    """Quietly pip-install a package if it is not already available."""
    import_name = import_name or package
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet', package])

install('pandas', 'pandas')
install('matplotlib', 'matplotlib')
install('seaborn', 'seaborn')
install('numpy', 'numpy')
install('requests', 'requests')
install('beautifulsoup4', 'bs4')
install('openpyxl', 'openpyxl')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Why this matters:

-   `sys.executable` points at the interpreter VS Code selected, so the package
    lands in the right environment.
-   `importlib.util.find_spec` skips the install when the package is already
    present, so re-running the cell is fast and safe.

### A robust, offline-friendly data loader

A network hiccup should not break your pipeline. This loader prefers a local
cache, retries a download a few times, and fails with an actionable message:

```python
import os
import urllib.request
import urllib.error

def load_dataset(url, local_path, retries=3):
    # 1) Prefer a previously downloaded local copy (works offline)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass  # fall through to re-download
    # 2) Otherwise download with a few retries
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, local_path)
            return pd.read_csv(local_path)
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            print(f"Download attempt {attempt}/{retries} failed: {e}")
    raise RuntimeError(
        f"Could not download the dataset.\nURL: {url}\n"
        f"Last error: {last_err}\n"
        "Check your internet connection / VPN / proxy, or place a copy of "
        f"the file named '{local_path}' next to this script.")

df = load_dataset('https://example.com/survey-data.csv', 'survey-data.csv')
```

## Collecting data

Before you can analyze anything, you have to get it. The three common sources
are APIs, web pages, and files you already have.

### From a REST API

`requests` is the standard HTTP client. The pattern is always the same: build a
URL, `GET` it, check the status, and parse the JSON.

```python
import requests

# A public "echo" API — useful for learning the mechanics.
url_get = 'http://httpbin.org/get'
payload = {"name": "Joseph", "ID": "123"}

r = requests.get(url_get, params=payload)

print(r.status_code)          # 200 means success
print(r.url)                  # the fully-built URL with query params
print(r.json())               # the response body as a Python dict
print(r.json()['args'])       # the query params echoed back
```

To make it production-ready, wrap the call in error handling and parse the
response defensively:

```python
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()          # raises on 4xx/5xx
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    data = None
```

**Capstone example.** The "Jobs API" lab collected job-posting counts for seven
U.S. cities. The client fetched the full dataset and filtered locally, which is
also a good pattern: fetch once, filter in memory, and cache the result.

### From a web page (web scraping)

When data lives in an HTML table, `BeautifulSoup` plus `requests` extracts it.
Always `str.split(';')`/`explode()` is a separate concern; here the core is
finding the table and its rows/columns.

```python
from bs4 import BeautifulSoup
import requests

url = 'https://example.com/programming_languages.html'
data = requests.get(url).text
soup = BeautifulSoup(data, 'html.parser')

table = soup.find('table')          # the first <table>
rows = []
for row in table.find_all('tr'):    # <tr> = table row
    cols = row.find_all('td')       # <td> = cell
    if len(cols) >= 4 and cols[0].getText().strip().isdigit():
        language = cols[1].getText().strip()
        salary   = cols[3].getText().strip()
        rows.append([language, salary])
```

Then write the result to CSV with the standard `csv` module:

```python
import csv

with open('popular-languages.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Language', 'Average Annual Salary'])
    writer.writerows(rows)
```

### Saving and reading Excel

`openpyxl` powers pandas' Excel support. Write rows to a workbook with
`Workbook`, or use pandas directly:

```python
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws.append(['Location', 'Number of Job Postings'])
ws.append(['Washington DC', 5316])
ws.append(['Detroit', 3945])
wb.save('job-postings.xlsx')

# Read it back:
df = pd.read_excel('job-postings.xlsx')
```

## Loading data

### CSV

```python
df = pd.read_csv('survey-data.csv')
```

Common options worth knowing:

```python
df = pd.read_csv(
    'survey-data.csv',
    low_memory=False,     # avoids mixed-type warnings on wide files
    encoding='utf-8',
    parse_dates=['Date'], # convert columns to datetime
)
```

### Excel

```python
df = pd.read_excel('job-postings.xlsx')
```

### SQL databases

SQLite is a file-based database that needs no server — perfect for analysis.
You can write a DataFrame into it and query it with SQL:

```python
import sqlite3

conn = sqlite3.connect('survey-data.sqlite')
df.to_sql('main', conn, if_exists='replace', index=False)

# Query it back with SQL:
df = pd.read_sql_query("SELECT * FROM main LIMIT 5", conn)
conn.close()
```

### Inspecting what loaded

```python
df.shape          # (rows, columns)
df.columns        # column names
df.dtypes         # data type of each column
df.head()         # first 5 rows
```

## Inspecting and understanding the data

Before cleaning, understand what you have. Three calls cover most of it.

```python
# 1) Structure: column names, non-null counts, and dtypes
df.info()

# 2) Summary statistics for numeric columns
df.describe()

# 3) Missing values per column
df.isna().sum().sort_values(ascending=False)
```

What to look for:

-   **Shape vs. expectation** — does the row count match the source?
-   **Data types** — are numbers stored as `int`/`float`, or as text (`str`)?
    Text numbers (like `'2'` in a `YearsCodePro` column) must be converted
    before arithmetic.
-   **Missing values** — which columns are empty, and how much?
-   **Cardinality** — a categorical column with thousands of unique values may
    need grouping.

### Value counts for categorical columns

```python
df['Employment'].value_counts()
df['Age'].value_counts()
```

## Statistical concepts and formulas

This section is a quick reference for the statistical and mathematical terms
used throughout the tutorial. Each entry gives the definition, the formula, and
what it means in plain English. You do not need to memorize these — you need to
know which one to reach for and why.

Throughout, we use **sample** notation: $n$ = number of observations, $\bar{x}$
("x-bar") = sample mean, $s$ = sample standard deviation. The equivalent
**population** symbols are $\mu$ (mean) and $\sigma$ (standard deviation).

### Central tendency — the "typical" value

**Mean (arithmetic average).** The sum of all values divided by the count.

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

-   `df['col'].mean()`
-   *Sensitive to outliers*: one enormous salary pulls the mean upward.

**Median.** The middle value when the data is sorted (the 50th percentile).

-   If $n$ is odd, it is the middle value; if $n$ is even, it is the average of
    the two middle values.
-   `df['col'].median()`
-   *Robust to outliers*: the recommended "typical" for skewed data such as
    salaries.

**Mode.** The most frequently occurring value.

-   `df['col'].mode()[0]`
-   The *only* measure of central tendency that works for categorical data.

**Rule of thumb.** For symmetric data, mean ≈ median ≈ mode. For right-skewed
data (income, compensation), the mean sits above the median — report the
median.

### Spread (dispersion) — how spread out the data is

**Range.** The difference between the largest and smallest values:
$\max(x) - \min(x)$. Simple but dominated by a single outlier.

**Variance.** The average of the squared distances from the mean.

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

-   The sample formula divides by $n-1$ (degrees of freedom) to produce an
    unbiased estimate of the population variance.
-   `df['col'].var()`
-   Units are squared (e.g., "dollars squared"), so it is hard to interpret
    directly — that is why we usually report its square root.

**Standard deviation (std).** The square root of the variance, in the same units
as the data.

$$s = \sqrt{s^2}$$

-   `df['col'].std()`
-   Small std = data clusters tightly around the mean; large std = data is
    spread out.
-   For an approximately normal distribution, the **68–95–99.7 rule** says
    roughly 68% of values fall within $\pm 1$ std of the mean, 95% within
    $\pm 2$, and 99.7% within $\pm 3$.

**Percentiles and quartiles.** A percentile is the value below which a given
percentage of observations fall.

-   $Q_1$ = 25th percentile, $Q_2$ = median = 50th percentile,
    $Q_3$ = 75th percentile.
-   `df['col'].quantile(0.25)` and `df['col'].quantile(0.75)`

**Interquartile range (IQR).** The spread of the middle 50% of the data.

$$\text{IQR} = Q_3 - Q_1$$

-   Robust to outliers because it ignores the top and bottom 25%.

**Five-number summary.** `min, Q1, median, Q3, max` — exactly what a box plot
shows. `df.describe()` prints the count, mean, std, min, quartiles, and max.

### Shape of the distribution

**Skewness.** A measure of asymmetry.

-   **Right (positive) skew:** a long tail to the right; mean > median
    (compensation data).
-   **Left (negative) skew:** a long tail to the left; mean < median.
-   **Symmetric:** mean ≈ median.

**Normal (Gaussian) distribution.** The familiar bell curve, fully described by
its mean and standard deviation. It is the basis for z-scores and the 68–95–99.7
rule. Many real datasets are *not* normal, which is why we use robust tools like
the median and IQR.

**Kernel density estimate (KDE).** A smoothed histogram that estimates the
underlying probability density function. `sns.kdeplot(...)`.

### Standardized scores and outliers

**Z-score (standard score).** How many standard deviations a value is from the
mean.

$$z = \frac{x - \bar{x}}{s}$$

-   A z-score of +2 means "two standard deviations above the mean."
-   A common outlier rule is $|z| > 3$.

**Outlier.** An observation that lies an abnormal distance from the other
values. Two common detection methods appear in this tutorial:

-   **IQR method:** flag values below $Q_1 - 1.5 \times \text{IQR}$ or above
    $Q_3 + 1.5 \times \text{IQR}$.
-   **Z-score / standard-deviation method:** flag values more than 3 standard
    deviations from the mean (e.g., $\bar{x} + 3s$ for high outliers).

Outliers are not always errors — in salary data they are usually real and
informative. *Detect* them, then decide whether to remove, cap, or keep them.

### Scaling and transformation

**Min-max normalization.** Rescale all values to the range $[0, 1]$.

$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

-   Useful when comparing columns with different scales.

**Standardization (z-score scaling).** Rescale to a distribution with mean 0 and
standard deviation 1 (the same formula as the z-score).

**Log transformation.** Take the logarithm of each value to compress large
values and reduce right skew.

$$x' = \log(x) \quad \text{or} \quad x' = \log(1 + x)$$

-   `np.log1p(x)` applies $\log(1 + x)$, which safely handles zero.

### Relationships between variables

**Correlation coefficient (Pearson's $r$).** A number from $-1$ to $+1$ that
measures the strength and direction of a *linear* relationship.

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}$$

-   $+1$ = perfect positive, $0$ = no linear relationship, $-1$ = perfect
    negative.
-   Rough guide to strength: $|r| \approx 0.3$ weak, $\approx 0.5$ moderate,
    $\approx 0.7$ strong.
-   `df[['x', 'y']].corr()` (Pearson is the default).
-   **Correlation is not causation** — two variables moving together does not
    mean one causes the other.

**Spearman rank correlation ($\rho$).** Pearson's $r$ computed on the *ranks*
rather than the raw values. It captures any *monotonic* relationship (not just
linear) and is robust to outliers.

-   `df[['x', 'y']].corr(method='spearman')`

### Which statistic to use

| You want to know…                    | Use                               |
| ------------------------------------ | --------------------------------- |
| The typical value (skewed data)      | Median                            |
| The typical value (symmetric data)   | Mean                              |
| The most common category             | Mode                              |
| How spread out the data is           | Standard deviation or IQR         |
| The spread of the middle half        | IQR                               |
| Whether a value is extreme           | Outlier rules (IQR or z-score)    |
| Whether two variables move together  | Correlation (Pearson/Spearman)    |
| Shape of the distribution            | Histogram, KDE, skewness          |

## Cleaning and preparing the data

This is where most of the real work happens. A clean dataset is the difference
between a trustworthy result and a misleading one.

### Finding and removing duplicates

Duplicates distort counts and averages. Identify them, inspect them, then
remove them — but first decide what "duplicate" means for your data.

```python
# Count exact duplicate rows
df.duplicated().sum()

# Inspect them (keep=False marks every copy, not just the later one)
df[df.duplicated(keep=False)].sort_values('ResponseId').head()

# Remove exact duplicates (keeps the first occurrence)
df = df.drop_duplicates()
```

**The subtle trap.** A duplicate *row* is not the same as a *shared answer
pattern*. In the survey, 65,290 of 65,437 respondents shared the same
`MainBranch` / `Employment` / `RemoteWork` combination — but they are different
people. Deduplicating on those three columns would collapse the dataset to a few
hundred rows. The rule: deduplicate on the record's identity (e.g., a unique
`ResponseId` or the full row), never on a small subset of answer columns.

To see whether a column identifies unique records:

```python
df['ResponseId'].nunique()   # == len(df) means it is a unique key
```

### Finding and imputing missing values

Quantify first:

```python
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
print(missing)
```

Three standard strategies, chosen by column type and context:

```python
# Categorical -> most frequent value (mode)
df['RemoteWork'] = df['RemoteWork'].fillna(df['RemoteWork'].mode()[0])

# Numeric -> median (robust to outliers; mean is sensitive)
df['JobSat'] = df['JobSat'].fillna(df['JobSat'].median())

# Ordered / time-like -> forward fill (carry the last valid value)
df['CodingActivities'] = df['CodingActivities'].ffill()
```

When to drop instead of impute:

-   A column is almost entirely empty (> ~90%) — it carries little signal.
-   The analysis *requires* the value (e.g., you cannot analyze compensation for
    respondents who did not report it). For that, drop just those rows:

```python
comp_df = df.dropna(subset=['ConvertedCompYearly']).copy()
```

### Standardizing inconsistent values

Categorical columns often contain the same concept written differently
(long-form country names, mixed casing). Map them to one canonical form:

```python
country_mapping = {
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'United States of America': 'United States',
}
df['Country'] = df['Country'].replace(country_mapping)
```

Always print the value counts before and after so the change is visible and
reversible.

### Detecting outliers (anomaly detection)

Outliers are legitimate-but-extreme values. Two common, explainable methods:

**IQR (interquartile range)** — robust to skew, the workhorse for skewed data
like compensation:

```python
col = 'ConvertedCompYearly'
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
print(f"Outliers: {len(outliers)}")

# A cleaned copy for correlation analysis:
df_no_outliers = df[
    (df[col] >= lower_bound) & (df[col] <= upper_bound)].copy()
```

**Standard-deviation threshold** — assumes an approximately normal
distribution; useful for spotting extreme highs:

```python
threshold = df[col].mean() + 3 * df[col].std()
high_outliers = df[df[col] > threshold]
```

Decide *before* removing: are these errors (remove), or real signal (keep and
note)? Compensation outliers are usually real — flag them, don't silently
delete them, and re-run key metrics with and without them.

### Normalizing and scaling

Scaling puts numeric columns on a common range, which matters for distance-based
models and for comparing columns with different units.

```python
col = 'ConvertedCompYearly'

# Min-Max scaling -> [0, 1]
df['MinMax'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Z-score (standardization) -> mean 0, std 1
df['Zscore'] = (df[col] - df[col].mean()) / df[col].std()

# Log transform -> reduces right-skew
df['Log'] = np.log1p(df[col])   # log1p handles zeros safely
```

### Feature engineering

Create new columns that capture meaning the raw columns don't. A common one is
binning a continuous or messy column into levels:

```python
def categorize_experience(years):
    if pd.isna(years):
        return 'Unknown'
    if isinstance(years, str):            # handle text values like
        if years == 'Less than 1 year':   # "Less than 1 year"
            return 'Junior'
        if years == 'More than 50 years':
            return 'Senior'
        years = float(years)
    years = float(years)
    if years < 5:
        return 'Junior'
    elif years < 10:
        return 'Mid-level'
    else:
        return 'Senior'

df['ExperienceLevel'] = df['YearsCodePro'].apply(categorize_experience)
```

Binning continuous values into ranges:

```python
df['AgeGroup'] = pd.cut(
    df['AgeNum'],
    bins=[0, 25, 35, 45, 100],
    labels=['<25', '25-34', '35-44', '45+'],
)
```

**One-hot encoding** for categorical variables (converts each category into a
0/1 column):

```python
df = pd.concat(
    [df, pd.get_dummies(df['Employment'], prefix='Employment')], axis=1)
```

**Splitting multi-value cells.** Survey columns often hold semicolon-separated
lists (`'Python;SQL;JavaScript'`). Split and explode to count each item:

```python
lang_counts = (
    df['LanguageHaveWorkedWith']
    .dropna()
    .str.split(';')
    .explode()
    .value_counts()
)
```

## Exploratory data analysis (EDA)

EDA is "look at the data before you model it." It surfaces what you should dig
into further.

### Distributions

```python
df['CompTotal'].plot(kind='hist', bins=50)
plt.show()

sns.kdeplot(df['CompTotal'].dropna(), fill=True)
plt.show()
```

### Correlation

Correlation measures linear association from -1 to +1. Use Pearson for linear
relationships and Spearman for monotonic (rank-based) ones:

```python
# Pearson correlation matrix
df[['ConvertedCompYearly', 'WorkExp', 'JobSatPoints_1']].corr()

# Spearman (rank) correlation
df[['YearsCodePro_num', 'JobSat']].corr(method='spearman')

# Visualize the matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
```

Interpret with caution: correlation is not causation, and a weak correlation
(like 0.07 between experience and satisfaction) is still a real *finding* — the
point is that the two are largely independent.

### Grouping and aggregation

The heart of most business questions is "how does X vary by Y?".

```python
# Median compensation by age group
df.groupby('Age')['ConvertedCompYearly'].median()

# Multiple aggregates at once
df.groupby('Employment')[['JobSatPoints_6', 'JobSatPoints_7']].median()

# Count by category
df['Country'].value_counts().head(10)
```

## Data visualization

A chart answers one of four questions. Pick the chart to match the question:

| Question                 | Best charts                                        |
| ------------------------ | -------------------------------------------------- |
| How is one variable distributed? | Histogram, box plot, density (KDE)          |
| How are two variables related?  | Scatter plot, bubble plot, line chart        |
| What is the composition?        | Pie chart, stacked bar chart                  |
| How do groups compare?          | Bar chart, grouped/stacked bar chart          |

### Histogram (distribution)

```python
plt.figure(figsize=(10, 5))
df['CompTotal'].dropna().plot(
    kind='hist', bins=50, color='steelblue', edgecolor='black')
plt.title('Distribution of Total Compensation')
plt.xlabel('CompTotal')
plt.ylabel('Number of respondents')
plt.tight_layout()
plt.show()
```

### Box plot (distribution + outliers)

```python
plt.figure(figsize=(8, 5))
plt.boxplot(df['AgeNum'].dropna(), vert=True)
plt.title('Box Plot of Age')
plt.ylabel('Age (years)')
plt.tight_layout()
plt.show()
```

Grouped box plots compare a distribution across categories:

```python
df.boxplot(column='ConvertedCompYearly', by='DevType', rot=45)
plt.suptitle('')
plt.show()
```

### Scatter plot (relationship)

```python
plt.scatter(df['AgeNum'], df['WorkExp'], alpha=0.3, s=10)
plt.xlabel('Age (years)')
plt.ylabel('Work Experience (years)')
plt.show()
```

Add a trend line with `numpy.polyfit`:

```python
m, b = np.polyfit(x, y, 1)
plt.plot(sorted(x), [m * xi + b for xi in sorted(x)], color='red')
```

### Bubble plot (three variables: x, y, size)

```python
plt.scatter(
    df['ConvertedCompYearly'], df['JobSatPoints_6'],
    s=df['AgeNum'] * 3,          # bubble size = age
    alpha=0.3, c='steelblue', edgecolors='none')
plt.xlabel('ConvertedCompYearly')
plt.ylabel('JobSatPoints_6')
plt.show()
```

### Pie chart (composition)

```python
db_counts = (
    df['DatabaseWantToWorkWith'].dropna().str.split(';').explode()
    .value_counts().head(5)
)
db_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.ylabel('')
plt.show()
```

Pie charts are best for a small number of categories; beyond ~5 slices switch to
a bar chart.

### Stacked bar chart (composition across groups)

```python
medians = df.groupby('Age')[['JobSatPoints_6', 'JobSatPoints_7']].median()
medians.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.show()
```

### Line chart (trend over an ordered axis)

```python
df.groupby('Age')['ConvertedCompYearly'].median().plot(
    kind='line', marker='o')
plt.show()
```

### Bar chart (comparison)

```python
# Vertical
df['LanguageWantToWorkWith'].dropna().str.split(';').explode() \
    .value_counts().head(5).plot(kind='bar', color='darkorange')
plt.show()

# Horizontal (good for long labels)
df['MainBranch'].value_counts().plot(kind='barh', color='steelblue')
plt.show()
```

### The `reset_index` fix for `explode`

`DataFrame.explode()` keeps the original (duplicated) index, which breaks
`pd.crosstab`. Add `reset_index(drop=True)` after exploding:

```python
exploded = df.explode('Language').reset_index(drop=True)
ct = pd.crosstab(exploded['Age'], exploded['Language'])
```

## SQL for data analysis

When your data lives in a relational database, use SQL — it is declarative,
portable, and often faster than pandas for filtering and grouping large tables.
This tutorial uses SQLite (no server needed).

### Setup: DataFrame to SQLite

```python
import sqlite3

conn = sqlite3.connect('survey-data.sqlite')
df.to_sql('main', conn, if_exists='replace', index=False)
```

### Core queries

```sql
-- Count rows
SELECT COUNT(*) FROM main;

-- Select a few rows
SELECT * FROM main LIMIT 5;

-- List tables
SELECT name AS Table_Name
FROM sqlite_master
WHERE type = 'table';

-- Group by a column and count
SELECT Age, COUNT(*) AS count
FROM main
GROUP BY Age
ORDER BY Age;

-- Show the schema of a table
SELECT sql FROM sqlite_master WHERE name = 'main';
```

### Filtering and aggregation in SQL

```sql
-- Filter
SELECT * FROM main
WHERE Age IN ('25-34 years old', '35-44 years old');

-- Median is NOT built into SQLite; approximate with percentile if available,
-- or compute in pandas after reading the filtered rows.
```

The practical pattern: use SQL to *extract and pre-aggregate*, then pandas to
*visualize and analyze*:

```python
# SQL for extraction
QUERY = """
SELECT Age, CompTotal
FROM main
WHERE CompTotal IS NOT NULL
"""
age_comp_df = pd.read_sql_query(QUERY, conn)

# pandas for analysis + plotting
median = age_comp_df.groupby('Age')['CompTotal'].median()
median.plot(kind='bar')
```

Always close the connection when done:

```python
conn.close()
```

## Communicating findings to executives

Executives don't read tables; they read a story. Every analysis should end with
three things: **findings** (what the data says), **implications** (why it
matters), and **recommendations** (what to do). A useful template:

### Findings → Implications → Recommendations

For each insight, run it through this chain:

| Step           | Question                                | Example (from the capstone) |
| -------------- | --------------------------------------- | --------------------------- |
| **Finding**    | What does the data show?                | PostgreSQL is the #1 used (11,514) and #1 desired (12,193) database. |
| **Implication**| Why does it matter?                     | PostgreSQL skills are the safest long-term investment. |
| **Recommendation** | What should we do?                  | Prioritize PostgreSQL in hiring and training; expect real-time systems (Redis) to grow. |

### Writing an executive summary

A strong executive summary answers five questions in order:

1.  **What did you analyze?** — "The 2023 Stack Overflow Developer Survey."
2.  **What are the key findings?** — "D.C. leads job postings; Swift tops
    salary; JavaScript and SQL lead languages; PostgreSQL leads databases."
3.  **What does it mean?** — "Pay rises with age and experience; demand is
    shifting toward TypeScript and Redis."
4.  **What should we do?** — "Target JavaScript/SQL skills; benchmark Swift and
    backend pay; invest in PostgreSQL training."
5.  **How can we monitor it?** — "Three dashboards track usage, trends, and
    demographics."

Keep each bullet to one line and lead with the number, not the method.

### Quantifying the story

Put concrete numbers in the narrative so it is checkable:

-   "Washington D.C. has the most job postings (5,316)."
-   "Swift commands the highest average salary ($130,801)."
-   "JavaScript stays #1 in both current (14,943) and future (11,541) usage."

## Using generative AI to automate the pipeline

This section is the bridge between "knowing Python" and "shipping analyses
fast." The goal is reliable, auditable code — not code you paste and trust.

### Prompting for reliable code

A good prompt gives the model context, constraints, and a way to verify. This
template works well:

```
You are a senior data analyst writing Python (pandas) for a survey dataset.

Columns include: Age (categorical strings), ConvertedCompYearly (numeric,
right-skewed, many NaN), Employment (semicolon-separated multi-select),
LanguageHaveWorkedWith (semicolon-separated).

Task: [what to compute, e.g., "median ConvertedCompYearly by Age group"].

Constraints:
- Use only pandas, numpy, matplotlib (no sklearn).
- Handle NaN explicitly (state the imputation or drop strategy).
- Handle the semicolon-separated columns with str.split(';').explode().
- Add reset_index(drop=True) after explode().
- Return a runnable script, plus a one-line comment on each step.

Then verify: [e.g., "the output should have 8 rows, one per age group"].
```

Key prompt ingredients:

-   **Schema** — paste `df.dtypes` and a couple of `value_counts()` samples so
    the model knows types and cardinalsity.
-   **Edge cases** — NaN, mixed types, multi-value cells, skewed data.
-   **Constraints** — libraries allowed, must-run-as-script, no `!pip` magic.
-   **Verification** — an expected shape or value the model (and you) can check.

### Review checklist for AI-generated code

Never ship AI code without checking these:

1.  **Runs end-to-end** — execute it; a syntax error is a red flag to inspect.
2.  **Handles NaN** — does it `dropna()` or `fillna()` before `mean()`/`corr()`?
3.  **Mixed types** — are `'2'` and `2` and `'Less than 1 year'` all converted?
4.  **Multi-value cells** — are `';'` columns split and exploded correctly?
5.  **Duplicate logic** — is it deduplicating on a unique key, not a subset?
6.  **Outlier policy** — does it remove, flag, or keep extremes — and is that
    the right call?
7.  **Determinism** — same input gives the same output (seed any randomness).

### Translating stats into narrative with AI

The same model can turn output into prose. Prompt it to *derive*, not invent:

```
Here is the output of my analysis (paste the table/figure).

Write a 3-bullet executive summary in this format:
- Finding: <one sentence, with the specific number>
- Implication: <one sentence on why it matters>
- Recommendation: <one sentence on what to do>

Rules: no jargon, no hedging, cite only numbers present in the output.
```

Always verify the AI's numbers against the actual output — a model will
confidently hallucinate a figure. The workflow is: *generate, verify, then
trust.*

## End-to-end example script

A complete, minimal pipeline that ties every stage together:

```python
import importlib.util, subprocess, sys, os
import urllib.request, urllib.error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 0. Install (if needed) ----
def install(package, import_name=None):
    import_name = import_name or package
    try:
        importlib.util.find_spec(import_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet', package])

install('pandas', 'pandas')
install('matplotlib', 'matplotlib')

# ---- 1. Load (cached) ----
url = 'https://example.com/survey-data.csv'
local = 'survey-data.csv'
if not (os.path.exists(local) and os.path.getsize(local) > 0):
    urllib.request.urlretrieve(url, local)
df = pd.read_csv(local, low_memory=False)

# ---- 2. Clean ----
df = df.drop_duplicates()                                     # duplicates
df['RemoteWork'] = df['RemoteWork'].fillna(                   # missing (mode)
    df['RemoteWork'].mode()[0])
df['JobSat'] = df['JobSat'].fillna(df['JobSat'].median())     # missing (median)

# ---- 3. Detect outliers (IQR) ----
col = 'ConvertedCompYearly'
Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)].copy()

# ---- 4. Feature engineering ----
df_clean['AgeNum'] = df_clean['Age'].map({
    'Under 18 years old': 17, '18-24 years old': 21,
    '25-34 years old': 29.5, '35-44 years old': 39.5,
    '45-54 years old': 49.5, '55-64 years old': 59.5,
    '65 years or older': 70,
})

# ---- 5. EDA ----
median_by_age = df_clean.groupby('Age')['ConvertedCompYearly'].median()
print(median_by_age)
print(df_clean[['AgeNum', 'JobSat']].corr())

# ---- 6. Visualize ----
median_by_age.plot(kind='line', marker='o')
plt.title('Median Compensation by Age')
plt.show()
```

## See also

- [pandas documentation](https://pandas.pydata.org/docs/)
- [matplotlib examples](https://matplotlib.org/stable/gallery/index.html)
- [seaborn tutorial](https://seaborn.pydata.org/tutorial.html)
- [Requests documentation](https://requests.readthedocs.io/)
- [Beautiful Soup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [SQLite documentation](https://www.sqlite.org/docs.html)
- [CRISP-DM overview](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
