# statistical_test
----------------------------------
Statistical testing forms the backbone of Machine Learning. In this project, the focus is on applying various statistical testing concept to Exploratory Data Analysis (EDA). The aim here is to have various EDA-useful statistical test in one place with easy access options to switch between tests, visual the test results and inform users of statistical signicance of the test.

The library focuses on 2 aspects of EDA statistical testing and their interpretation:
1. Normality test:
  * shapiro
  * d' agnostino
  * anderson-darling
2. Correlation test:
  * Between continuous variables: pearson, spearman and kendalltau
  * Between categorical variables: cramers v
  * Between a categorical and continuous variable: pointbiserial (shortcut for pearson)

While the library includes simple conditional checks such as checks for continuous data, checks for NA, the checks for assumptions of each test are not provided. Users are encouraged to test for the statistical tests assumptions before applything them, otherwise naive interpretation of p-value and subsequently significance level might product misleading results. 



