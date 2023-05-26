# Regression

Regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables. The aim of regression analysis is to find a mathematical equation that can predict the value of the dependent variable based on the values of the independent variables.

## Linear Regression

Linear regression is a type of regression analysis where the relationship between the dependent variable and the independent variable(s) is assumed to be linear. The linear regression model assumes that the dependent variable y is a linear function of one or more independent variables x, plus some random error ε:

$$y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n + ε$$

where $w_0$ is the intercept or constant term, $w_1$, $w_2$, ..., $w_n$ are the coefficients or slopes corresponding to the independent variables $x_1$, $x_2$, ..., $x_n$, and ε represents the random error term.

The goal of linear regression is to estimate the values of the coefficients $w_0$, $w_1$, $w_2$, ..., $w_n$ that minimize the sum of squared errors between the predicted values of y and the actual values of y from the training data. In other words, we want to find the line that best fits the data.

To estimate the coefficients, we use a technique called ordinary least squares (OLS) regression. OLS regression finds the values of the coefficients that minimize the sum of squared errors or `Cost Function`:

   $$E=Σ(y - w_0 - w_1x_1 - w_2x_2 - ... - w_nx_n)^2$$

We can find the values of the coefficients that minimize this expression using calculus. The resulting equations for the coefficients of simple singular variable function are:

$$w_1 = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)^2$$
$$w_0 = ȳ - w_1x̄$$

where $x_i$ and $y_i$ are the values of the independent and dependent variables, respectively, x̄ and ȳ are the mean values of the independent and dependent variables, respectively.

Once we have estimated the coefficients, we can use them to predict the value of the dependent variable y for new values of the independent variable(s) x.

## Nonlinear Regression
We learnt how a linear function would be use in regression to model our data, but not for all of datasets, using a linear regression is proper choice. there is other kind of regressions that called Non-linear Regression. the difference is instead of using $y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n + ε$ as a model function, we use non linear function such as:$$y = w_0 + w_1x^1 + w_2x^2 + ... + w_nx^n$$
to find best relation between x and y. The `Cost Function` respectivly would be:
   $$E=Σ(y - w_0 - w_1x^1 - w_2x^2 - ... - w_nx^n)^2$$
like linear regression, our goal is to minimize cost to fit our model properly.
The interactive panel below allows students to explore nonlinear regression for the function: $$y = w_1x^2 + w_2x + b$$

### Interactive Panel
Students can adjust the value of n (Degree) using slider and see the resulting regression function and R-squared value in real-time. The R-squared value is a measure of how well the fitted model explains the variability in the data, with values closer to 1 indicating a better fit.

To use the interactive panel, students can adjust the slider `degree` and observe how the regression function changes to fit the data. They can also observe how the R-squared value changes as they adjust the
