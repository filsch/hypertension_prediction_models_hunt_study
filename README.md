Hypertension risk models developed using the HUNT Study data
================
Filip Schjerven
2022-10-27

Dissemination and external validation of hypertension risk models is
important, but rarely accommodated for in the literature. This is
especially true when machine learning models are being used, as they are
often too complex to describe in traditional appendices. We used machine
learning to predict risk of hypertension 11 years later on data from
23722 HUNT Study participants, and provide the machine learning models
developed using the XGBoost, Random Forest, and Elastic regression
methods \[[1](#ref-åsvold2022)\] \[[2](#ref-chen2021)\]
\[[3](#ref-breiman2001)\] \[[4](#ref-zou2005)\] . See
\[[5](#ref-schjerven2022)\] for details on how these models were fitted.

In addition, we include three additional models:

- A smaller logistic regression model developed as part of a sensitivity
  analysis on the HUNT Study data. This model utilizes only a few, more
  easily available features,
- An adaptation of the externally developed Framingham risk model, and
- An adaptation of the Framingham risk model recalibrated on the HUNT
  Study data \[[6](#ref-RN10737)\] .

Two more modelling methods were reported in our study: The K-Nearest
Neighbour and the SVM models. These are not provided here, as they could
not be detached from the development data which we do not have
permission to share. See \[[5](#ref-schjerven2022)\] for more info and
how to apply for access to these data.

#### Models, preprocessings and helper functions

``` r
load("models.rds")
load("prep.rds")
source("helper_functions.r")
```

#### Data

We include some example data to illustrate the input data for the
different models. All variables and levels are represented. See
\[[5](#ref-schjerven2022)\] for information on variables and how they
were recorded and constructed. Note that the example data is not real
data and is only included to demonstrate how the resources in this
repository may be used.

``` r
load("example_data.rds")
```

##### Table 1: Variables needed as input to the XGBoost, Random Forest and Elastic regression models:

| Variable      | Type           | Levels                                                          |
|---------------|----------------|-----------------------------------------------------------------|
| Sex           | factor         | female, male                                                    |
| Height        | numeric        | \-                                                              |
| BMI           | numeric        | \-                                                              |
| SysBP         | numeric        | \-                                                              |
| DiaBP         | numeric        | \-                                                              |
| SeChol        | numeric        | \-                                                              |
| SeHDLChol     | numeric        | \-                                                              |
| SeTrig        | numeric        | \-                                                              |
| SeGluNonFast  | numeric        | \-                                                              |
| SeCreaCorr    | numeric        | \-                                                              |
| GFREstStag    | factor         | stage 1, stage 2-5                                              |
| Age           | numeric        | \-                                                              |
| FamCVDHist    | factor         | no, yes                                                         |
| SmokingStatus | factor         | never, formerly, currently                                      |
| Education     | factor         | secondary, upper_secondary, high_school, college, post_graduate |
| FamHypHist    | factor         | no, yes                                                         |
| HypOutcome    | ordered factor | Normotensive \< Hypertensive                                    |
| PhysAct       | factor         | low, mid, high                                                  |

#### Preprocessing and predictions

The ‘prep’ object found in ‘prep.rds’ contains a pre-processing function
that standardizes numerical variables and one-hot-encodes factors so
they match model input requirements for the machine learning models. The
parameters of the ‘prep’ variable has been learned from the training
data used to develop the models. Note, we assume that that the data
should be complete, i.e., no missing entries, and with similar names and
format as the example data.

``` r
prepped_example_data <- bake(prep, example_data)

model <- models$xgb #xgb, rf or elastic

y_hat <- predict(model, newdata=prepped_example_data, type="prob")
head(y_hat, n=2)
```

    ##   Normotensive Hypertensive
    ## 1    0.6148387    0.3851613
    ## 2    0.3835679    0.6164321

#### Model using simple-to-collect variables

We include a logistic regression model that uses 6 more easily-available
variables. These are: *Age*, *systolic blood pressure*, *diastolic blood
pressure*, *BMI*, *height*, and *history of hypertension in close
family*. This model was derived from a sensitivity analysis performed
after development of the full models. The corresponding regularization
penalty applied in model fitting was log( lambda ) = -4.384, and gave
performance shown in Table 2. See Figure 3 in the development article
\[[5](#ref-schjerven2022)\].

##### Table 2: Performance of model using simple-to-collect variables

| AUC                    | Brier Score            | ICI                    |
|------------------------|------------------------|------------------------|
| 0.796 \[0.786, 0.806\] | 0.151 \[0.146, 0.155\] | 0.033 \[0.025, 0.041\] |

Reported as mean with 95% confidence interval after bootstrapping on the
test set.

``` r
source("auxiliary_models.r")

# Explicit example of needed input
prepped_example_data <- bake(prep, example_data)
prepped_example_subset <- prepped_example_data[,c("Age", "SysBP", "DiaBP", "BMI", "FamHypHist_yes", "Height", "HypOutcome")]

# Predictions
y_hat <- predict(simpler_model, newdata=prepped_example_subset)
preds <- format_predictions(outcome=prepped_example_subset$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
#twoClassSummary throws error for tibbles input
twoClassSummary(as.data.frame(preds), lev = levels(preds$obs))
```

    ##   ROC  Sens  Spec 
    ## 0.375 0.500 0.500

#### The Framingham risk model

The Framingham risk model was the only externally developed model we
could find in the literature that we could implement using the available
HUNT Study data \[[6](#ref-RN10737)\] . To encourage reproduction and
further external validation, we include the model used here. This model
only require the following variables in the dataframe/tibble provided to
the ‘newdata’ argument: *Age*, *systolic BP*, *diastolic BP*, *BMI*,
*sex*, *smoking status* and *family history of hypertension*.

Note that the data must be in its original form when the Framingham risk
model is used, i.e., without preprocessing. See Table in S4 Table in the
development article for details on model adaptations to fit the
available HUNT Study data \[[5](#ref-schjerven2022)\].

``` r
# Note the use of example_data and not prepped_example_data

# Explicit example of needed input
example_subset <- example_data[,c("Age", "SysBP", "DiaBP", "BMI", "FamHypHist", "SmokingStatus", "Sex", "HypOutcome")]

# Predictions
y_hat <- predict(framingham, newdata=example_subset)
preds <- format_predictions(outcome=example_subset$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
#twoClassSummary throws error for tibbles input
twoClassSummary(as.data.frame(preds), lev = levels(preds$obs))
```

    ##       ROC      Sens      Spec 
    ## 0.5000000 0.3333333 0.7500000

#### The Framingham risk model recalibrated using HUNT Study data

Lastly, we include the Framingham risk model after recalibration to the
HUNT Study data using method 2 suggested by Steyerberg
\[[7](#ref-steyerberg2004)\] . See Table in S4 Table in the development
article for details on model adaptations to fit the available HUNT Study
data \[[5](#ref-schjerven2022)\].

``` r
#We use the same dataset as for 'framingham'

# Predictions
y_hat <- predict(recalibrated_framingham, newdata=example_subset)
preds <- format_predictions(outcome=example_subset$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
#twoClassSummary throws error for tibbles input
twoClassSummary(as.data.frame(preds), lev = levels(preds$obs))
```

    ##  ROC Sens Spec 
    ##  0.5  0.5  0.5

#### Performance measures

We include a formatting function for easy use of caret-package and
custom metrics. We included the Brier Score and Integrated Calibration
Index as custom functions \[[8](#ref-austin2019)\]
\[[9](#ref-brier1950)\] . See, e.g.,
<https://topepo.github.io/caret/measuring-performance.html> for more
performance metrics. Note that since the example data is only for
demonstration, the performance measures are shown for illustrative
purposes only and do not reflect real performance.

``` r
preds <- format_predictions(outcome=prepped_example_data$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
#twoClassSummary throws error for tibbles input
twoClassSummary(as.data.frame(preds), lev = levels(preds$obs))
```

    ##  ROC Sens Spec 
    ##  0.5  0.5  0.5

``` r
#Area under the precision-recall curve
prSummary(preds, lev = levels(preds$obs))
```

    ##       AUC Precision    Recall         F 
    ## 0.4904431 0.6000000 0.5000000 0.5454545

``` r
#Brier score
brierSummary(preds)
```

    ## Brier Score 
    ##    0.266582

``` r
#Integrated calibration index
iciSummary(preds)
```

    ##       ICI 
    ## 0.1641066

#### References

<div id="refs" class="references csl-bib-body">

<div id="ref-åsvold2022" class="csl-entry">

<span class="csl-left-margin">1. </span><span
class="csl-right-inline">Åsvold BO, Langhammer A, Rehn TA, Kjelvik G,
Grøntvedt TV, Sørgjerd EP, et al. Cohort Profile Update: The HUNT Study,
Norway. International Journal of Epidemiology. 2022.
doi:[10.1093/ije/dyac095](https://doi.org/10.1093/ije/dyac095)</span>

</div>

<div id="ref-chen2021" class="csl-entry">

<span class="csl-left-margin">2. </span><span
class="csl-right-inline">Chen T, He T, Benesty M, Khotilovich V, Tang Y,
Cho H, et al. Xgboost: Extreme gradient boosting. 2021. Available:
<https://github.com/dmlc/xgboost></span>

</div>

<div id="ref-breiman2001" class="csl-entry">

<span class="csl-left-margin">3. </span><span
class="csl-right-inline">Breiman L. Random forests. Machine Learning.
2001;45: 5–32.
doi:[10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)</span>

</div>

<div id="ref-zou2005" class="csl-entry">

<span class="csl-left-margin">4. </span><span
class="csl-right-inline">Zou H, Hastie T. Regularization and variable
selection via the elastic net. Journal of the Royal Statistical Society:
Series B (Statistical Methodology). 2005;67: 301–320.
doi:[10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)</span>

</div>

<div id="ref-schjerven2022" class="csl-entry">

<span class="csl-left-margin">5. </span><span
class="csl-right-inline">Schjerven FE, Ingeström E, Lindseth F,
Steinsland I. Can machine learning improve risk prediction of incident
hypertension? An internal method comparison and external validation of
the Framingham risk model using HUNT Study data. 2022 Nov.
doi:[10.1101/2022.11.02.22281859](https://doi.org/10.1101/2022.11.02.22281859)</span>

</div>

<div id="ref-RN10737" class="csl-entry">

<span class="csl-left-margin">6. </span><span
class="csl-right-inline">Parikh NI, Pencina MJ, Wang TJ, Benjamin EJ,
Lanier KJ, Levy D, et al. A risk score for predicting near-term
incidence of hypertension: The framingham heart study. Annals of
Internal Medicine. 2008;148: 102–110.
doi:[10.7326/0003-4819-148-2-200801150-00005](https://doi.org/10.7326/0003-4819-148-2-200801150-00005)</span>

</div>

<div id="ref-steyerberg2004" class="csl-entry">

<span class="csl-left-margin">7. </span><span
class="csl-right-inline">Steyerberg EW, Borsboom GJJM, Houwelingen HC
van, Eijkemans MJC, Habbema JDF. Validation and updating of predictive
logistic regression models: a study on sample size and shrinkage.
Statistics in Medicine. 2004;23: 2567–2586.
doi:[10.1002/sim.1844](https://doi.org/10.1002/sim.1844)</span>

</div>

<div id="ref-austin2019" class="csl-entry">

<span class="csl-left-margin">8. </span><span
class="csl-right-inline">Austin PC, Steyerberg EW. The Integrated
Calibration Index (ICI) and related metrics for quantifying the
calibration of logistic regression models. Statistics in Medicine.
2019;38: 4051–4065.
doi:[10.1002/sim.8281](https://doi.org/10.1002/sim.8281)</span>

</div>

<div id="ref-brier1950" class="csl-entry">

<span class="csl-left-margin">9. </span><span
class="csl-right-inline">Brier GW. Verification of forecasts expressed
in terms of probability. Monthly weather review. 1950;78: 1–3. </span>

</div>

</div>
