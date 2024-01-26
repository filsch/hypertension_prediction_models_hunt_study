Hypertension risk models developed using the HUNT Study data
================
Filip Schjerven
2024-01-24

Dissemination and external validation of hypertension risk models is
important, but rarely accommodated for in the literature. This is
especially true when machine learning models are being used, as they are
often too complex to describe in traditional appendices. We used machine
learning to predict risk of hypertension 11 years later on data from 17
852 HUNT Study participants, and provide the machine learning models
developed using the XGBoost, Random Forest, and Elastic regression
methods<sup>[1](#ref-asvold_cohort_2022)</sup><sup>[2](#ref-chen_xgboost_2021)</sup><sup>[3](#ref-breiman_random_2001)</sup><sup>[4](#ref-zou_regularization_2005)</sup>.
A reference to the development-article detailing data flow, model
development, and data accessibility will be made upon publication.

In total, we provide 13 risk models:

<ol>
<li>
An XGBoost model,
</li>
<li>
an elastic regression model,
</li>
<li>
a Random Forest model,
</li>
<li>
a smaller logistic regression model used as a reference,
</li>
<li>
a ‘high normal BP’ decision rule,
</li>
<li>
an adaptation of the externally developed Framingham risk model to the
HUNT Study data, and
</li>
<li>
an adaptation of the Framingham risk model recalibrated on the HUNT
Study data
</li>
<li>
six more externally developed risk models that were applicable on our
data, developed from Chinese, Iranian and Korean populations
</li>
</ol>

Two more modelling methods were applied in our study: The K-Nearest
Neighbour and the SVM methods. These models are not provided here, as
they could not be detached from the development data which we do not
have permission to share. Data can be obtained upon approval from REK
and HUNT Research Centre. For more information see:
www.ntnu.edu/hunt/data.

#### Models, preprocessings and helper functions

``` r
# library(caret) #version 6.0.93
# library(xgboost) #version 1.7.5.1 
# library(RRF) #version 1.9.4
# library(randomForest) #version 4.7.1.1
# library(recipes) #version 1.0.1

load("git_models.rds")
load("git_prep.rds")
source("helper_functions.r")
source("auxiliary_models.r")
```

#### Data

We include some example data to illustrate the input data for the
different models. All variables and levels are represented. See the
development-article for information on variables and how they were
recorded and constructed. Note that the example data is simulated, i.e.,
not real data, and is only included to demonstrate how the resources in
this repository may be used.

``` r
load("git_example_data.rds")
head(example_data, 2)
```

    ## # A tibble: 2 × 19
    ##   Sex   Height   BMI SysBP DiaBP SeChol SeHDLChol SeTrig SeGluNonFast SeCreaCorr
    ##   <fct>  <dbl> <dbl> <dbl> <dbl>  <dbl>     <dbl>  <dbl>        <dbl>      <dbl>
    ## 1 fema…    163  25     130    81    4.8       1.2   1.35          5.2         61
    ## 2 male     181  23.3   135    71    8.2       0.7   0.8           4.7         65
    ## # ℹ 9 more variables: GFREstStag <fct>, Age <dbl>, FamCVDHist <fct>,
    ## #   SmokingStatus <fct>, Education <fct>, FamHypHist <fct>, HypOutcome <fct>,
    ## #   PhysAct <fct>, LoveStat <fct>

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
| PhysAct       | factor         | low, mid, high                                                  |
| LoveStatus    | factor         | married, never married, divorced                                |
| HypOutcome    | ordered factor | Not hypertension \< Hypertensive                                |

#### Preprocessing and predictions

The ‘prep’ object found in ‘git_prep.rds’ contains two pre-processing
functions, which paired with the ‘dummies_to_categorical()’ function
provides all preprocessing needed for the ML models included. The steps
needed for the various models are as follows:

##### Table 2: Preprocessing needed for included models:

| models                                                                     | Preprocessing needed                                        |
|----------------------------------------------------------------------------|-------------------------------------------------------------|
| Elastic regression                                                         | Standardization (numerical)\* + Dummification (categorical) |
| XGBoost                                                                    | Standardization (numerical)\*                               |
| Random Forest                                                              | Standardization (numerical)\*                               |
| Decision rule                                                              | \-                                                          |
| Logistic regression reference model                                        | \-                                                          |
| Framingham risk model                                                      | \-                                                          |
| Recalibrated Framingham risk model                                         | \-                                                          |
| Chinese clinical risk model                                                | \-                                                          |
| Chinese clinical risk model, from individuals without diabetes at baseline | \-                                                          |
| KoGES model                                                                | \-                                                          |
| TLGS model                                                                 | \-                                                          |
| CAVAS model                                                                | \-                                                          |
| F-CAVAS model                                                              | \-                                                          |

\* Standardization to the mean and standard deviation of the training
set, see the development article. These are already included in the
‘prep’ object.

The following example show how to predict using the example data for all
models:

``` r
#Elastic regression
prepped_example_data <- bake(prep, example_data)

model <- models$elastic

y_hat <- predict(model, newdata=prepped_example_data, type="prob")

## XGBoost or Random forest
prepped_example_data <- bake(prep, example_data) %>% dummies_to_categorical()

model <- models$xgb #xgb, rf

y_hat <- predict(model, newdata=prepped_example_data, type="prob")

#auxiliary models, note the use of 'example_data', i.e., unprepped
#External models: chinese_risk_model, chinese_risk_model_no_diabetes, koges, tlgs, cavas, f_cavas, framingham
#other: logreg_reference, high_normal_bp_rule, recalibrated_framingham
model <- logreg_reference 
  
y_hat <- predict(model, newdata=example_data)
head(y_hat, n=2)
```

    ##   Normotensive Hypertensive
    ## 1    0.7570686    0.2429314
    ## 2    0.8176177    0.1823823

#### Externally developed risk models

To encourage the use of existing models on hypertension risk modelling,
we searched the literature for risk models suitable for our time-frame
with variables that could be adapted to those found in the HUNT Study
Data. We found seven models from five articles: The Framingham risk
model<sup>[5](#ref-parikh_risk_2008)</sup>, the Chinese risk
models<sup>[6](#ref-chien_prediction_2011)</sup>, the KoGES
model<sup>[7](#ref-lim_validation_2016)</sup>, the TLGS
model<sup>[8](#ref-koohi_validation_2021)</sup>, and the CAVAS and
F-CAVAS models<sup>[9](#ref-namgung_development_2022)</sup>. To
encourage reproduction and further external validation, we include the
models used here. The only needed features are *Age*, *systolic BP*,
*diastolic BP*, *BMI*, *sex*, *smoking status* and *family history of
hypertension*. A recalibrated version of the Framingham risk model is
also included.

Details on the adaptations made to the risk models are described in the
development article.

#### Performance measures

We include a formatting function for easy use of caret-package and
custom metrics. We included the scaled Brier Score and Integrated
Calibration Index as custom
functions<sup>[10](#ref-austin_integrated_2019)</sup><sup>[11](#ref-brier_verification_1950)</sup><sup>[12](#ref-steyerberg_assessing_2010)</sup>.
See, e.g., <https://topepo.github.io/caret/measuring-performance.html>
for more performance metrics. Note that since the example data is only
for demonstration, the performance measures are shown for illustrative
purposes only and do not reflect real performance.

``` r
preds <- format_predictions(outcome=prepped_example_data$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
#twoClassSummary throws error for tibbles input
twoClassSummary(as.data.frame(preds), lev = levels(preds$obs))
```

    ##       ROC      Sens      Spec 
    ## 0.5191888 0.6938776 0.3962264

``` r
#Scaled Brier score, Integrated Calibration Index, Area under the receiver operator curve
totalSummary(preds, lev = levels(preds$obs))
```

    ## sBrier Score          ICI          AUC 
    ##  -0.09612304   0.09939779   0.51918881

#### References

<div id="refs" class="references csl-bib-body" line-spacing="2">

<div id="ref-asvold_cohort_2022" class="csl-entry">

<span class="csl-left-margin">1.
</span><span class="csl-right-inline">Csvold, B. O. *et al.* Cohort
Profile Update: The HUNT Study, Norway. *International Journal of
Epidemiology* (2022)
doi:[10.1093/ije/dyac095](https://doi.org/10.1093/ije/dyac095).</span>

</div>

<div id="ref-chen_xgboost_2021" class="csl-entry">

<span class="csl-left-margin">2.
</span><span class="csl-right-inline">Chen, T. *et al.* [Xgboost:
Extreme Gradient Boosting](https://github.com/dmlc/xgboost).
(2021).</span>

</div>

<div id="ref-breiman_random_2001" class="csl-entry">

<span class="csl-left-margin">3.
</span><span class="csl-right-inline">Breiman, L. [Random
Forests](https://doi.org/10.1023/A:1010933404324). *Machine Learning*
**45**, 5–32 (2001).</span>

</div>

<div id="ref-zou_regularization_2005" class="csl-entry">

<span class="csl-left-margin">4.
</span><span class="csl-right-inline">Zou, H. & Hastie, T.
[Regularization and variable selection via the elastic
net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the
Royal Statistical Society: Series B (Statistical Methodology)* **67**,
301–320 (2005).</span>

</div>

<div id="ref-parikh_risk_2008" class="csl-entry">

<span class="csl-left-margin">5.
</span><span class="csl-right-inline">Parikh, N. I. *et al.* [A risk
score for predicting near-term incidence of hypertension: The Framingham
Heart Study.](https://doi.org/10.7326/0003-4819-148-2-200801150-00005)
*Annals of internal medicine* **148**, 102–110 (2008).</span>

</div>

<div id="ref-chien_prediction_2011" class="csl-entry">

<span class="csl-left-margin">6.
</span><span class="csl-right-inline">Chien, K.-L. *et al.* [Prediction
models for the risk of new-onset hypertension in ethnic Chinese in
Taiwan.](https://doi.org/10.1038/jhh.2010.63) *Journal of human
hypertension* **25**, 294–303 (2011).</span>

</div>

<div id="ref-lim_validation_2016" class="csl-entry">

<span class="csl-left-margin">7.
</span><span class="csl-right-inline">Lim, N.-K., Lee, J.-W. & Park,
H.-Y. [Validation of the Korean Genome Epidemiology Study Risk Score to
Predict Incident Hypertension in a Large Nationwide Korean
Cohort.](https://doi.org/10.1253/circj.CJ-15-1334) *Circulation journal
: official journal of the Japanese Circulation Society* **80**,
1578–1582 (2016).</span>

</div>

<div id="ref-koohi_validation_2021" class="csl-entry">

<span class="csl-left-margin">8.
</span><span class="csl-right-inline">Koohi, F. *et al.* [Validation of
the Framingham hypertension risk score in a middle eastern population:
Tehran lipid and glucose study
(TLGS)](https://doi.org/10.1186/s12889-021-10760-6). *BMC PUBLIC HEALTH*
**21**, (2021).</span>

</div>

<div id="ref-namgung_development_2022" class="csl-entry">

<span class="csl-left-margin">9.
</span><span class="csl-right-inline">Namgung, H. K. *et al.*
Development and validation of hypertension prediction models: The Korean
Genome and Epidemiology Study\_cardiovascular Disease Association Study
(KoGES\_CAVAS). *Journal of human hypertension* (2022)
doi:[10.1038/s41371-021-00645-x](https://doi.org/10.1038/s41371-021-00645-x).</span>

</div>

<div id="ref-austin_integrated_2019" class="csl-entry">

<span class="csl-left-margin">10.
</span><span class="csl-right-inline">Austin, P. C. & Steyerberg, E. W.
[The Integrated Calibration Index (ICI) and related metrics for
quantifying the calibration of logistic regression
models](https://doi.org/10.1002/sim.8281). *Statistics in Medicine*
**38**, 4051–4065 (2019).</span>

</div>

<div id="ref-brier_verification_1950" class="csl-entry">

<span class="csl-left-margin">11.
</span><span class="csl-right-inline">Brier, G. W. Verification of
forecasts expressed in terms of probability. *Monthly weather review*
**78**, 1–3 (1950).</span>

</div>

<div id="ref-steyerberg_assessing_2010" class="csl-entry">

<span class="csl-left-margin">12.
</span><span class="csl-right-inline">Steyerberg, E. W. *et al.*
[Assessing the performance of prediction models: A framework for
traditional and novel
measures](https://doi.org/10.1097/EDE.0b013e3181c30fb2). *Epidemiology
(Cambridge, Mass.)* **21**, 128–138 (2010).</span>

</div>

</div>
