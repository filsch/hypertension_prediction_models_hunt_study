---
title: "Hypertension risk models developed using the HUNT Study data"
author: "Filip Schjerven"
date: "2022-10-03"
output: rmarkdown::github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(caret)
library(xgboost) #pre 1.6 version
library(RRF)
library(randomForest)
library(glmnet)
library(kernlab)
library(recipes)
 ```

Dissemination and external validation of hypertension risk models is important, but rarely accommodated for in the literature. This is especially true when machine learning models are being used, as they are often too complex to describe in traditional appendiceses. We provide the machine learning models developed using the XGBoost, Random Forest, and Elastic regression methods on the HUNT Study data [cite]. These models predict the 11-year risk of incident hypertension and were fitted using a large population cohort of 23722 individuals and 18 input features. See ... for details on how these models were fitted.

In addition, we include three additional models:
  
- A smaller logistic regression model developed as a sensitivity analysis on the HUNT Study data. This model utilizes only a few, more easily available features. 
- An adaptation of the externally developed Framingham risk model, and
- an adaptation of the Framingham risk model recalibrated on the HUNT Study data. 

Two more modelling methods were reported in our study: The K-Nearest Neighbour and the SVM models. These are not provided here, as they cannot be detached from the development data which we do not have permission to share freely. See ... for more info and how to apply for access to these data. 

#### Models, preprocessings and helper functions.
```{r load models}
setwd("C:/Users/fschj/Downloads")
load("models.rds")
load("prep.rds")
source("helper_functions.r")
```

```{r delete_data, echo=F}
# mod <- git$models$xgb
# mod$control$index <- NULL
# git$models$xgb <- mod
# 
# mod <- git$models$rf
# 
# mod$control$index <- NULL
# 
# mod$finalModel$predicted <- NULL
# mod$finalModel$err.rate <- NULL
# mod$finalModel$votes <- NULL
# mod$finalModel$oob.times <- NULL
# mod$finalModel$y <- NULL
# git$models$rf <- mod
# 
# mod <- git$models$svmp
# mod$control$index <- NULL
# attr(mod$finalModel, 'xmatrix') <- NULL
# attr(mod$finalModel, 'ymatrix') <- NULL
# attr(mod$finalModel, 'alpha') <- NULL
# attr(mod$finalModel, 'SVindex') <- NULL
# git$models$svmp <- mod
# 
# mod <- git$models$elastic
# mod$control$index <- NULL
# mod$call <- NULL
# mod$finalModel$call <- NULL
# git$models$elastic <- mod
# #save(git, file="git_cleansed.rds")

```

#### Data

We include some example data to illustrate the input data required for the different models. All possible variable types and levels are represented. See ... for information on variables and how they were recorded and constructed.

``` {r data}
load("example_data.rds")
```
The variables needed as input to the XGBoost, Random Forest and Elastic regression models:

| Variable | Type | Levels |
| --- | --- | --- |
|Sex           | factor  | female, male |
|Height        | numeric | - |
|BMI           | numeric | - |
|SysBP         | numeric | - |
|DiaBP         | numeric | - |
|SeChol        | numeric | - |
|SeHDLChol     | numeric | - |
|SeTrig        | numeric | - |
|SeGluNonFast  | numeric | - |
|SeCreaCorr    | numeric | - |
|GFREstStag    | factor  | stage 1, stage 2-5 |
|Age           | numeric | - |
|FamCVDHist    | factor  | no, yes |
|SmokingStatus | factor  | never, formerly, currently |
|Education     | factor  | secondary, upper_secondary, high_school, college, post_graduate |
|FamHypHist    | factor  | no, yes |
|HypOutcome    | ordered factor  | Normotensive < Hypertensive |
|PhysAct       | factor  | low, mid, high |


#### Preprocessing and predictions
The 'prep' variable contains pre-processing function that standardizes numerical variables and one-hot-encodes factors so they match model input requirements for the machine learning models. The parameters of the 'prep' variable has been learned from the training data used to develop the models. Note, we assume that that the data should be complete, i.e., no missing entries, and with similar names and format as the example data. 

```{r prep_predict, results = 'hide'}
prepped_example_data <- bake(prep, example_data)

model <- models$xgb #xgb, rf or elastic

y_hat <- predict(model, newdata=prepped_example_data, type="prob")
head(y_hat, n=2)
```
```{r echo=F}
head(y_hat, n=2)
```

#### Model using simple-to-collect variables
We include a logistic regression model that only uses variables that are more easily available. These are: *Age*, *systolic blood pressure*, *diastolic blood pressure*, *BMI*, *height*, and *history of hypertension in close family*. This model was derived from a sensitivity analysis performed after development of the full models [cite]. The corresponding regularization penalty applied in model fitting was log($\lambda$) = -4.4301, see Figure ... in the development article.

```{r simpler}
source("auxiliary_models.r")

# Explicit example of needed input
prepped_example_data <- bake(prep, example_data)
prepped_example_subset <- prepped_example_data[,c("Age", "SysBP", "DiaBP", "BMI", "FamHypHist_yes", "Height", "HypOutcome")]

# Predictions
y_hat <- predict(simpler_model, newdata=prepped_example_subset)
preds <- format_predictions(outcome=prepped_example_subset$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
twoClassSummary(preds, lev = levels(preds$obs))

```

#### The Framingham risk model
The Framingham risk model was the only externally developed model we could find that we were able to implement using the available variables in the HUNT Study data [cite]. To encourage reproduction and further external validation, we include the model here. This model only require the following variables in the dataframe/tibble provided to the 'newdata' argument: *Age*, *systolic BP*, *diastolic BP*, *BMI*, *sex*, *smoking status* and *family history of hypertension*.

Note that the data must be in its original form when the Framingham risk model is used, i.e., without preprocessing. See table A ... in ... for details on adaptations made to the model to fit the available data in the HUNT Study.

```{r fr}
# Note the use of example_data and not prepped_example_data

# Explicit example of needed input
example_subset <- example_data[,c("Age", "SysBP", "DiaBP", "BMI", "FamHypHist", "SmokingStatus", "Sex", "HypOutcome")]

# Predictions
y_hat <- predict(framingham, newdata=example_subset)
preds <- format_predictions(outcome=example_subset$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
twoClassSummary(preds, lev = levels(preds$obs))
```
#### The Framingham risk model recalibrated using HUNT Study data
Lastly, we include the Framingham risk model after recalibration to the HUNT Study data using method 2 suggested by Steyerberg [cite]. Again, note the use of the original data. See Table A ... in ... for details on the recalibrated model.

```{r recal fr}
#We use the same dataset as for 'framingham'

# Predictions
y_hat <- predict(recalibrated_framingham, newdata=example_subset)
preds <- format_predictions(outcome=example_subset$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
twoClassSummary(preds, lev = levels(preds$obs))
```

#### Performance measures
We include a formatting function for easy use of caret-package metrics as well as custom metrics. We have included the Brier Score and Integrated Calibration Index as custom functions [cite]. See, e.g., https://topepo.github.io/caret/measuring-performance.html for more performance metrics. Note that since the example data is randomly sampled data, the performance measures are shown only for illustrative purposes.
```{r measure performance}
preds <- format_predictions(outcome=prepped_example_data$HypOutcome, predicted_probs=y_hat) 

#Area under the receiver operator curve:
twoClassSummary(preds, lev = levels(preds$obs))

#Are under the precision-recall curve
prSummary(preds, lev = levels(preds$obs))

#Brier score
compute_brier(preds$y, preds$Hypertensive)

#Integrated calibration index
compute_ici(preds$y, preds$Hypertensive)
```


