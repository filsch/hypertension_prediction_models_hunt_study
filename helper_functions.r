##Helper functions
totalSummary <- function (data, lev = NULL, model = NULL) {
  # Input
  # data: tibble/data.frame containing columns 'obs' and 'Hypertensive', 
  #         - 'obs': ordered factor with levels 'Normotensive' < 'Hypertensive'
  #         - 'Hypertensive': Predicted probability of 'Hypertensive' outcome
  #         Each row corresponds to an individual
  # Output
  # Brier: Squared value between 'obs' (coded 'Normotensive' = 0 and 'Hypertensive'= 1)
  #        and the predicted probabilities of 'Hypertensive' outcome, averaged over all rows.
  target <- as.numeric(data$obs)-1
  pred <- data$Hypertensive
  Brier <- mean((target - pred)^2)
  naive_brier <- mean((target - mean(target))^2)
  
  #ICI <- iciSummary(data)
  y <- as.numeric(data$obs)-1
  p <- data$Hypertensive
  
  #Excluding some cases where the interpolation fails.
  if (all(is.na(p))){
    ICI = NA
  } else if (all(diff(p)==0)) {
    ICI = NA
  } else if (length(unique(p)) < 5){
    ICI = NA
  } else {
    try(calib <- loess(y ~ p), silent=T)
    if(is.null(calib)){
      ICI = NA
    } else {
      P.calibrate <- predict(calib, newdata = p)
      ICI <- mean(abs(P.calibrate - p))
    }  
  }
  
  AUC <- twoClassSummary(as.data.frame(data), lev = levels(data$obs))[["ROC"]]
  
  return(c("sBrier Score"=1-Brier/naive_brier, "ICI"=ICI, "AUC"=AUC))
}

format_predictions <- function(outcome, predicted_probs, cutoff=0.25){
  # Input
  # outcome: n-length vector of factor with observed outcomes, either 'Normotensive' or 'Hypertensive'
  # predicted_probs: n x 2 tibble with columns 'Normotensive' and 'Hypertensive', with predicted probability of outcome as each row
  # cutoff: value in which probabilities above are belong to class 'Hypertensive'
  #
  # Output
  # preds_formatted: formatted tibble for use in Caret summary functions.
  ordered_outcomes <- levels(outcome)
  predicted_class <- rep("Normotensive", length(outcome))
  predicted_class[predicted_probs$Hypertensive > cutoff] <- "Hypertensive"
  
  preds_formatted <- tibble(obs=outcome,
                            pred=factor(predicted_class, levels=ordered_outcomes, ordered=T),
                            "Hypertensive"=predicted_probs$Hypertensive,
                            "Normotensive"=1-predicted_probs$Hypertensive,
                            y=as.numeric(outcome) - 1)
  return(preds_formatted)
}

dummies_to_categorical <- function(data){
  # Helper function for recategorize factors with multiple levels for xgboost and rf model
  # Input 
  # data: tibble with dummified and numerical columns of smokingstatus, education and physact
  
  # Output
  # data: tibble with smokingstatus, education and physact as categories
  
  levels_to_undummify = list("SmokingStatus"=c("never","formerly","currently"),
                             "Education"=c("secondary", "upper_secondary", "high_school", "college", "post_graduate"),
                             "PhysAct"=c("low", "moderate", "high"), "LoveStat"=c("Never married", "Married.Reg..partner", "Divorced"))
  for (name in names(levels_to_undummify)){
    arr <- rep(NA, dim(data)[1])
    for (category in levels_to_undummify[[name]]){
      column <- paste0(name, "_", category)
      if (column %in% names(data)){
        arr[data[[column]]==1] <- category
      } else { 
        comparator <- category
      } 
      data[[column]] <- NULL
    }
    #Assumed complete data
    arr[is.na(arr)] <- comparator
    if(name %in% c("Education", "PhysAct")){
      data[[name]] <- factor(arr, levels = levels_to_undummify[[name]], ordered=T)  
    } else {
      data[[name]] <- factor(arr)  
    }
    
  }
  return(data)
}
