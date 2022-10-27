#Helper functions
iciSummary <- function(predictions, lev = NULL, model = NULL){
  # Input
  # predictions: formatted tibble outputted by format_predictions
  #
  # Output
  # ici: the integrated calibration index
  
  y <- as.numeric(predictions$obs)-1
  p <- predictions$Hypertensive
  
  #Excluding some cases where the interpolation fails.
  if (all(is.na(p))){
    return(NA)
  } else if (all(diff(p)==0)) {
    return(NA)
  } else if (length(unique(p)) < 5){
    return(NA)
  }
  
  try(calib <- loess(y ~ p), silent=T)
  
  if(is.null(calib)){
    return(NA)
  }
  
  P.calibrate <- predict(calib, newdata = p)
  ici <- mean(abs(P.calibrate - p))
  
  return(c("ICI"=ici))
}

brierSummary <- function (data, lev = NULL, model = NULL) {
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
  Brier=mean((target - pred)^2)
  return(c("Brier Score"=Brier))
}

format_predictions <- function(outcome, predicted_probs, cutoff=0.25){
  # Input
  # outcome: n-length vector of factor with observed outcomes, either 'Normotensive' or 'Hypertensive'
  # predicted_probs: n x 2 tibble with columns 'Normotensive' and 'Hypertensive', with predicted probability of outcome as each row
  # cutoff: value in which probabilities above are belong to class 'Hypertensive'. 0.25 is the outcome rate in the HUNT Study.
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
