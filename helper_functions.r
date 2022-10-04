compute_ici <- function(y, p){
  # y - numeric vector of outcomes. 'Normotensive' = 0, 'Hypertensive' = 1
  # p - predicted probability of 'Hypertensive' outcome
  
  if(all(diff(p)==0)){
    return(NA)
  }
  calibration_line <- loess(y ~ p)
  p_hat <- predict(calibration_line, newdata = p)
  return(mean(abs(p_hat - p)))
}

compute_brier <- function(y, p){
  # y - numeric vector of outcomes. 'Normotensive' = 0, 'Hypertensive' = 1
  # p - numeric vector with predicted probabilities of 'Hypertensive' outcome
  
  return(mean((y-p)**2))
}

format_predictions <- function(outcome, predicted_probs){
  # outcome - n length vector of outcome class. factor with ordered levels 'Normotensive < Hypertensive'
  # predicted_probs - n x 2 dataframe, each row contain predicted probability of class 'Normotensive' and 'Hypertensive' respectively
  
  ordered_outcomes <- levels(outcome)
  predicted_class <- rep("Normotensive", length(outcome))
  predicted_class[predicted_probs$Hypertensive > 0.25] <- "Hypertensive" #0.25 is the outcome rate of HUNT Study data
  
  preds_formatted <- data.frame(obs=outcome,
                                pred=factor(predicted_class, levels=ordered_outcomes, ordered=T),
                                "Hypertensive"=predicted_probs$Hypertensive,
                                "Normotensive"=1-predicted_probs$Hypertensive,
                                y=as.numeric(outcome) - 1)
  return(preds_formatted)
}
