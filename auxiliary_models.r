predict.simple_lr_model <- function(obj, newdata){
  #Corresponds to LASSO with log(Alpha) = -4.383767, see Figure 3 in development article. 
  df=newdata
  x = -1.385575 - 0.07447008*df$Height + 0.18416595*df$BMI + 0.44515847*df$SysBP + 0.39681464*df$DiaBP
      + 0.02019300*df$FamHypHist_yes + 0.49290664*df$Age           
  
  p <- 1/(1 + exp(-x))
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}


predict.FraminghamModel <- function(obj, newdata){
  age <- newdata$Age
  sys_bp <- newdata$SysBP
  dia_bp <- newdata$DiaBP
  bmi <- newdata$BMI
  
  #Adjustment of coefficients, see Table in S4 Table in publication.
  smoke <- as.numeric(newdata$SmokingStatus == "currently")
  par_hyp <- as.numeric(newdata$FamHypHist == "yes")*1.6394
  women <- as.numeric(newdata$Sex == "female")
  
  x = log(11) - (22.9495 - 0.1564*age - 0.2029*women - 0.0593*sys_bp - 0.1285*dia_bp - 0.1907*smoke - 0.1661*par_hyp - 0.0339*bmi + 0.0016*age*dia_bp)
  z = exp(x / 0.8769)
  p <- 1 - exp(-z)
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}

predict.RecalibratedFraminghamModel <- function(obj, newdata){
  age <- newdata$Age
  sys_bp <- newdata$SysBP
  dia_bp <- newdata$DiaBP
  bmi <- newdata$BMI
  
  #Adjustment of coefficients, see Table in S4 Table in publication.
  smoke <- as.numeric(newdata$SmokingStatus == "currently")
  par_hyp <- as.numeric(newdata$FamHypHist == "yes")*1.6394
  women <- as.numeric(newdata$Sex == "female")
  
  x = log(11) - (22.9495 - 0.1564*age - 0.2029*women - 0.0593*sys_bp - 0.1285*dia_bp - 0.1907*smoke - 0.1661*par_hyp - 0.0339*bmi + 0.0016*age*dia_bp)
  x = x / 0.8769
  #Recalibration
  s = -0.4251 + x
  p <- 1/(1+exp(-s))
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}

simpler_model <- structure(list(), class="simple_lr_model")
framingham <- structure(list(), class="FraminghamModel")
recalibrated_framingham <- structure(list(), class="RecalibratedFraminghamModel")
