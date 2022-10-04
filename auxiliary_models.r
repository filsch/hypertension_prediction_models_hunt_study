predict.simple_lr_model <- function(obj, newdata){
  df=newdata
  x = -1.3946 + -0.077554*df$Height + 0.186892*df$BMI + 0.448547*df$SysBP + 0.399027*df$DiaBP + 0.036866*df$FamHypHist_yes + 0.496008*df$Age           
  
  p <- 1/(1 + exp(-x))
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}


predict.FraminghamModel <- function(obj, newdata){
  age <- newdata$Age
  sys_bp <- newdata$SysBP
  dia_bp <- newdata$DiaBP
  bmi <- newdata$BMI
  
  smoke <- as.numeric(newdata$SmokingStatus == "currently")
  #Adjustment of coefficient
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
  
  
  smoke <- as.numeric(newdata$SmokingStatus == "currently")
  par_hyp <- as.numeric(newdata$FamHypHist == "yes")*1.6394
  women <- as.numeric(newdata$Sex == "female")
  
  x = log(11) - (22.9495 - 0.1564*age - 0.2029*women - 0.0593*sys_bp - 0.1285*dia_bp - 0.1907*smoke - 0.1661*par_hyp - 0.0339*bmi + 0.0016*age*dia_bp)
  x = x / 0.8769
  #Recalibration
  s = -0.4248 + x
  p <- 1/(1+exp(-s))
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}

simpler_model <- structure(list(), class="simple_lr_model")
framingham <- structure(list(), class="FraminghamModel")
recalibrated_framingham <- structure(list(), class="RecalibratedFraminghamModel")
