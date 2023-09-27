predict.LogisticRegressionReferenceModel <- function(obj, newdata){
  df=newdata
  x = -14.73339 + 0.05055*df$Age + 0.05074*df$SysBP + 0.06527*df$DiaBP
  p <- 1/(1 + exp(-x))
  
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}


predict.FraminghamModel <- function(obj, newdata){
  age <- newdata$Age
  sys_bp <- newdata$SysBP
  dia_bp <- newdata$DiaBP
  bmi <- newdata$BMI
  
  #Adjustment of coefficients, see details in publication.
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
  s = -0.3367 + x
  p <- 1/(1 + exp(-s))
  return(data.frame("Normotensive"=1-p, "Hypertensive"=p))
}

predict.HighNormalBPRule <- function(obj, newdata, type=NULL){
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  preds <- as.numeric((DiaBP >= 85) | (SysBP >= 130))
  return(data.frame("Normotensive"=1-preds, "Hypertensive"=preds))
}

#Other external models that were validated
predict.ChineseRiskModel_Chien <- function(obj, newdata, type=NULL){
  Age <- newdata$Age
  Female <- abs(newdata$Sex_male - 1)
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  BMI <- newdata$BMI
  
  z = log(11) - (8.173 - 0.011*Age + 0.124*Female - 0.029*SysBP
                 - 0.014*DiaBP - 0.043*BMI)
  preds <- 1 - exp(-exp(z/0.592))
  return(tibble("Normotensive"=1-preds, "Hypertensive"=preds))
}

predict.ChineseRiskModel_DiabetesFreePop_Chien <- function(obj, newdata, type=NULL){
  Age <- newdata$Age
  Female <- abs(newdata$Sex_male - 1)
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  BMI <- newdata$BMI
  
  z = log(11) - (8.141 - 0.009*Age + 0.154*Female - 0.029*SysBP
                 - 0.014*DiaBP - 0.048*BMI)
  preds <- 1 - exp(-exp(z/0.59))
  return(tibble("Normotensive"=1-preds, "Hypertensive"=preds))
}

predict.KoGES_Lim2016 <- function(obj, newdata, type=NULL){
  Age <- newdata$Age
  Female <- abs(newdata$Sex_male - 1)
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  Smoking <- newdata$SmokingStatus_currently
  FamHypHist <- newdata$FamHypHist_yes
  BMI <- newdata$BMI
  
  z = log(11) - (30.3547 - 0.2838*Age - 0.2084*Female - 0.0642*SysBP
                 - 0.2033*DiaBP - 0.2912*Smoking - 0.0875*FamHypHist #Coefficient for FamHypHist has been changed in Lim 2016 from Lim 2013.
                 - 0.0665*BMI + 0.0031*Age*DiaBP)
  preds <- 1 - exp(-exp(z/1.0805))
  return(tibble("Normotensive"=1-preds, "Hypertensive"=preds))
}

predict.TLGS_Koohi <- function(obj, newdata, type=NULL){
  Age <- newdata$Age
  Female <- abs(newdata$Sex_male - 1)
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  Smoking <- newdata$SmokingStatus_currently
  FamHypHist <- newdata$FamHypHist_yes
  BMI <- newdata$BMI
  
  z = log(11) - (17.0917 - 0.1279*Age + 0.1355*Female - 0.0322*SysBP
                 - 0.1113*DiaBP - 0.1912*Smoking - 0.2302*FamHypHist 
                 - 0.0294*BMI + 0.0013*Age*DiaBP)
  preds <- 1 - exp(-exp(z/0.6401))
  return(tibble("Normotensive"=1-preds, "Hypertensive"=preds))
}


predict.CAVAS_Namgung <- function(obj, newdata, type=NULL){
  Age <- newdata$Age
  Female <- abs(newdata$Sex_male - 1)
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  Smoking <- newdata$SmokingStatus_currently
  FamHypHist <- newdata$FamHypHist_yes
  BMI <- newdata$BMI
  
  z = log(11) - (13.5440 - 0.0801*Age - 0.0874*Female - 0.0366*SysBP
                 - 0.0605*DiaBP - 1.108*0.1906*FamHypHist #Increased by the fraction of individuals with two to one hypertensive parents, as for fr
                 - 0.0277*BMI + 0.0008*Age*DiaBP)
  preds <- 1 - exp(-exp(z/0.6908))
  return(tibble("Normotensive"=1-preds, "Hypertensive"=preds))
}

predict.F_CAVAS_Namgung <- function(obj, newdata, type=NULL){
  Age <- newdata$Age
  Female <- abs(newdata$Sex_male - 1)
  SysBP <- newdata$SysBP
  DiaBP <- newdata$DiaBP
  Smoking <- newdata$SmokingStatus_currently
  FamHypHist <- newdata$FamHypHist_yes
  BMI <- newdata$BMI
  
  z = log(11) - (13.5468 - 0.0803*Age - 0.0827*Female - 0.0366*SysBP
                 - 0.0606*DiaBP + 0.0146*Smoking - 1.108*0.1902*FamHypHist #Increased by the fraction of individuals with two to one hypertensive parents, as for fr
                 - 0.0277*BMI + 0.0008*Age*DiaBP)
  preds <- 1 - exp(-exp(z/0.6908))
  return(tibble("Normotensive"=1-preds, "Hypertensive"=preds))
}

chinese_risk_model=structure(list(), class="ChineseRiskModel_Chien")
chinese_risk_model_no_diabetes=structure(list(), class="ChineseRiskModel_DiabetesFreePop_Chien")
koges=structure(list(), class="KoGES_Lim2016")
tlgs=structure(list(), class="TLGS_Koohi")
cavas=structure(list(), class="CAVAS_Namgung")
f_cavas=structure(list(), class="F_CAVAS_Namgung")
logreg_reference <- structure(list(), class="LogisticRegressionReferenceModel")
high_normal_bp_rule <- structure(list(), class="HighNormalBPRule")
framingham <- structure(list(), class="FraminghamModel")
recalibrated_framingham <- structure(list(), class="RecalibratedFraminghamModel")
