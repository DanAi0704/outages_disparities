library(dplyr)
library(here)
library(fixest)
library(xtable)
library(stargazer)
Outage_final<-readr::read_csv(here::here("Merged data","Outage_final.csv"))

Outage_t<-Outage_final%>%
  mutate(SAIDI_Nonweather_t=SAIDI_Nonweather+0.000000000000000001,
         SAIDI_Weather_t=SAIDI_Weather+0.000000000000000001,
         SAIDI=SAIDI+0.00000000000000000001)

#===========================================================================================
#Main results
#============================================================================================

run_feols <- function(dependent_var, data, fixed_effects, clustering_var) {
  
  formula_str <- paste0("log(", dependent_var, ") ~ Minority_Racial + English_Dep + Poverty + Population_Density + Road_Density+TreeCover")
  
  # No fixed effects
  fe_none <- feols(as.formula(formula_str), data = data, cluster = clustering_var)
  formula_str_with_fe <- paste0(formula_str, "|")
  
  # County fixed effects
  fe_county <- feols(as.formula(paste(formula_str_with_fe, fixed_effects[1])), data = data, cluster = clustering_var)
  
  # Year fixed effects
  fe_time <- feols(as.formula(paste(formula_str_with_fe, fixed_effects[2])), data = data, cluster = clustering_var)
  
  # County and year fixed effects
  fe_twoway <- feols(as.formula(paste(formula_str_with_fe, paste(fixed_effects, collapse = "+"))), data = data, cluster = clustering_var)
  
  # Results and latex table
  et1 <- etable(fe_none, fe_county, fe_time, fe_twoway, signif.code = c("***" = 0.05, "**" = 0.1, "*" = 0.15))
  et2 <- etable(fe_none, fe_county, fe_time, fe_twoway, signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1), tex = TRUE)
  
  return(list(et1, et2))
}

# Variables
data = Outage_t
clustering_var = ~GEOID
fixed_effects = c("GEOID", "YEAR")

# For SAIDI
results_T <- run_feols("SAIDI", data, fixed_effects, clustering_var)
print(results_T)
# For Non-weather related SAIDI
results_N <- run_feols("SAIDI_Nonweather_t", data, fixed_effects, clustering_var)
print(results_N)
# For Weather-related SAIDI
results_W <- run_feols("SAIDI_Weather_t", data, fixed_effects, clustering_var)
print(results_W)

#=====================================================================
#Robust test
#=====================================================================
data = Outage_t
clustering_var = ~GEOID
fixed_effects = c("UtilityID", "YEAR")

# For SAIDI
results_robust_T <- run_feols("SAIDI", data, fixed_effects, clustering_var)
print(results_robust_T)
# For Non-weather related SAIDI
results_robust_N <- run_feols("SAIDI_Nonweather_t", data, fixed_effects, clustering_var)
print(results_robust_N)
# For Weather-related SAIDI
results_robust_W <- run_feols("SAIDI_Weather_t", data, fixed_effects, clustering_var)
print(results_robust_W)

#=====================================================================================
#step wise regression
#===================================================================================
stepwise_feols_models <- function(dependent_var, indep_var_sets, data, cluster_var = "~GEOID") {
  models <- list()
  
  for (i in 1:length(indep_var_sets)) {
    formula_str <- paste0("log(", dependent_var, ") ~ ", indep_var_sets[[i]], "|GEOID+YEAR")
    model <- feols(as.formula(formula_str), data = data, cluster = cluster_var)
    models[[i]] <- model
  }
  
  # Print etable
  do.call(etable, c(models, list(signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1))))
  do.call(etable, c(models, list(signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1), tex = TRUE)))
}

# Define independent variable sets
indep_vars_T <- list(
  "Minority_Racial+Population_Density+Road_Density+TreeCover",
  "English_Dep+Population_Density+Road_Density+TreeCover",
  "Poverty+Population_Density+Road_Density+TreeCover",
  "Minority_Racial+English_Dep+Population_Density+Road_Density+TreeCover",
  "Minority_Racial+English_Dep+Poverty+Population_Density+Road_Density+TreeCover"
)

#  for total SAIDI
results_step_T<-stepwise_feols_models("SAIDI", indep_vars_T, Outage_t)

# for Non-weather related SAIDI
stepwise_feols_models("SAIDI_Nonweather_t", indep_vars_T, Outage_t)

# for weather related SAIDI
stepwise_feols_models("SAIDI_Weather_t", indep_vars_T, Outage_t)






