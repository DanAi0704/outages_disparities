library(dplyr)
library(here)
library(xtable)
Outage<-readr::read_csv(here::here("Merged data","Outage_final.csv"))


# calculate summary statistics for a single variable
calculate_summary_single_var <- function(data, var_name) {
  
  summary_stats <- data %>%
    group_by(YEAR) %>%
    summarise(
      n = n(),
      Mean = round(mean(!!sym(var_name), na.rm = TRUE), 2),
      Std = round(sd(!!sym(var_name), na.rm = TRUE), 2),
      Min = round(min(!!sym(var_name), na.rm = TRUE), 2),
      Max = round(max(!!sym(var_name), na.rm = TRUE), 2)
    )
  
  # Generate LaTeX code
  latex_table <- xtable(summary_stats, caption = paste(var_name, "Summary Statistics by Year"))
  print(latex_table, type = "latex", digits = 2)
  
  return(summary_stats)
}


# Calculate summary statistics for each variable

var_list <- c("SAIDI_Nonweather","SAIDI_Weather","SAIDI","English_Dep", "Poverty", "Minority_Racial",
              "Population_Density","Road_Density","TreeCover")
summary_list <- lapply(var_list, function(var) calculate_summary_single_var(Outage, var))


