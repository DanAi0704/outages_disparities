library(dplyr)
library(tidyr)
library(here)
library(ggplot2)
library(scales)
library(forcats)
library(broom)

Outage_month<-readr::read_csv(here::here("Data_Raw","monthly_state_saidi_v3.csv"))
Outage<-readr::read_csv(here::here("Merged data","Outage_final.csv"))

Outage_month<-Outage_month%>%
  filter(year!=2022)

Outage_t<-Outage%>%
  mutate(SAIDI_Nonweather_t=SAIDI_Nonweather+0.000000000000000001,
         SAIDI_Weather_t=SAIDI_Weather+0.000000000000000001,
         SAIDI=SAIDI+0.00000000000000000001)
#========================================================================================
#short time trend
#========================================================================================
# monthly country level data
monthly_country_saidi_v3<-Outage_month%>%
  group_by(year,month)%>%
  summarize(
    CustomerHoursOutTotal = sum(CustomerHoursOutTotal),
    CustomerHoursOutTotal_only_storm = sum(CustomerHoursOutTotal_only_storm),
    CustomerHoursOutTotal_no_storm = sum(CustomerHoursOutTotal_no_storm),
    CustomersTracked = sum(CustomersTracked),
    saidi = sum(CustomerHoursOutTotal) / sum(CustomersTracked),
    saidi_only_storm = sum(CustomerHoursOutTotal_only_storm) / sum(CustomersTracked),
    saidi_no_storm = sum(CustomerHoursOutTotal_no_storm) / sum(CustomersTracked)
  )

monthly_country_saidi_v3 <- monthly_country_saidi_v3 %>%
  mutate(Date = as.Date(sprintf("%d-%02d-01", year, month), format = "%Y-%m-%d"))

#scalling 
monthly_country_saidi_v3$CustomerHoursOutTotal<-round(monthly_country_saidi_v3$CustomerHoursOutTotal,-6)/1000000#time trend 

#time trend
ggplot(monthly_country_saidi_v3, aes(x = Date, y = CustomerHoursOutTotal)) +
  geom_line(color = "red") +
  theme_classic() +
  theme(
    axis.line.x = element_line(colour = 'black', size=0, linetype='solid'),
    axis.line.y = element_line(colour = 'black', size=0, linetype='solid'),
    panel.grid.major.y = element_line(),
    panel.grid.minor.y = element_line(),
    panel.border = element_rect(colour = "black", fill=NA, size=1),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  
    axis.text.x = element_text(color="black", angle = 45, hjust = 1, size = 12, face = "bold"), 
    axis.text.y = element_text(color="black", size = 12, face = "bold"),
    axis.title.x = element_text(size = 12, face = "bold"),  
    axis.title.y = element_text(size = 12, face = "bold")   
  ) +  
  labs(x = "Year", y = "Hours (Millions)", title = "Total Customer Outage Hours") +
  scale_x_date(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))

#=============================================================================================
# mean and standard deviation of SAIDI by year
#=============================================================================================
Outage_county<-readr::read_csv(here::here("Data_Raw","yearly_county_saidi_v3.csv"))

Outage_final<-Outage_county%>%
  filter(year!=2022)%>%
  mutate(SAIDI=saidi,YEAR=year,SAIDI_Weather=saidi_only_storm, SAIDI_Nonweather=saidi_no_storm)

# Combine all summaries into one dataframe
outage_summary_combined <- bind_rows(
  Outage_final %>%
    group_by(YEAR) %>%
    summarise(mean_saidi = mean(SAIDI, na.rm = TRUE),
              sd_saidi = sd(SAIDI, na.rm = TRUE)) %>%
    mutate(Type = "Total SAIDI"),
  
  Outage_final %>%
    group_by(YEAR) %>%
    summarise(mean_saidi = mean(SAIDI_Weather, na.rm = TRUE),
              sd_saidi = sd(SAIDI_Weather, na.rm = TRUE)) %>%
    mutate(Type = "Weather-related SAIDI"),
  
  Outage_final %>%
    group_by(YEAR) %>%
    summarise(mean_saidi = mean(SAIDI_Nonweather, na.rm = TRUE),
              sd_saidi = sd(SAIDI_Nonweather, na.rm = TRUE)) %>%
    mutate(Type = "Non-weather-related SAIDI")
) %>%
  ungroup()

# Type factor
outage_summary_combined$Type <- fct_relevel(outage_summary_combined$Type,
                                            "Total SAIDI", "Weather-related SAIDI", "Non-weather-related SAIDI")
# Calculate  min and max including standard deviation for y-axis limits
global_min <- min(outage_summary_combined$mean_saidi - outage_summary_combined$sd_saidi, na.rm = TRUE)
global_max <- max(outage_summary_combined$mean_saidi + outage_summary_combined$sd_saidi, na.rm = TRUE)
y_breaks <- max(abs(global_min), abs(global_max))
y_breaks <- ceiling(y_breaks / 25) * 25 # Round up to the nearest 25

# Plotting all summaries in one plot 
  ggplot(outage_summary_combined, aes(x = YEAR, y = mean_saidi, color = Type)) +
    geom_errorbar(aes(ymin = mean_saidi - sd_saidi, ymax = mean_saidi + sd_saidi), width = 0.1) +
    geom_point(size = 3) +
    geom_line() +
    theme_minimal() +
    labs(y = "SAIDI", x = "Year") +  
    scale_color_manual(values = c("Total SAIDI" = "#FFA500", 
                                  "Weather-related SAIDI" = "#00BFFF", 
                                  "Non-weather-related SAIDI" = "#008000")) +
    facet_wrap(~Type, ncol = 1, scales = 'fixed') + 
    theme(
      legend.position = "top",  
      panel.grid.major.x = element_blank(), 
      panel.grid.minor.x = element_blank(), 
      panel.grid.minor.y = element_blank(), 
      axis.line = element_line(color = "black"), 
      axis.text.x = element_text(color = "black", size = 10, face = "bold"),  
      axis.text.y = element_text(color = "black", size = 10, face = "bold"),  
      axis.title.x = element_text(size = 12, face = "bold"), 
      axis.title.y = element_text(size = 12, face = "bold"), 
      legend.box.background = element_rect(color = "black", size = 0.5, linetype = "solid"),
      axis.ticks.length = unit(0.25, "cm"),
      strip.text = element_text(face = "bold")
    ) +
    scale_y_continuous(breaks = seq(-y_breaks, y_breaks, by = 25), limits = c(-y_breaks, y_breaks)) 
  

#=====================================================================================
# plot coefficient bar chart
#=====================================================================================
  m1<-feols(log(SAIDI)~English_Dep+Poverty+ Minority_Racial+
              + Population_Density+Road_Density+TreeCover|YEAR+GEOID,cluster=~GEOID,data = Outage_t)
  m2<-feols(log(SAIDI_Weather_t)~English_Dep+Poverty+ Minority_Racial+
              + Population_Density+Road_Density+TreeCover|YEAR+GEOID,cluster=~GEOID,data = Outage_t)
  m3<-feols(log(SAIDI_Nonweather_t)~English_Dep+Poverty+ Minority_Racial+
              + Population_Density+Road_Density+TreeCover|YEAR+GEOID,cluster=~GEOID,data = Outage_t)
  
  # Extract the coefficients in a tidy format 
  m1_tidy <- tidy(m1) %>%
    filter(term %in% c("Poverty", "Minority_Racial")) %>%
    mutate(model = "Total SAIDI")
  m2_tidy <- tidy(m2) %>%
    filter(term %in% c("Poverty", "Minority_Racial")) %>%
    mutate(model = "Weather-related SAIDI")
  m3_tidy <- tidy(m3) %>%
    filter(term %in% c("Poverty", "Minority_Racial")) %>%
    mutate(model = "Non-weather-related SAIDI")
  
  # Combine the coefficients into one data frame
  coefficients <- bind_rows(m1_tidy, m2_tidy, m3_tidy) %>%
    mutate(model = factor(model, levels = c("Total SAIDI", "Weather-related SAIDI", "Non-weather-related SAIDI")))
  
  # Plot the coefficients as a bar chart
  ggplot(coefficients, aes(x = term, y = estimate, fill = model)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.6) +
    geom_text(aes(label = paste("SE =", sprintf("%.2f", std.error))),
              position = position_dodge(width = 0.6),
              vjust = -0.5, color = "black", size = 3) +
    geom_text(aes(label = paste("P-value =", sprintf("%.2f", p.value))),
              position = position_dodge(width = 0.6),
              vjust = -2.5, color = "black", size = 3) +
    theme_minimal() +
    labs(x = "", y = "Coefficient Estimate", fill = "Model") +
    scale_x_discrete(labels = c("Poverty" = "Percent Poverty", "Minority_Racial" = "Percent Racial Minorities")) +
    theme(
      legend.position = "top",
      legend.title = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey80"),
      axis.line = element_line(color = "black"),
      axis.text.x = element_text(color="black",size = 14, face = "bold"),  
      axis.text.y = element_text(color="black",size = 14, face = "bold"),  
      axis.title.x = element_text(color="black",size = 16, face = "bold"), 
      axis.title.y = element_text(color= "black",size = 16, face = "bold"),
      legend.box.background = element_rect(color = "black", size = 0.5, linetype = "solid"),
      axis.ticks.margin = unit(0, "points")
    ) +
    scale_fill_manual(values = c("Total SAIDI" = "#FFA500", "Weather-related SAIDI" = "#00BFFF", "Non-weather-related SAIDI" = "#008000")) +
    scale_y_continuous(
      breaks = seq(0, max(coefficients$estimate, na.rm = TRUE), by = 0.05),  
      expand = expansion(mult = c(0.008, 0.05))  
    )
  

  
  
  