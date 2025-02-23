#####################
# ANALYSE DE SURVIE #
#####################


library(readxl)
db <- read.csv("~/Desktop/Devoir #2 - Analyse Multi/DonneesDevoir2/LifeTimes.csv")
View(db)
#############

# Create a new dataframe with an inverted "Censored" variable
df_new <- db
df_new$Censored <- ifelse(df_new$Censored == 1, 0, 1)

# View the original and new data for comparison
head(db[, "Censored", drop = FALSE])  # Original dataframe
head(df_new[, "Censored", drop = FALSE])  # New dataframe with changes


library(survival)
library(broom)
library(hecmulti)
library(ggplot2)
library(gt)


# Estimateur de Kaplan-Meier
# La réponse "temps"est le temps de survie
# et l'indicateur de censure "censure" est
# "0" pour censuré à droite, "1" pour événement
kapm <-
  survfit(Surv(Time, Censored) ~ 1,
          conf.type = "log",
          data = df_new)

print(tidy(kapm$time[1800]))
summary(kapm)
quantile(kapm)
plot(kapm)
print(kapm, print.rmean = TRUE)

###############
## Graphique ## 
###############

library(survival)

# Plot the survival curves
plot(
  survfit(Surv(Time, Censored) ~ Plan, data = df_new),
  conf.int = FALSE,                      # No confidence intervals
  col = c("red", "blue", "green"),       # Colors for the groups
  lty = c(1, 2, 3),                      # Line types for the groups
  xlab = "Time (in days)",               # X-axis label
  ylab = "Survival Probability",         # Y-axis label
  main = "Survival Curves by Plan"       # Plot title
)

# Add gridlines for better readability
grid(col = "lightgray", lty = "dotted")

# Add a legend to distinguish between the groups
legend(
  "topright",                            # Position of the legend
  legend = c("Plan 1", "Plan 2", "Plan 3"), # Match the numeric plan labels
  col = c("red", "blue", "green"),       # Match the colors
  lty = c(1, 2, 3),                      # Match the line types
  title = "Plan Type",                   # Legend title
  cex = 0.8                              # Text size
)

#################################
## prob de survie à 1800 jours ##
#################################
plan_surv <-
  survfit(Surv(Time, Censored) ~ Plan,
          # conf.type = "log",
          data = df_new)

surv_1800 <- summary(plan_surv, times = 1800)




##############
## bo tablo ##
##############
library(dplyr)
# Create a data frame from the survival summary
surv_1800_table <- data.frame(
  Plan = surv_1800$strata,
  Time = surv_1800$time,
  N_Risk = surv_1800$n.risk,
  N_Event = surv_1800$n.event,
  Survival = surv_1800$surv,
  Lower_CI = surv_1800$lower,
  Upper_CI = surv_1800$upper
)

# Format survival probabilities as percentages
surv_1800_table <- surv_1800_table %>%
  mutate(Survival = scales::percent(Survival, accuracy = 0.1),
         Lower_CI = scales::percent(Lower_CI, accuracy = 0.1),
         Upper_CI = scales::percent(Upper_CI, accuracy = 0.1))
library(gt)

# Create a styled table
surv_1800_table %>%
  gt() %>%
  tab_header(
    title = "Survival at 1800 Days by Plan",
    subtitle = "Includes survival probabilities and confidence intervals"
  ) %>%
  fmt_number(
    columns = c(N_Risk, N_Event),
    decimals = 0
  ) %>%
  cols_label(
    Plan = "Plan",
    Time = "Time (Days)",
    N_Risk = "Number at Risk",
    N_Event = "Number of Events",
    Survival = "Survival Probability",
    Lower_CI = "Lower 95% CI",
    Upper_CI = "Upper 95% CI"
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels(everything())
  ) %>%
  tab_options(
    table.border.top.color = "black",
    table.border.bottom.color = "black",
    heading.title.font.size = 16,
    heading.subtitle.font.size = 12
  )

###############
## quartiles ##
###############

quantile(plan_surv)

########################
## bo tablo quartiles ##
########################
# Calculate quantiles for the survival object
quantile_data <- as.data.frame(quantile(plan_surv))


Plan <- c("Plan 1", "Plan 2", "Plan 3")
quantile_data <- cbind(Plan, quantile_data)
gt(quantile_data[1:4])


################
## bo grafike ##
################

## Fit the survival curves
surv_fit <- survfit(Surv(Time, Censored) ~ Plan, data = df_new)

# Plot the survival curves
plot(
  surv_fit,
  conf.int = FALSE,                      # No confidence intervals
  col = c("red", "blue", "green"),       # Colors for the groups
  lty = c(1, 2, 3),                      # Line types for the groups
  xlab = "Time (in days)",               # X-axis label
  ylab = "Survival Probability",         # Y-axis label
  main = "Survival Curves by Plan"       # Plot title
)

# Add gridlines for better readability
grid(col = "lightgray", lty = "dotted")

# Add a legend to distinguish between the groups
legend(
  "topright",                            # Position of the legend
  legend = c("Plan 1", "Plan 2", "Plan 3"),        # Names of the groups
  col = c("red", "blue", "green"),                # Match the colors
  lty = c(1, 2, 3),                         # Match the line types
  title = "Plan Type",                   # Legend title
  cex = 0.8                              # Text size
)

# Add horizontal lines and annotations for quartiles not reached
abline(h = 0.25, col = "blue", lty = 2)  # 25% survival probability line
text(
  x = max(df_new$Time) * 0.8, y = 0.27,  # Position for text
  labels = "75th percentile not reached (Plan 2 & Plan 3)",
  col = "blue", cex = 0.8
)



#################
# MODELE DE COX #
#################

###############################
## transformer Plan en facor ##
###############################

df_new$Plan<- as.factor(df_new$Plan)

################################
## Fitting the Cox regression ##
################################

cox1 <- coxph(Surv(Time, Censored) ~ Plan,
              data = df_new,
              ties = "exact")

###########################################
## Coefficients, tests et IC 95% de Wald ##
###########################################

summary(cox1)
library(gtsummary)
cox1 %>% tbl_regression(exponentiate = T) 


######################################
## Test du rapport de vraisemblance ##
######################################

Cox_apport_vraissemblance<- as.data.frame(car::Anova(cox1, type = 3))
ID <- c("NULL", "Plan")
Cox_apport_vraissemblance <- cbind(ID, Cox_apport_vraissemblance)
gt(Cox_apport_vraissemblance)


################################
## Fitting the Cox regression ##
## With Plan 2 as reference.  ##
################################
# Set the reference category for 'Plan' to 2
df_newref <- df_new
df_newref$Plan <- relevel(factor(df_newref$Plan), ref = "3")

cox2 <- coxph(Surv(Time, Censored) ~ Plan,
              data = df_newref,
              ties = "exact")

summary(cox2)

cox2 %>% tbl_regression(exponentiate = T) 
###############
## Postulats ##
###############

test_score_rprop <- cox.zph(cox2)

plot(test_score_rprop)



results <- as.data.frame(test_score_rprop$table)
Cox_table <- c( "Plan", "Global")
results <- cbind(Cox_table, results)
gt(results)

