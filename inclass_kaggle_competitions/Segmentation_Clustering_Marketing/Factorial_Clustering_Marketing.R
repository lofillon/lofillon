############
# DEVOIR 2 #
############



#######################
# ANALYSE FACTORIELLE #
#######################

library(readxl)
perso <- read.csv("~/Desktop/Devoir #2 - Analyse Multi/DonneesDevoir2/TestPersonnalite.csv")
View(perso)

######

library(gt)
summary(perso)

perso_noID <- perso[,-1]

# Ajuster le modèle factoriel par maximum de vraisemblance
fa3 <- factanal(x = perso_noID,
                factors = 3,
                rotation = "none")
print(fa4$loadings,
      cutoff = 0.3)

fa4 <- factanal(x = perso_noID,
                factors = 4,
                rotation = "promax")
print(fa4$loadings,
      cutoff = 0.45)

fa5 <- factanal(x = perso_noID,
                factors = 5,
                rotation = "promax")
print(fa5$loadings,
      cutoff = 0.45)



#############################
## choix du nb de facteurs ##
#############################
library(hecmulti)
adj <- ajustement_factanal(
  covmat = cov(perso_noID),
  factors = 1:6,
  n.obs = nrow(perso_noID)) 
plot(adj$BIC)
plot(adj$AIC)
gt(adj)

# Cohérence interne (alpha de Cronbach)
alphaC(perso_noID[,c("Talkative","Outgoing","Forgiving")])
alphaC(perso_noID[,c("Thorough","Efficient","Energetic")])
alphaC(perso_noID[,c("Helpful_unselfish","Considerate","Cooperative")])
alphaC(perso_noID[,c("Relaxed","Emotionally_stable","Sophisticated_in_arts")])
alphaC(perso_noID[,c("Easily_distracted","Moody","Curious")])


# Création des échelles
perso_noID$Extraversion <- rowMeans(perso_noID[,c("Talkative","Outgoing","Forgiving")])
perso_noID$Conscientiousness <- rowMeans(perso_noID[,c("Thorough","Efficient","Energetic")])
perso_noID$Agreeableness <- rowMeans(perso_noID[,c("Helpful_unselfish","Considerate","Cooperative")])
perso_noID$Openness <- rowMeans(perso_noID[,c("Relaxed","Emotionally_stable","Sophisticated_in_arts")])
perso_noID$Neuroticism <- rowMeans(perso_noID[,c("Easily_distracted","Moody","Curious")])

# Création des échelles Avec ID
perso$Extraversion <- rowMeans(perso[,c("Talkative","Outgoing","Forgiving")])
perso$Conscientiousness <- rowMeans(perso[,c("Thorough","Efficient","Energetic")])
perso$Agreeableness <- rowMeans(perso[,c("Helpful_unselfish","Considerate","Cooperative")])
perso$Openness <- rowMeans(perso[,c("Relaxed","Emotionally_stable","Sophisticated_in_arts")])
perso$Neuroticism <- rowMeans(perso[,c("Easily_distracted","Moody","Curious")])



############################
# ANALYSE DE REGROUPEMENTS #
############################

library(tidyLPA)
library(dplyr)
library(ggplot2)
library(hecmulti)
library(gt)
##########################


set.seed(60602)
kmoy <- list()
ngmax <- 10L
for(i in seq_len(ngmax)){
  kmoy[[i]] <- kmeans(perso[17:21],
                      centers = i,
                      nstart = 10)
}

###########################
scd <- sapply(kmoy, function(x){x$tot.withinss})

###########################
#graphiques#
homogene <- homogeneite(scd)
bic_kmoy <- sapply(kmoy, BIC)

gt(homogene)


plot(homogene$SCD, type = "b", ylab = "Somme du Carré des Distances", xlab = "Nombre de regroupements") 
abline(v = 5, col = "blue", lty = 2)  # Add vertical line at x = 5
points(5, homogene$SCD[5], col = "red", pch = 19, cex = 1.5)  # Highlight point for cluster 5

# Plot for Rc
plot(homogene$Rc, type = "b", ylab = "R-Carré", xlab = "Nombre de regroupements")
abline(v = 5, col = "blue", lty = 2)   # Add vertical line at x = 5
points(5, homogene$Rc[5], col = "red", pch = 19, cex = 1.5)  # Highlight point for cluster 5

# Plot for Rcsp
plot(homogene$Rcsp, type = "b", ylab = "R-Carré Semi-Partiel", xlab = "Nombre de regroupements") 
abline(v = 5, col = "blue", lty = 2)   # Add vertical line at x = 5
points(5, homogene$Rcsp[5], col = "red", pch = 19, cex = 1.5)  # Highlight point for cluster 5
###########################

kmoy3 <- kmoy[[3]]
profiles3 <- perso[17:21] |>
  group_by(groupe = kmoy3$cluster) |>
  summarise_all(mean)

kmoy4 <- kmoy[[4]]
profiles4 <- perso[17:21] |>
  group_by(groupe = kmoy4$cluster) |>
  summarise_all(mean)

kmoy5 <- kmoy[[5]]
profiles5 <- perso[17:21] |>
  group_by(groupe = kmoy5$cluster) |>
  summarise_all(mean)

kmoy6 <- kmoy[[6]]
profiles6 <- perso[17:21] |>
  group_by(groupe = kmoy6$cluster) |>
  summarise_all(mean)


kmoy3$size
###############################

library(ggplot2)
library(tidyr)
# Reshape data to long format


#graph avec 3
data_long3 <- profiles3 %>%
  pivot_longer(cols = -groupe, names_to = "Trait", values_to = "Value")

ggplot(data_long3, aes(x = Trait, y = Value, group = as.factor(groupe), color = as.factor(groupe))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Personality Traits by Group", x = "Trait", y = "Score", color = "Group") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#graph avec 4
data_long4 <- profiles4 %>%
  pivot_longer(cols = -groupe, names_to = "Trait", values_to = "Value")

ggplot(data_long4, aes(x = Trait, y = Value, group = as.factor(groupe), color = as.factor(groupe))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Personality Traits by Group", x = "Trait", y = "Score", color = "Group") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#graph avec 5
data_long5 <- profiles5 %>%
  pivot_longer(cols = -groupe, names_to = "Trait", values_to = "Value")

ggplot(data_long5, aes(x = Trait, y = Value, group = as.factor(groupe), color = as.factor(groupe))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Personality Traits by Group", x = "Trait", y = "Score", color = "Group") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#graph avec 6
data_long6 <- profiles6 %>%
  pivot_longer(cols = -groupe, names_to = "Trait", values_to = "Value")

ggplot(data_long6, aes(x = Trait, y = Value, group = as.factor(groupe), color = as.factor(groupe))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Personality Traits by Group", x = "Trait", y = "Score", color = "Group") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#################################
## interprétation des segments ##
#################################


gt(profiles5)



# Ajouter la colonne "groupe" à la base de données `perso`
df_cluster <- perso |>
  mutate(groupe = kmoy5$cluster)

df_cluster <- df_cluster %>% rename(ID = ID.ID5.)

head(df_cluster)
##########################################
## importation des bases de donnees.    ##
##########################################
DonationHistory <- read.csv("~/Desktop/Devoir #1/Devroi1-Fundraising/DonationHistory.csv")

MembersList <- read.csv("~/Desktop/Devoir #1/Devroi1-Fundraising/MembersList.csv")

NewsletterRead <- read.csv("~/Desktop/Devoir #1/Devroi1-Fundraising/NewsletterRead.csv")

SocialNetworkUsage <- read.csv("~/Desktop/Devoir #1/Devroi1-Fundraising/SocialNetworkUsage.csv")

############################################
## création de la base de donnée commune  ##
############################################
library(tidyr)
NewsletterRead <- NewsletterRead |>
  mutate(frq_ouverture = rowSums(select(NewsletterRead, 2:13), na.rm = TRUE))

DonationHistoryWide <- 
  pivot_wider(DonationHistory, names_from = Yr, values_from = Amount, values_fill = 0)
View(DonationHistoryWide)

db <- full_join(df_cluster, MembersList, by = "ID")
db <- full_join(db, DonationHistoryWide, by = "ID")

# Inclure uniquement la colonne 14 avec `email`
selected_col <- NewsletterRead |>
  select(email, total_reads = 14)  # Remplacez `14` par le nom ou index de la colonne
# Fusionner les bases
db <- full_join(db, selected_col, by = "email")

db_complet <- db |>
  filter(complete.cases(groupe))

db_complet <- db_complet %>%
  mutate(across(32:43, ~ replace_na(., 0)))


colnames(db_complet)

#############################################################
## création des variables pertinentes pour identifier les  ##
## différences entre les segments.                         ##
#############################################################
db_complet <- db_complet |>
  mutate(dons_total = rowSums(select(db_complet, 32:43), na.rm = TRUE))

db_analyse <- db_complet[, c(22, 26:31, 44:45)]


####################################################################
## Regression multinomiale pour décrire les compts des segments   ##
####################################################################
library(nnet)
library(hecmulti)
library(gtsummary)
library(ggstats)
library(dplyr)

str(db_scaled)
str(db_analyse)

db_analyse$Woman <- as.factor(db_analyse$Woman)
db_analyse$Joined <- as.factor(db_analyse$Woman)



db_scaled <- db_analyse %>%
  mutate(across(where(is.numeric) & -groupe, scale))

description <- nnet::multinom(
  groupe ~ .,
  data = db_scaled,
  Hess = TRUE,
  trace = FALSE
)
colnames(db_scaled)

description %>%
  tbl_regression() %>%
  bold_p(t = .1)


description %>%
  ggcoef_multinom(exponentiate = TRUE)


description %>%
  ggcoef_multinom(exponentiate = TRUE, 
                  type = "f")

ggeffect(description)

library(ggeffects)
ggeffect(description) %>%
  plot()
# %>%
#   cowplot::plot_grid(plotlist = ., ncol = 4)
ggpredict(description)%>%
  plot()


library(jmv)
descriptives(db_analyse,splitBy = "groupe", mode = TRUE, missing = FALSE, n = FALSE, min = FALSE, max = FALSE, )


library(gt)
library(ggplot2)

## table de frequence pour l'education
db_analyse %>%
  group_by(Education) %>%
  summarize(N = n()) %>%
  mutate(Frequency = N/sum(N),
         Frequency = round(Frequency, 2)) %>%
  gt()




##Woman
ggplot(data = db_analyse, aes(x = groupe, group = Woman, fill = Woman)) +
  geom_bar(position = "fill") +/
  ylab("Proportion") +
  stat_count(
    geom = "text",
    aes(label = scales::percent(..count.. / tapply(..count.., ..PANEL.., sum)[..PANEL..], accuracy = 0.1)),
    position = position_fill(vjust = 0.5),
    colour = "white"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

##education
ggplot(data = db_analyse, aes(x = groupe, group = Education, fill = Education)) +
  geom_bar(position = "fill") +
  ylab("Proportion") +
  stat_count(
    geom = "text",
    aes(label = scales::percent(..count.. / tapply(..count.., ..PANEL.., sum)[..PANEL..], accuracy = 0.1)),
    position = position_fill(vjust = 0.5),
    colour = "white"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

# City
ggplot(data = db_analyse, aes(x = groupe, group = City, fill = City)) +
  geom_bar(position = "fill") +
  ylab("Proportion") +
  stat_count(
    geom = "text",
    aes(label = scales::percent(..count.. / tapply(..count.., ..PANEL.., sum)[..PANEL..], accuracy = 0.1)),
    position = position_fill(vjust = 0.5),
    colour = "white"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()




ggplot(db_analyse, aes(x = Age, fill = groupe, group = groupe)) +
  geom_density(alpha = 0.3) +  # Semi-transparent density curves
  labs(
    title = "Density Plot of Age by Group",
    x = "Age",
    y = "Density",
    fill = "Group"
  ) +
  theme_minimal()+
  theme_ggeffects()



















