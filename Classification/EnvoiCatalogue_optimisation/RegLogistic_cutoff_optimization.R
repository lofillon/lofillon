#################################
## creation d'un echantillon.  ##
## d'apprentissage, validation ##
## et test                     ##
#################################

# Save all objects in the current workspace to an .RData file
save.image(file = "all_objects_Devoir_1_15oct.RData")

library(dplyr)
library(tidyr)
library(dplyr)
library(tidyr)

# Apply the transformation
db1_nomiss <- db1 %>%
  mutate(across(all_of(year_columns), ~ replace_na(., 0)))

# anciennete
db1_nomiss <- mutate(db1_nomiss, ancientete = 2024 - db1_nomiss$Joined)

# montant total donne aucours des années
db1_nomiss <- db1_nomiss %>%
  mutate(dontotal = rowSums(select(., all_of(year_columns)), na.rm = TRUE))

# sans l'année 2023
year_columns_excluding_2023 <- year_columns[year_columns != "2023"]
db1_nomiss <- db1_nomiss %>%
  mutate(don_no2023 = rowSums(select(., all_of(year_columns_excluding_2023)), na.rm = TRUE))

# don en 2022
db1_nomiss <- db1_nomiss %>%
  mutate(don_2022 = `2022` > 0)

# don en 2021
db1_nomiss <- db1_nomiss %>%
  mutate(don_2021 = `2021` > 0)

# don en 2023
db1_nomiss <- db1_nomiss %>%
  mutate(don_2023 = `2023` > 0)

# fréquence d'ouverture de la newsletter
db1_nomiss <- db1_nomiss %>%
  mutate(frq_ouverture = rowSums(select(., 23:34), na.rm = TRUE))

# Chargement du package nécessaire
set.seed(123)  # Pour la reproductibilité
# Taille de l'échantillon de test (par exemple 20%)
test_ratio <- 0.99
# Nombre total de lignes dans la base de données 'db11'
n <- nrow(db1_nomiss)
# Sélection aléatoire des indices pour l'échantillon test
test_indices <- sample(1:n, size = floor(test_ratio * n))
# Création de la variable 'test' dans la base de données
db1_nomiss$test <- 0  # Initialisation de la variable à 0 (par défaut entraînement/validation)
db1_nomiss$test[test_indices] <- 1  # Les indices sélectionnés sont marqués comme échantillon de test (1)
# Vérification de la répartition
table(db1$test)
# Affichage des dimensions des ensembles
cat("Nombre d'échantillons de test :", sum(db1_nomiss$test == 1), "\n")
cat("Nombre d'échantillons d'entraînement/validation :", sum(db1_nomiss$test == 0), "\n")

set.seed(60602)
# Diviser les bases de données 
# en échantillons d'apprentissage
# et de validation

valid_glm <- db1_nomiss[db1_nomiss$test == 1,] |>
  dplyr::select(! c(`2023`, test))
appr_glm <- db1_nomiss[db1_nomiss$test == 0, ] |>
  dplyr::select(! c(`2023`, test))

valid_lm <- db1_nomiss[db1_nomiss$test == 1,] |>
  dplyr::select(! c(don_2023, test))
appr_lm <- db1_nomiss[db1_nomiss$test == 0, ] |>
  dplyr::select(! c(don_2023, test))

###############################################
## définition des formules du modèle complet ##
###############################################

# Formule du modèle avec toutes les interactions 
# d'ordre 2 (.^2) et les termes quadratiques I(x^2)
var_principales_glm <-  (don_2023 ~Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture )
formule_glm <-  formula(don_2023 ~
                          (Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture 
                             # +
                             # `2012` + `2013` + `2014` + `2015` + `2016` + `2017` + `2018` + `2019` +
                             # `2020` + `2021` + `2022` 
                           ) +
                          
                          (Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture 
                             # +
                             # `2012` + `2013` + `2014` + `2015` + `2016` + `2017` + `2018` + `2019` +
                             # `2020` + `2021` + `2022`
                          )^2 +  
                          
                          I(Age^2) + I(Salary^2) + I(ancientete^2) + I(don_no2023^2) + I(frq_ouverture^2) 
                        # +
                        #   I(`2012`^2) + I(`2013`^2) + I(`2014`^2) + I(`2015`^2) + I(`2016`^2) + I(`2017`^2) +
                        #   I(`2018`^2) + I(`2019`^2) + I(`2020`^2) + I(`2021`^2) + I(`2022`^2)
)
formule_glm_no_interaction <-  formula(don_2023 ~
                          (Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture 
                           # +
                           # `2012` + `2013` + `2014` + `2015` + `2016` + `2017` + `2018` + `2019` +
                           # `2020` + `2021` + `2022` 
                          # ) 
                           +
                          # 
                          # (Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture 
                          #  +
                          #    # `2012` + `2013` + `2014` + `2015` + `2016` + `2017` + `2018` + `2019` +
                          #    # `2020` + `2021` + `2022`
                          # )^2 +  
                          
                          I(Age^2) + I(Salary^2) + I(ancientete^2) + I(don_no2023^2) + I(frq_ouverture^2) )
                        # +
                        #   I(`2012`^2) + I(`2013`^2) + I(`2014`^2) + I(`2015`^2) + I(`2016`^2) + I(`2017`^2) +
                        #   I(`2018`^2) + I(`2019`^2) + I(`2020`^2) + I(`2021`^2) + I(`2022`^2)
)
formule_lm <-  formula(`2023` ~
                         (Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture 
                          # +
                          #   `2012` + `2013` + `2014` + `2015` + `2016` + `2017` + `2018` + `2019` +
                          #   `2020` + `2021` + `2022` 
                          ) +
                         
                         (Woman + Age + Salary + Education + City + ancientete + don_no2023 + frq_ouverture
                          # +
                          #   `2012` + `2013` + `2014` + `2015` + `2016` + `2017` + `2018` + `2019` +
                          #   `2020` + `2021` + `2022`
                         )^2 +  
                         
                         I(Age^2) + I(Salary^2) + I(ancientete^2) + I(don_no2023^2) + I(frq_ouverture^2) 
                       # + I(`2012`^2) + I(`2013`^2) + I(`2014`^2) + I(`2015`^2) + I(`2016`^2) + I(`2017`^2) +
                       #   I(`2018`^2) + I(`2019`^2) + I(`2020`^2) + I(`2021`^2) + I(`2022`^2)
)

# reg_log_mod <- glm(formule_glm, data = appr, family = binomial(link = "logit"))
# summary(reg_log_mod)
# lm_mod <- lm(formule_lm, data = db1)
# summary(lm_mod)

###################################
## Lasso pour le modèle binomial ##
###################################

# Sélection par LASSO
library(glmnet)
# Paramètre de pénalité déterminé par 
# validation croisée à partir d'un vecteur
# de valeurs candidates
lambda_seq <- seq(from = 0.1,
                  to = 10, 
                  by = 0.01)
cv_output <- 
  glmnet::cv.glmnet(x = as.matrix(appr_glm[,5:40]), 
                    y = appr_glm$don_2023, 
                    alpha = 1, 
                    lambda = lambda_seq)
plot(cv_output)

# On réestime le modèle avec la pénalité
lambopt <- cv_output$lambda.min # ou lambda.1se
lasso_best <- 
  glmnet::glmnet(
    x = as.matrix(appr_glm[,5:40]),
    y = appr_glm$don_2023,
    alpha = 1, 
    lambda = lambopt)
# Prédictions et calcul de l'EQM
# On pourrait remplacer `newx` par 
# d'autres données (validation externe)
pred <- predict(lasso_best, 
                s = lambopt, 
                newx = as.matrix(appr_glm[,5:40]))
eqm_lasso <- mean((pred - appr_glm$don_2023)^2)

print(eqm_lasso)


stopCluster(cl)

###################################
## Lasso pour le modèle linéaire ##
###################################

# Sélection par LASSO
library(glmnet)
# Paramètre de pénalité déterminé par 
# validation croisée à partir d'un vecteur
# de valeurs candidates
lambda_seq <- seq(from = 0.1,
                  to = 10, 
                  by = 0.01)
cv_output <- 
  glmnet::cv.glmnet(x = as.matrix(appr_lm[,5:40]), 
                    y = appr_lm$`2023`, 
                    alpha = 1, 
                    lambda = lambda_seq)
plot(cv_output)

# On réestime le modèle avec la pénalité
lambopt <- cv_output$lambda.min # ou lambda.1se
lasso_best <- 
  glmnet::glmnet(
    x = as.matrix(appr_lm[,5:40]),
    y = appr_lm$`2023`,
    alpha = 1, 
    lambda = lambopt)
# Prédictions et calcul de l'EQM
# On pourrait remplacer `newx` par 
# d'autres données (validation externe)
pred <- predict(lasso_best, 
                s = lambopt, 
                newx = as.matrix(appr_lm[,5:40]))
eqm_lasso <- mean((pred - appr_lm$`2023`)^2)

print(eqm_lasso)


# Extract coefficients for the model with lambda.min
coef_lasso_min <- coef(lasso_best, s = lambopt)  # Use lambda.min or lambopt
selected_vars_min <- coef_lasso_min[coef_lasso_min != 0]  # Select non-zero coefficients

# Get the variable names (excluding the intercept)
selected_variable_names_min <- rownames(coef_lasso_min)[coef_lasso_min[, 1] != 0]
selected_variable_names_min <- selected_variable_names_min[selected_variable_names_min != "(Intercept)"]

# Print the selected variables for lambda.min
print("Selected variables for lambda.min:")
print(selected_variable_names_min)


############################################
##### Modélisation et point de coupure #####
############################################

# Nouvelles bases de données avec toutes ces variables
# On retire la première colonne (1, ordonnée à l'origine)
appr_c <- data.frame(
  cbind(model.matrix(formule_glm, data = appr_glm)[,-1]),
  y = as.integer(appr_glm$don_2023))
valid_c <- data.frame(
  cbind(model.matrix(formule_glm, data = valid_glm)[,-1]),
  y = as.integer(valid_glm$don_2023))
valid_2023 <- with(db1_nomiss, `2023`[test == 1L])

# Ajustement des différents modèles

# Modèle avec toutes les variables principales
base <- glm(var_principales_glm, 
            data = appr_glm, 
            family = binomial)

# Calcul du point de coupe optimal
#  (par validation croisée)
base_coupe <- hecmulti::select_pcoupe(
  modele = base, 
  c00 = 0, 
  c01 = 0, 
  c10 = -10, 
  c11 = 57)
# Performance sur données de validation
base_pred <- 
  predict(object = base, 
          newdata = valid_glm, 
          type = "response") > base_coupe$optim
base_perfo <- 
  -10*sum(base_pred) + 
  sum(valid_2023[base_pred], na.rm = TRUE)

# Modèle avec toutes les variables + interactions
# Ajustement
complet <- glm(formula = formule_glm, 
               data = appr_glm, 
               family = binomial)
# Sélection du point de coupure
complet_coupe <- hecmulti::select_pcoupe(
  modele = complet, c00 = 0, 
  c01 = 0, c10 = -10, c11 = 57)
# Performance sur données de validation
complet_pred <- 
  predict(object = complet, 
          newdata = valid_glm, 
          type = "response") > complet_coupe$optim
# Revenu
complet_perfo <- 
  -10*sum(complet_pred) + 
  sum(valid_2023[complet_pred], na.rm = TRUE)



# Sélection de modèle avec algorithme glouton
# Recherche séquentielle (AIC)
seqAIC <- step(object = complet, 
               direction = "both", # séquentielle
               k = 2, # AIC 
               trace = 0) 
seqAIC_coupe <- 
  hecmulti::select_pcoupe(
    modele = seqAIC, c00 = 0, 
    c01 = 0, c10 = -10, c11 = 57)
seqAIC_pred <- 
  predict.glm(object = seqAIC, 
              newdata = valid_glm, 
              type = "response") > 
  seqAIC_coupe$optim
seqAIC_perfo <- 
  -10*sum(seqAIC_pred) + 
  sum(valid_2023[seqAIC_pred], 
      na.rm = TRUE)

# Recherche séquentielle (BIC)
seqBIC <- step(object = complet,
               direction = "both", # séquentielle
               k = log(nobs(complet)), #BIC
               trace = 0)  
seqBIC_coupe <- hecmulti::select_pcoupe(
  modele = seqBIC, c00 = 0,
  c01 = 0, c10 = -10, c11 = 57)
seqBIC_pred <- 
  predict.glm(object = seqBIC, 
              newdata = valid_glm, 
              type = "response") > 
  seqBIC_coupe$optim
seqBIC_perfo <- 
  -10*sum(seqBIC_pred) + 
  sum(valid_2023[seqBIC_pred], 
      na.rm = TRUE)


print(seqAIC)
# Recherche exhaustive par algorithm génétique
# avec moins de variables
appr_r <- data.frame(
  cbind(model.matrix(seqAIC)[,-1], 
        y = appr_glm$don_2023))
valid_r <- data.frame(
  model.matrix(formula(seqAIC), 
               data = valid_glm)[,-1])

###  Modèle Exgen


library(tictoc)
library(glmulti)

tic()
exgen <- glmulti::glmulti(
  y = formule_glm_no_interaction,
  #nombre de variables limitées
  data = appr_glm,
  level = 1,           # sans interaction
  method = "g",        # recherche génétique
  crit = "bic",            # critère (AIC, BIC, ...)
  confsetsize = 1,         # meilleur modèle uniquement
  plotty = FALSE,
  report = FALSE,  # sans graphique ou rapport
  fitfunction = "glm")
toc()

# Redéfinir le modèle via "glm"
exgen_modele <- 
  glm(exgen@objects[[1]]$formula,
      data = appr_glm,
      family = binomial)
exgen_coupe <- 
  hecmulti::select_pcoupe(
    modele = exgen_modele, 
    c00 = 0, c01 = 0, c10 = -10, c11 = 57)
exgen_pred <- 
  predict(exgen_modele, 
          newdata = valid_glm, 
          type = "response") > exgen_coupe$optim
exgen_perfo <-  
  -10*sum(exgen_pred) + 
  sum(valid_2023[exgen_pred], 
      na.rm = TRUE)


# LASSO
# Trouver le paramètre de pénalisation par
# validation croisée (10 groupes)
cvfit <- glmnet::cv.glmnet(
  x = as.matrix(valid_glm[, -ncol(valid_glm)]), 
  y = valid_glm$don_2023, 
  family = "binomial", 
  type.measure = "auc") # aire sous courbe
# Le critère par défaut est la déviance (-2ll)
# Ajuster modèle avec pénalisation 
lasso <- glmnet::glmnet(
  x = as.matrix(valid_glm[,-ncol(valid_glm)]), 
  y = valid_glm$don_2023, 
  family = "binomial", 
  lambda = cvfit$lambda.1se)
# Calculer performance selon les points de coupure
probs_lasso <- 
  predict(lasso, 
          newx = as.matrix(valid_glm[,-ncol(valid_glm)]), 
          type = "resp")
lasso_coupe <- with(
  hecmulti::perfo_logistique(
    prob = probs_lasso,
    resp = valid_glm$don_2023),
  coupe[which.max(VP*57 - FN*10)])
lasso_pred <- c(predict(lasso, 
                        newx = as.matrix(valid_glm[,-ncol(valid_glm)]), 
                        type = "resp")) > lasso_coupe
lasso_perfo <- -10*sum(lasso_pred) + 
  sum(valid_2023[lasso_pred], na.rm = TRUE)




classif <- rbind(
  c(table(base_pred, valid_glm$don_2023)),
  c(table(complet_pred, valid_glm$don_2023)),
  c(table(seqAIC_pred, valid_glm$don_2023)),
  c(table(seqBIC_pred, valid_glm$don_2023)),
  c(table(exgen_pred, valid_glm$don_2023)),
  c(table(lasso_pred,valid_glm$don_2023))
)
colnames(classif) <- c("VN","FN","FP","VP")
sensibilite <- c(1, classif[,"VP"]/(classif[,"VP"] + classif[,"FN"]))
tauxbonneclassif <- c(0.232, (classif[,"VP"] + classif[,"VN"])/rowSums(classif))

datf <- data.frame(
  modele = paste0("(", letters[1:7], ")"),
  ncoef = c(NA,
            length(coef(base)),
            length(coef(complet)),
            length(coef(seqAIC)),
            length(coef(seqBIC)),
            length(coef(exgen_modele)),
            lasso$df + 1L) - 1L, # retirer l'ordonnée à l'origine
  pcoupe = c(NA,
             base_coupe$optim,
             complet_coupe$optim,
             seqAIC_coupe$optim,
             seqBIC_coupe$optim,
             exgen_coupe$optim,
             lasso_coupe),
  classif = tauxbonneclassif,
  sensibilite = sensibilite,
  gain = c(601212,
           base_perfo,
           complet_perfo,
           seqAIC_perfo,
           seqBIC_perfo,
           exgen_perfo,
           lasso_perfo))
# Imprimer le tableau
knitr::kable(datf,
             col.names =  c("modèle",
                            "no. variables",
                            "pt. coupure",
                            "sensibilité",
                            "taux bonne classif.",
                            "profit"),
             row.names = FALSE,
             booktabs = TRUE,
             longtable = FALSE,
             align =  paste0(c("l",rep("r", 5)),
                             collapse = ""),
             round = c(0, 0, 2, 2, 3, 0),
             escape = FALSE)


print(base_coupe$optim)

############################
## Estimation Simultannée ##
############################

library(sampleSelection)

select_modlin <- 
  MASS::stepAIC(
    object = lm(formule_complet,
                data = dbm[dbm$test == 0,]),
    scope = formula(ymontant ~ 1),
    k = log(sum(dbm$test == 0)),
    trace = FALSE)
fachat <- formula(seqBIC)
fmontant <- formula(select_modlin)
heckit.ml <- sampleSelection::heckit(
  selection = fachat,
  outcome = fmontant, 
  method = "ml", 
  data = dbm[dbm$test == 0,])
sortie_heckit <- summary(heckit.ml)
pred_achat <- 
  predict(heckit.ml, 
          part = "selection", 
          newdata = dbm[dbm$test == 1,], 
          type = "response") * 
  predict(object = heckit.ml,
          part = "outcome", 
          newdata = dbm[dbm$test == 1,])
#Remplacer valeurs manquantes par zéros
valid_ymontant[is.na(valid_ymontant)] <- 0
# On envoie le catalogue seulement si la
# prédiction du montant d'achat est supérieure à 10$

# Revenu total avec cette stratégie
heckit_perfo <-
  sum(valid_ymontant[which(pred_achat > 10)] - 10)



######################################
## Optimisation d'envoi de trousses ##
## avec le modele "base"            ##
######################################
# Calcul de la performance pour le modèle base
calcul_performance_dyn_base <- function(point_coupe, base, valid_glm, valid_donations) {
  pred_dyn <- predict(object = base, newdata = valid_glm, type = "response") > point_coupe
  
  nb_envois_dyn <- sum(pred_dyn)
  
  cout_dyn <- ifelse(nb_envois_dyn <= 60000, 
                     -5 * nb_envois_dyn, 
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  performance_dyn <- cout_dyn + sum(valid_donations[pred_dyn], na.rm = TRUE)
  
  return(performance_dyn)
}

resultats_optim_dyn_base <- optimize(
  f = function(coupe) calcul_performance_dyn_base(coupe, base, valid_glm, valid_2023),
  interval = c(0, 1),
  maximum = TRUE
)

meilleur_coupe_dyn_base <- resultats_optim_dyn_base$maximum

# Graphe pour base
performance_par_coupe_base <- sapply(seq(0, 1, by = 0.01), function(coupe) {
  calcul_performance_dyn_base(coupe, base, valid_glm, valid_2023)
})

plot(seq(0, 1, by = 0.01), performance_par_coupe_base, type = "l", col = "blue",
     xlab = "Point de coupe", ylab = "Performance",
     main = "Performance en fonction du point de coupe (tarification dynamique - base)")
abline(v = meilleur_coupe_dyn_base, col = "red", lwd = 2, lty = 2)


################################
## Tarifaction dynamique avec ##
## le modele "complet"        ##
################################

# Calcul de la performance pour le modèle complet
calcul_performance_dyn_complet <- function(point_coupe, complet, valid_glm, valid_donations) {
  pred_dyn <- predict(object = complet, newdata = valid_glm, type = "response") > point_coupe
  
  nb_envois_dyn <- sum(pred_dyn)
  
  cout_dyn <- ifelse(nb_envois_dyn <= 60000, 
                     -5 * nb_envois_dyn, 
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  performance_dyn <- cout_dyn + sum(valid_donations[pred_dyn], na.rm = TRUE)
  
  return(performance_dyn)
}

resultats_optim_dyn_complet <- optimize(
  f = function(coupe) calcul_performance_dyn_complet(coupe, complet, valid_glm, valid_2023),
  interval = c(0, 1),
  maximum = TRUE
)

meilleur_coupe_dyn_complet <- resultats_optim_dyn_complet$maximum

# Graphe pour complet
performance_par_coupe_complet <- sapply(seq(0, 1, by = 0.01), function(coupe) {
  calcul_performance_dyn_complet(coupe, complet, valid_glm, valid_2023)
})

plot(seq(0, 1, by = 0.01), performance_par_coupe_complet, type = "l", col = "blue",
     xlab = "Point de coupe", ylab = "Performance",
     main = "Performance en fonction du point de coupe (tarification dynamique - complet)")
abline(v = meilleur_coupe_dyn_complet, col = "red", lwd = 2, lty = 2)



################################
## Tarifaction dynamique avec ##
## le modele "seqAIC"        ##
################################

# Calcul de la performance pour le modèle seqAIC
calcul_performance_dyn_seqAIC <- function(point_coupe, seqAIC, valid_glm, valid_donations) {
  pred_dyn <- predict(object = seqAIC, newdata = valid_glm, type = "response") > point_coupe
  
  nb_envois_dyn <- sum(pred_dyn)
  
  cout_dyn <- ifelse(nb_envois_dyn <= 60000, 
                     -5 * nb_envois_dyn, 
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  performance_dyn <- cout_dyn + sum(valid_donations[pred_dyn], na.rm = TRUE)
  
  return(performance_dyn)
}

resultats_optim_dyn_seqAIC <- optimize(
  f = function(coupe) calcul_performance_dyn_seqAIC(coupe, seqAIC, valid_glm, valid_2023),
  interval = c(0, 1),
  maximum = TRUE
)

meilleur_coupe_dyn_seqAIC <- resultats_optim_dyn_seqAIC$maximum

# Graphe pour seqAIC
performance_par_coupe_seqAIC <- sapply(seq(0, 1, by = 0.01), function(coupe) {
  calcul_performance_dyn_seqAIC(coupe, seqAIC, valid_glm, valid_2023)
})

plot(seq(0, 1, by = 0.01), performance_par_coupe_seqAIC, type = "l", col = "blue",
     xlab = "Point de coupe", ylab = "Performance",
     main = "Performance en fonction du point de coupe (tarification dynamique - seqAIC)")
abline(v = meilleur_coupe_dyn_seqAIC, col = "red", lwd = 2, lty = 2)

################################
## Tarifaction dynamique avec ##
## le modele "seqBIC"        ##
################################

# Calcul de la performance pour le modèle seqBIC
calcul_performance_dyn_seqBIC <- function(point_coupe, seqBIC, valid_glm, valid_donations) {
  pred_dyn <- predict(object = seqBIC, newdata = valid_glm, type = "response") > point_coupe
  
  nb_envois_dyn <- sum(pred_dyn)
  
  cout_dyn <- ifelse(nb_envois_dyn <= 60000, 
                     -5 * nb_envois_dyn, 
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  performance_dyn <- cout_dyn + sum(valid_donations[pred_dyn], na.rm = TRUE)
  
  return(performance_dyn)
}

resultats_optim_dyn_seqBIC <- optimize(
  f = function(coupe) calcul_performance_dyn_seqBIC(coupe, seqBIC, valid_glm, valid_2023),
  interval = c(0, 1),
  maximum = TRUE
)

meilleur_coupe_dyn_seqBIC <- resultats_optim_dyn_seqBIC$maximum

# Graphe pour seqBIC
performance_par_coupe_seqBIC <- sapply(seq(0, 1, by = 0.01), function(coupe) {
  calcul_performance_dyn_seqBIC(coupe, seqBIC, valid_glm, valid_2023)
})

plot(seq(0, 1, by = 0.01), performance_par_coupe_seqBIC, type = "l", col = "blue",
     xlab = "Point de coupe", ylab = "Performance",
     main = "Performance en fonction du point de coupe (tarification dynamique - seqBIC)")
abline(v = meilleur_coupe_dyn_seqBIC, col = "red", lwd = 2, lty = 2)


################################
## Tarifaction dynamique avec ##
## le modele "exgen"        ##
################################

# Calcul de la performance pour le modèle exgen
calcul_performance_dyn_exgen <- function(point_coupe, exgen, valid_glm, valid_donations) {
  pred_dyn <- predict(object = exgen, newdata = valid_glm, type = "response") > point_coupe
  
  nb_envois_dyn <- sum(pred_dyn)
  
  cout_dyn <- ifelse(nb_envois_dyn <= 60000, 
                     -5 * nb_envois_dyn, 
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  performance_dyn <- cout_dyn + sum(valid_donations[pred_dyn], na.rm = TRUE)
  
  return(performance_dyn)
}

resultats_optim_dyn_exgen <- optimize(
  f = function(coupe) calcul_performance_dyn_exgen(coupe, exgen, valid_glm, valid_2023),
  interval = c(0, 1),
  maximum = TRUE
)

meilleur_coupe_dyn_exgen <- resultats_optim_dyn_exgen$maximum

# Graphe pour exgen
performance_par_coupe_exgen <- sapply(seq(0, 1, by = 0.01), function(coupe) {
  calcul_performance_dyn_exgen(coupe, exgen, valid_glm, valid_2023)
})

plot(seq(0, 1, by = 0.01), performance_par_coupe_exgen, type = "l", col = "blue",
     xlab = "Point de coupe", ylab = "Performance",
     main = "Performance en fonction du point de coupe (tarification dynamique - exgen)")
abline(v = meilleur_coupe_dyn_exgen, col = "red", lwd = 2, lty = 2)


################################
## Tarifaction dynamique avec ##
## le modele "lasso"        ##
################################

# Convert the validation dataset to a matrix for glmnet (lasso)
newx_valid_glm <- as.matrix(valid_glm[, -which(colnames(valid_glm) == "don_2023")])

# Function to calculate performance with dynamic pricing for lasso
calcul_performance_dyn_lasso <- function(point_coupe, modele_glmnet, newx, valid_donations) {
  # Prédictions des probabilités de don
  probas_don <- predict(modele_glmnet, newx = newx, type = "response")
  
  # Appliquer le point de coupure
  selectionnes <- probas_don > point_coupe
  
  # Nombre d'envois
  nb_envois_dyn <- sum(selectionnes)
  
  # Appliquer la tarification dynamique
  cout_dyn <- ifelse(nb_envois_dyn <= 60000,
                     -5 * nb_envois_dyn,
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  # Calculer la performance totale
  performance_dyn <- cout_dyn + sum(valid_donations[selectionnes], na.rm = TRUE)
  
  return(performance_dyn)
}

# Recherche du point de coupe optimal avec tarification dynamique pour lasso
resultats_optim_dyn_lasso <- optimize(
  f = function(coupe) calcul_performance_dyn_lasso(coupe, lasso, newx_valid_glm, valid_2023),
  interval = c(0, 1),
  maximum = TRUE
)

# Afficher le point de coupure optimal
meilleur_coupe_dyn_lasso <- resultats_optim_dyn_lasso$maximum
meilleur_coupe_dyn_lasso

# Générer une série de points de coupe pour le modèle lasso
points_de_coupe_lasso <- seq(0, 1, by = 0.01)

# Calculer la performance pour chaque point de coupe pour le modèle lasso
performance_par_coupe_lasso <- sapply(points_de_coupe_lasso, function(coupe) {
  calcul_performance_dyn_lasso(coupe, lasso, newx_valid_glm, valid_2023)
})

# Création du graphique de la performance en fonction du point de coupe pour le modèle lasso
plot(points_de_coupe_lasso, performance_par_coupe_lasso, type = "l", col = "blue",
     xlab = "Point de coupe", ylab = "Performance",
     main = "Performance en fonction du point de coupe (tarification dynamique - lasso)")

# Ajouter une ligne verticale pour le point de coupe optimal
abline(v = meilleur_coupe_dyn_lasso, col = "red", lwd = 2, lty = 2)


#################################################
## ajout d'éléments présents dans les tableaux ##
## de la prochaine section                     ##
#################################################

# Base model predictions
base_pred_prob <- predict(base, newdata = valid_glm, type = "response")
base_pred_dyn <- base_pred_prob > meilleur_coupe_dyn_base

# Complet model predictions
complet_pred_prob <- predict(complet, newdata = valid_glm, type = "response")
complet_pred_dyn <- complet_pred_prob > meilleur_coupe_dyn_complet

# seqAIC model predictions
seqAIC_pred_prob <- predict(seqAIC, newdata = valid_glm, type = "response")
seqAIC_pred_dyn <- seqAIC_pred_prob > meilleur_coupe_dyn_seqAIC

# seqBIC model predictions
seqBIC_pred_prob <- predict(seqBIC, newdata = valid_glm, type = "response")
seqBIC_pred_dyn <- seqBIC_pred_prob > meilleur_coupe_dyn_seqBIC

# exgen model predictions
exgen_pred_prob <- predict(exgen_modele, newdata = valid_glm, type = "response")
exgen_pred_dyn <- exgen_pred_prob > meilleur_coupe_dyn_exgen

# Lasso model predictions (using the matrix form for glmnet)
lasso_pred_prob <- predict(lasso, newx = newx_valid_glm, type = "response")
lasso_pred_dyn <- lasso_pred_prob > meilleur_coupe_dyn_lasso

############################################
## profits selon la tarifaction dynamique ##
############################################

# Calculating dynamic performance for the base model
base_performance_dyn <- calcul_performance_dyn(meilleur_coupe_dyn_base, base, valid_glm, valid_2023)

# Calculating dynamic performance for the complet model
complet_performance_dyn <- calcul_performance_dyn(meilleur_coupe_dyn_complet, complet, valid_glm, valid_2023)

# Calculating dynamic performance for the seqAIC model
seqAIC_performance_dyn <- calcul_performance_dyn(meilleur_coupe_dyn_seqAIC, seqAIC, valid_glm, valid_2023)

# Calculating dynamic performance for the seqBIC model
seqBIC_performance_dyn <- calcul_performance_dyn(meilleur_coupe_dyn_seqBIC, seqBIC, valid_glm, valid_2023)

# Calculating dynamic performance for the exgen model
exgen_performance_dyn <- calcul_performance_dyn(meilleur_coupe_dyn_exgen, exgen_modele, valid_glm, valid_2023)


calcul_performance_dyn <- function(point_coupe, modele_glm, validation_set, valid_donations, is_glmnet = FALSE) {
  if (is_glmnet) {
    # For glmnet models (like lasso), use 'newx' and ensure it's a matrix
    pred_dyn <- predict(modele_glm, newx = validation_set, type = "response") > point_coupe
  } else {
    # For standard glm models
    pred_dyn <- predict(object = modele_glm, newdata = validation_set, type = "response") > point_coupe
  }
  
  nb_envois_dyn <- sum(pred_dyn)  # Number of incentive emails sent
  
  # Apply dynamic pricing
  cout_dyn <- ifelse(nb_envois_dyn <= 60000, 
                     -5 * nb_envois_dyn, 
                     -5 * 60000 - 25 * (nb_envois_dyn - 60000))
  
  # Calculate the total performance (profit)
  performance_dyn <- cout_dyn + sum(valid_donations[pred_dyn], na.rm = TRUE)
  
  return(performance_dyn)
}



lasso_performance_dyn <- calcul_performance_dyn(
  point_coupe = meilleur_coupe_dyn_lasso, 
  modele_glm = lasso, 
  validation_set = newx_valid_glm,  # Matrix for glmnet
  valid_donations = valid_2023, 
  is_glmnet = TRUE  # Specify that this is a glmnet model
)


# Calculating dynamic performance for the lasso model (note the use of newx_valid_glm)
lasso_performance_dyn <- calcul_performance_dyn(meilleur_coupe_dyn_lasso, lasso, newx = as.matrix(newx_valid_glm), valid_2023)


###############################################
## Aggrégation des résultats dans un tableau ##
###############################################

sum(db1_nomiss$`2023`)
table(db1_nomiss$don_2023)
classif_everyone <- 142143/857857
profit_envoi_everyone <- 9356270 - ((60000*5)+ (964000*25))

# Calcul des tables de classification pour chaque modèle
classif <- rbind(
  c(table(base_pred_dyn, valid_glm$don_2023)),
  c(table(complet_pred_dyn, valid_glm$don_2023)),
  c(table(seqAIC_pred_dyn, valid_glm$don_2023)),
  c(table(seqBIC_pred_dyn, valid_glm$don_2023)),
  c(table(exgen_pred_dyn, valid_glm$don_2023)),
  c(table(lasso_pred_dyn, valid_glm$don_2023))
)
colnames(classif) <- c("VN","FN","FP","VP")

# Check dimensions of the classification table
print(dim(classif))  # Should show the correct number of columns

# Créer une colonne pour les noms des modèles
modeles <- c("base", "complet", "seqAIC", "seqBIC", "exgen_modele", "lasso")

# Convertir la matrice de classification en data frame
classif_df <- as.data.frame(classif)

# Ajouter une colonne pour indiquer le modèle
classif_df <- cbind(Modèle = modeles, classif_df)

# Noms de colonnes de classification
colnames(classif_df)[2:5] <- c("VN", "FN", "FP", "VP")

# Afficher le tableau final
gt(classif_df)


# Calcul de la sensibilité et du taux de bonne classification pour chaque modèle
sensibilite <- c(1, classif[,"VP"] / (classif[,"VP"] + classif[,"FN"]))
specificite <- c(0, classif[,"VN"] / (classif[,"VN"] + classif[,"FP"]))
tauxbonneclassif <- c(classif_everyone, (classif[,"VP"] + classif[,"VN"]) / rowSums(classif))


#################################
## talbeau avec la Spécificité ##
#################################

# Création du tableau des résultats des modèles
datf <- data.frame(
  modele = c("Baseline", "base", "complet", "seqAIC", "seqBIC", "exgen_modele", "lasso"),
  ncoef = c(NA,
            length(coef(base)),
            length(coef(complet)),
            length(coef(seqAIC)),
            length(coef(seqBIC)),
            length(coef(exgen_modele)),
            lasso$df + 1L) - 1L, # retirer l'ordonnée à l'origine
  pcoupe = c(NA,
             meilleur_coupe_dyn_base,
             meilleur_coupe_dyn_complet,
             meilleur_coupe_dyn_seqAIC,
             meilleur_coupe_dyn_seqBIC,
             meilleur_coupe_dyn_exgen,
             meilleur_coupe_dyn_lasso),
  classif = tauxbonneclassif,
  sensibilite = sensibilite,
  specificite = specificite,  # Inclusion de la colonne de spécificité
  gain = c(profit_envoi_everyone,  # profit initial
           base_performance_dyn,
           complet_performance_dyn,
           seqAIC_performance_dyn,
           seqBIC_performance_dyn,
           exgen_performance_dyn,
           lasso_performance_dyn)
)

# Création du tableau avec gt
gt_table <- datf %>%
  gt() %>%
  tab_header(
    title = "Résultats des modèles"
  ) %>%
  cols_label(
    modele = "Modèle",
    ncoef = "No. variables",
    pcoupe = "Pt. coupure",
    sensibilite = "Sensibilité",
    specificite = "Spécificité",
    classif = "Taux bonne classif.",
    gain = "Profit"
  ) %>%
  fmt_number(
    columns = vars(ncoef, pcoupe, sensibilite, specificite, classif, gain),
    decimals = 2
  ) %>%
  cols_align(
    align = "right",
    columns = vars(ncoef, pcoupe, sensibilite, specificite, classif, gain)
  )

# Afficher le tableau
gt_table



# Compter le nombre de valeurs dans lasso_pred_problème >= 0.9995296
seuil <- 0.9995296
nb_envois_total_lasso <- sum(lasso_pred_prob >= seuil, na.rm = TRUE)

# Afficher le résultat
print(nb_envois_total_lasso)




