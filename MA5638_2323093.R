
setwd("~/Brunel Notes/MA5638 Big Data Analytics/Course work/New folder")
# Libraries
library(dplyr)          ; library(ggplot2)
library(cluster)        ; library(tree)
library(randomcoloR)    ; library(rsample)
library(ClusterR)       ; library(GGally)
library(neuralnet)      ; library(partykit)
library(corrplot)       ; library(randomForest)
library(vtable)         ; library(factoextra)
library(kernlab)        ; library(caret) 
library(class)

# Data import and Preparation
world_pop1<-read.csv("world_population.csv")
country<-read.csv("Country-data.csv")

# Quick look at the data
str(world_pop1)
head(world_pop1)
# Select attributes to use in the project
world_pop<-world_pop1[, c("Country", "Continent", "Rank", "Growth.Rate", 
                           "Density..per.km.." )]

# Merge the two data sets by country
world<-merge(world_pop, country,  by='Country')

# Getting data summary of the different sets
str(world_pop)
str(country)
str(world)
summary(world_pop)
summary(country)
summary(world)

penguins<-read.csv("penguins.csv")

################################################################################
#     Exploratory Data Analysis
################################################################################
# Get summary statistics
sumtable(world[, 3:14], title="Summary statistics of the merged data")


# histogram for each variable
par(mfrow=c(3,4))
hist(world[, 3], main = names(world)[3], xlab = names(world)[3], col='skyblue')
hist(world[, 4], main = names(world)[4], xlab = names(world)[4], col='skyblue')
hist(world[, 5], main = names(world)[5], xlab = names(world)[5], col='skyblue')
hist(world[, 6], main = names(world)[6], xlab = names(world)[6], col='skyblue')
hist(world[, 7], main = names(world)[7], xlab = names(world)[7], col='skyblue')
hist(world[, 8], main = names(world)[8], xlab = names(world)[8], col='skyblue')
hist(world[, 9], main = names(world)[9], xlab = names(world)[9], col='skyblue')
hist(world[, 10], main = names(world)[10], xlab = names(world)[10], col='skyblue')
hist(world[, 11], main = names(world)[11], xlab = names(world)[11], col='skyblue')
hist(world[, 12], main = names(world)[12], xlab = names(world)[12], col='skyblue')
hist(world[, 13], main = names(world)[13], xlab = names(world)[13], col='skyblue')
hist(world[, 14], main = names(world)[14], xlab = names(world)[14], col='skyblue')
par(mfrow=c(1,1))

par(mfrow=c(3,4))
boxplot(world[, 3], main = names(world)[3], xlab = names(world)[3], col='skyblue')
boxplot(world[, 4], main = names(world)[4], xlab = names(world)[4], col='skyblue')
boxplot(world[, 5], main = names(world)[5], xlab = names(world)[5], col='skyblue')
boxplot(world[, 6], main = names(world)[6], xlab = names(world)[6], col='skyblue')
boxplot(world[, 7], main = names(world)[7], xlab = names(world)[7], col='skyblue')
boxplot(world[, 8], main = names(world)[8], xlab = names(world)[8], col='skyblue')
boxplot(world[, 9], main = names(world)[9], xlab = names(world)[9], col='skyblue')
boxplot(world[, 10], main = names(world)[10], xlab = names(world)[10], col='skyblue')
boxplot(world[, 11], main = names(world)[11], xlab = names(world)[11], col='skyblue')
boxplot(world[, 12], main = names(world)[12], xlab = names(world)[12], col='skyblue')
boxplot(world[, 13], main = names(world)[13], xlab = names(world)[13], col='skyblue')
boxplot(world[, 14], main = names(world)[14], xlab = names(world)[14], col='skyblue')
par(mfrow=c(1,1))

# Get the correlation between the variables in matrix form
M<-cor(world[,c(3:14)])
corrplot(M, method="number", type="full")

#######################################################
# Penguin data 
str(penguins)

penguin_Nasum<-apply(is.na(penguins[,-1]), 2, sum)
penguin_Nasum

#Impute the NAs from the different numeric variables by their mean
# And categorical variables by their most frequent 

sumtable(penguins)
table(penguins$sex)

# Summary of the NAs in the different columns
penguin_Nasum<-apply(is.na(penguins[,-1]), 2, sum)
# impute missing values of sex
#   then set missing sex to male
penguins[is.na(penguins$sex), 'sex'] = 'female'
#for the categorical variables
penguins[is.na(penguins$bill_length_mm), 'bill_length_mm'] = 44
penguins[is.na(penguins$bill_depth_mm), 'bill_depth_mm'] = 17
penguins[is.na(penguins$flipper_length_mm), 'flipper_length_mm'] = 201
penguins[is.na(penguins$body_mass_g), 'body_mass_g'] = 4202
str(penguins)

# Checking whether the NAs still exist
penguin_Nasum<-apply(is.na(penguins[,-1]), 2, sum)
penguin_Nasum

# Get summary statistics
sumtable(penguins[, 2:7], title="Summary statistics of penguin data")

# Getting the distribution of the data
par(mfrow=c(2,2))
hist(penguins[, 4], main = names(penguins)[4], xlab = names(penguins)[4], col='skyblue')
hist(penguins[, 5], main = names(penguins)[5], xlab = names(penguins)[5], col='skyblue')
hist(penguins[, 6], main = names(penguins)[6], xlab = names(penguins)[6], col='skyblue')
hist(penguins[, 7], main = names(penguins)[7], xlab = names(penguins)[7], col='skyblue')
par(mfrow=c(1,1))
# Box plot penguin data
par(mfrow=c(2,2))
boxplot(penguins[, 4], main = names(penguins)[4], xlab = names(penguins)[4], col='skyblue')
boxplot(penguins[, 5], main = names(penguins)[5], xlab = names(penguins)[5], col='skyblue')
boxplot(penguins[, 6], main = names(penguins)[6], xlab = names(penguins)[6], col='skyblue')
boxplot(penguins[, 7], main = names(penguins)[7], xlab = names(penguins)[7], col='skyblue')
par(mfrow=c(1,1))


# Scatter plot of bill length by bill width by penguin species
ggplot(
  data = penguins, 
  aes(x = bill_length_mm, y = bill_depth_mm, color = species)) +
  geom_point() +
  labs(title = "Bill Length vs. Bill Width by Species",
       x = "Bill Length",
       y = "Bill Width") +
  theme_minimal()

ggplot(
  data = penguins, 
  aes(x = body_mass_g, y = flipper_length_mm, color = species)) +
  geom_point() +
  labs(title = "Body Mass vs. Flipper length by Species",
       x = "Body Mass",
       y = "Flipper Length") +
  theme_minimal()



# Box plot of penguin by species
ggplot(penguins, aes(x = species, y = bill_length_mm, fill = species)) + 
  stat_boxplot(geom = "errorbar",
               width = 0.25) + 
  geom_boxplot()

# Box plot of penguin by species
ggplot(penguins, aes(x = species, y = bill_depth_mm, fill = species)) + 
  stat_boxplot(geom = "errorbar",
               width = 0.25) + 
  geom_boxplot()

###############################################################################
#           A.  Unsupervised Learning
#           1. PCA
###############################################################################
# PCA
world_pca<-world[, -c(1:2)]
pca_world<-prcomp(world_pca, center=T, scale=T)
attributes(pca_world)

# Visual analysis of PCA results
# calculate the proportion of the explained variance (PEV) from the std values
pca_world_var<-pca_world$sdev^2
pca_world_var
pca_world_PEV<-pca_world_var/sum(pca_world_var)
pca_world_PEV
summary(pca_world)

#Plot the proportion of variance explained by each principal component
plot(pca_world_PEV, type="b", xlab="Principal Component", 
     ylab="Proportion of Variance Explained", col="skyblue")

# plotting the variance per PC
plot(pca_world, ylim=c(0,5), main='Variance per PC', col="skyblue")

# plotting the cumulative value of PEV for increasing number of additional PC 
# we add an 80% threshold line to inform the feature extraction.
plot(cumsum(pca_world_PEV),
     ylim=c(0,1),
     xlab='PC',
     ylab='Cumulative PEV',
     pch=20,
     col='orange'
)
abline(h=0.8, col='red', lty='dashed') 

# get and inspect the loadings for each PC
#   note: loadings are reported as a rotation matrix (see lecture)
pca_world_loadings <- pca_world$rotation
pca_world_loadings[, 1:4]

# plot the loadings for the first four PCs as a bar plot
#   note: two vectors for colors and labels are created for convenience
#     for details on the other parameters see the help for bar plot and legend

colvector =c("blueviolet", "brown", "gold", "lightgreen", "firebrick",
             "red", "yellow", "cyan1", "ivory3", "deeppink", "blue", "orange3") 
labvector = c('PC1', 'PC2', 'PC3', 'PC4')
barplot(
  pca_world_loadings[,c(1:4)],
  beside = T,
  yaxt = 'n',
  names.arg = labvector,
  col = colvector,
  ylim = c(-1,1),
  border = 'white',
  ylab = 'loadings'
)
axis(2, seq(-1,1,0.1))
legend(
  'bottomright',
  bty = 'n',
  col = colvector,
  pch = 15,
  row.names(pca_world_loadings)
)

par(mfrow=c(1,2))
### 4.4 generate a biplot for PC1 and PC2
biplot(
  pca_world,
  scale = 0,
  col = c('grey40','orange')
)
### 4.4 generate a biplot for PC1 and PC3

biplot(
  pca_world,
  choices = c(1,3),
  scale = 0,
  col = c('grey40','orange')
)
### 4.4 generate a biplot for PC1 and PC4
biplot(
  pca_world,
  choices = c(1,4),
  scale = 0,
  col = c('grey40','orange')
)
### 4.4 generate a biplot for PC2 and PC3
biplot(
  pca_world,
  choices = c(2,3),
  scale = 0,
  col = c('grey40','orange')
)

### 4.4 generate a biplot for PC2 and PC4
biplot(
  pca_world,
  choices = c(2,4),
  scale = 0,
  col = c('grey40','orange')
)
par(mfrow=c(1,1))

###############################################################################
#             Unsupervised Learning
#           2. Hierarchical Clustering              
#           3. Center-based Clustering (K-Means)
###############################################################################

#     Cluster analysis (Agglomerative hierarchical and k-means)
#     cluster analysis
#     hierarchical clustering - complete linkage

dist_world <- dist(world[, -c(1:2)], method = 'euclidian')
set.seed(240)  # Setting seed
hc_world <- hclust(dist_world, method = 'complete')

### 2.2 plot the associated dendrogram
plot(hc_world, hang = -0.1, labels = world$Continent)

## Remove out liers from attributes
total_fer_boxplot<-boxplot(world$gdpp, plot=F)
#box plot(world$total_fer)
life_expec_boxplot<-boxplot(world$life_expec, plot=F)
life_expec_theshold<-max(life_expec_boxplot$out) 

inflation_boxplot<-boxplot(world$inflation, plot=F)
inflation_threshold<-min(inflation_boxplot$out)   

income_boxplot<-boxplot(world$income, plot=F)
income_threshold<-min(income_boxplot$out)  

imports_boxplot<-boxplot(world$imports, plot=F)
imports_threshold<-min(imports_boxplot$out)  

health_boxplot<-boxplot(world$health, plot=F)
health_threshold<-min(health_boxplot$out)  #

exports_boxplot<-boxplot(world$exports, plot=F)
exports_threshold<-min(exports_boxplot$out)  

child_mort_boxplot<-boxplot(world$child_mort, plot=F)
child_mort_threshold<-min(child_mort_boxplot$out) 

growthrate_boxplot<-boxplot(world$Growth.Rate, plot=F)
growthrate_threshold<-min(growthrate_boxplot$out) 
growthrate_threshold1<-max(growthrate_boxplot$out)

world_filter <- world$life_expec > life_expec_theshold & world$income < income_threshold &
  world$inflation < inflation_threshold & world$imports < imports_threshold & 
  world$health < health_threshold & world$exports < exports_threshold & 
  world$child_mort < child_mort_threshold & world$Growth.Rate > growthrate_threshold &
  world$Growth.Rate < growthrate_threshold1
world_clean<-world[world_filter, ]

# we select the numeric variable for cluster analysis 
scworld<-(world_clean[, -c(1:2,5)])

## Center based clustering  k-means

# Use the Elbow Method to select the number of k to use
fviz_nbclust(scworld, kmeans, method = "wss") + 
  labs(subtitle = "Elbow Method With PCA Value")
# This shows the best number of clusters is could be 4.

### k-means with 4 groups
#   note: select k = 4 groups
set.seed(3500)
k_scworld = kmeans(scworld, 4)
attributes(k_scworld)

# Get the cluster sizes
k_scworld$size

# Extract the cluster IDs
k_sccluster_id<-k_scworld$cluster

# Table 10. Obtain the centers of the different clusters. 
k_scworld$centers

# Plot the clusters
fviz_cluster(k_scworld, data = scworld,
             palette = c("#2E9FDF", "#8B0000", "#E7B800","#8A2BE2" ), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw() , main = "K-Mneas clustering of merged data with K=4"
)

############################################################
# hierarchical clustering - complete linkage
#   note: exclude the country and region attribute
dist_scworld <- dist(scworld, method = 'euclidian')
hc_scworld <- hclust(dist_scworld, method = 'complete')
### 2.2 plot the associated Dendrogram with cleaned data
plot(hc_scworld, hang = -0.1, labels = world_clean$Continent)


# 'cut' the Dendrogram to select one partition with 4 groups
#   note: the cutree command requires a distance cutoff h
#      or the number k of desired groups
hc_sccluster_id <- cutree(hc_scworld, k = 4)
table(hc_sccluster_id)


###############################################################################
# Evaluation of the different cluster results with Silhoutte plot
# then calculate the silhoutte score for the two cluster solutions
sil_k_means <- cluster::silhouette(k_sccluster_id, dist_scworld)
sil_hierachy <- cluster::silhouette(hc_sccluster_id, dist_scworld)
fviz_silhouette(sil_k_means)
fviz_silhouette(sil_hierachy)

### get the attributes averages per cluster
###   for the best clustering result (according to the silhoutte plots)
###   and join the results in a data frame
cluster1 <- apply(scworld[k_sccluster_id == 1,-c(1:2)], 2, mean)
cluster2 <- apply(scworld[k_sccluster_id == 2,-c(1:2)], 2, mean)
cluster3 <- apply(scworld[k_sccluster_id == 3,-c(1:2)], 2, mean)
cluster4 <- apply(scworld[k_sccluster_id == 4,-c(1:2)], 2, mean)
world_clean_cluster_averages <- rbind(cluster1, cluster2, 
                                      cluster3, cluster4)
world_clean_cluster_averages


###############################################################################
#           B.  Supervised Learning
#           1. K Nearest Neighbors 
###############################################################################

# transform the data using a min-max function
MinMax <- function(x){
  tx <- (x - min(x)) / (max(x) - min(x))
  return(tx)
}
penguins_minmax <- apply(penguins[ , c(4:7)], 2, MinMax)
set.seed(1500)
# we cast the matrix generated above to a data frame
penguins_minmax<-as.data.frame(penguins_minmax)
penguins_cen<-cbind(penguins[,c(2)], penguins_minmax)
names(penguins_cen)[1] <- "species"
# we create a 70/30 training test split of the data
n_rows <- nrow(penguins_cen)
# sample 70% (n_rows * 0.7) indices in the ranges 1:nrows
training_idx <- sample(n_rows, n_rows * 0.7)
# filter the data frame with the training indices (and the complement)
training_penguins_cen <- penguins_cen[training_idx,]
test_penguins_cen <- penguins_cen[-training_idx,]
str(training_penguins_cen)
str(test_penguins_cen)
# Get summary statistics of training and test sets
summary(training_penguins_cen)
summary(test_penguins_cen)

sumtable(training_penguins_cen)
sumtable(test_penguins_cen)
pairs(penguins[, 4:7], main = "Pairs Plot of Penguin Data", col = penguins$species)

# predict the class for the test data set
# k is set to the square root of the number of instances in the training set
k_value = sqrt(dim(training_penguins_cen)[1])
knn_training_penguins_cen_pred <- knn(train = training_penguins_cen[,-1], 
                                    test = test_penguins_cen[,-1], 
                                    cl = training_penguins_cen[,1],
                                    k = k_value)
# create a table with actual values and the predictions
penguins_cen_results <- data.frame(
  actual = test_penguins_cen[,1],
  knn = knn_training_penguins_cen_pred
)
# create a contingency table for the actual VS predicted
knn_results_table <- table(penguins_cen_results [,c('actual', 'knn')])
knn_results_table

# calculate accuracy from the contingency table
acc_knn <- sum(diag(knn_results_table)) / sum(knn_results_table)
acc_knn

# If you want to use the caret package for a more detailed confusion matrix
confusion_matrix <- confusionMatrix(data = as.factor(knn_training_penguins_cen_pred), 
                                    reference = as.factor(test_penguins_cen[,1]))
confusion_matrix


###############################################################################
#           B.  Supervised Learning
#           2. Multiple Linear Regression 
###############################################################################
## We use multiple linear regression to predict the average child mortality
head(country)
str(country)
#make this example reproducible
set.seed(1515)

#use 70% of data set as training set and 30% as test set
sample <- sample(c(TRUE, FALSE), nrow(country), replace=TRUE, prob=c(0.7,0.3))
country_train  <- country[sample, ]
country_test   <- country[!sample, ]
str(country_train)
str(country_test)

# Model building
country.lm<-lm(child_mort~exports + health + imports + inflation + life_expec + 
                 total_fer + gdpp, data=country)
summary(country.lm)
# Model evaluation
par(mfrow=c(2,2))
plot(country.lm, col='skyblue')
par(mfrow=c(1,1))

# model with log child mortality
country.lm<-lm(log(child_mort)~exports + health + imports + inflation + life_expec + 
                 total_fer + gdpp, data=country)
summary(country.lm)

# remove imports from the model
country.lm<-lm(log(child_mort)~exports + health + inflation + life_expec + 
                 total_fer + gdpp, data=country)
summary(country.lm)

# Next remove inflation from the model resulting in final model
country.lm<-lm(log(child_mort)~exports + health  + life_expec + 
                 total_fer + gdpp, data=country)
summary(country.lm)

# This model looks okay.
#model diagnostics
par(mfrow=c(2,2))
plot(country.lm, col='skyblue')
par(mfrow=c(1,1))

# Extract the exponentiation coefficients 
exp(country.lm$coefficients)

############# Prediction ##########################
counrty_variables<-c("exports", "health", "life_expec", "total_fer", "gdpp")
country_test_variables<-country_test[, counrty_variables]
country_pred<-predict(country.lm, country_test_variables)
# Check the correlation between predicted and actual 
cor(country_pred, country_test$child_mort)



###############################################################################
#           B.  Supervised Learning
#           3. Decision Tree 
###############################################################################
#change sex and species to factors
penguins$species<-as.factor(penguins$species)
penguins$sex<-as.factor(penguins$sex)
penguins$island<-as.factor(penguins$island)
str(penguins)

summary(penguins)
penguins<-penguins[, -c(1,9)]
# set random seed
set.seed(1999)
# create a 70/30 training/test set split
n_rows <- nrow(penguins)
# sample 70% (n_rows * 0.7) indices in the ranges 1:nrows
training_idx <- sample(n_rows, n_rows * 0.7)
# filter the data frame with the training indices (and the complement)
training_penguin_sp <- penguins[training_idx,]
test_penguin_sp <- penguins[-training_idx,]
training_penguin_sp$sex<-as.factor(training_penguin_sp$sex)
test_penguin_sp$sex<-as.factor(test_penguin_sp$sex)

###############################################################
# 3. Decision tree training
str(training_penguin_sp)
str(test_penguin_sp)
# define a formula for predicting Sales  island + species+ 
penguin_sp_formula = species ~ sex + bill_length_mm + bill_depth_mm + flipper_length_mm +
  body_mass_g + island 

# train a decision tree
tree_penguin_sp <- tree(penguin_sp_formula, data=training_penguin_sp)
summary(tree_penguin_sp)

# plot the tree
plot(tree_penguin_sp)
text(tree_penguin_sp, pretty = 0)

# prune the tree using cross-validation
cv_penguin_sp <- cv.tree(tree_penguin_sp, FUN=prune.misclass)
# create a table of tree size and classification error

cv_penguin_sp_table <- data.frame(
  size = cv_penguin_sp$size,
  error = cv_penguin_sp$dev
)

# plot the cv_penguin_sp_table
plot(
  cv_penguin_sp_table,
  xaxt = 'n',
  yaxt = 'n'
)
axis(1, seq(1,max(cv_penguin_sp_table$size)))
axis(2, seq(5,150,5))


# select the tree size with the minimum error
pruned_tree_size_sp <- cv_penguin_sp_table[which.min(cv_penguin_sp_table$error),
                                            'size']

# prune the tree to the required size
pruned_tree_penguin_sp <- prune.misclass(tree_penguin_sp, 
                                             best = pruned_tree_size_sp)

# inspect the pruned tree
summary(pruned_tree_penguin_sp)

# plot the tree
plot(pruned_tree_penguin_sp)
text(pruned_tree_penguin_sp, pretty = 0)

# compare the un/pruned trees
par(mfrow = c(1,2))
plot(tree_penguin_sp)
text(tree_penguin_sp, pretty = 0)
plot(pruned_tree_penguin_sp)
text(pruned_tree_penguin_sp, pretty = 0)
par(mfrow = c(1,1))
# There is no tree improvement after pruning 

#################################################################
# Decision tree prediction

# compute the prediction for un/pruned trees
#   note: the Sales attribute (column 1) is excluded from the test data set
tree_penguin_sp_pred <- predict(tree_penguin_sp, test_penguin_sp[,-1],
                                    type= "class")
pruned_tree_penguin_sp_pred <- predict(pruned_tree_penguin_sp, 
                                           test_penguin_sp[,-1], type= "class")

# create a table with actual values and the two predictions
penguin_sp_results <- data.frame(
  actual = test_penguin_sp$species,
  unpruned = tree_penguin_sp_pred,
  pruned = pruned_tree_penguin_sp_pred
)

# create a contingency table for the actual VS predicted for both predictions
unpruned_results_table <- table(penguin_sp_results[,c('actual', 'unpruned')])
unpruned_results_table
pruned_results_table <- table(penguin_sp_results[,c('actual', 'pruned')])
pruned_results_table

# calculate accuracy from each contingency table
#   as sum of diagonal elements over sum of the matrix values
acc_unpruned <- sum(diag(unpruned_results_table)) / sum(unpruned_results_table)
acc_unpruned
acc_pruned <- sum(diag(pruned_results_table)) / sum(pruned_results_table)
acc_pruned

###############################################################################
#           B.  Supervised Learning
#           4. Neural Networks
###############################################################################

# transform the data using a min-max function
MinMax <- function(x){
  tx <- (x - min(x)) / (max(x) - min(x))
  return(tx)
}
country_minmax <- apply(world[,-c(1,2)], 2, MinMax)

# we cast the matrix generated above to a data frame
country_minmax<-as.data.frame(country_minmax)

# we create a 70/30 training test split of the data
n_rows <- nrow(country_minmax)
# sample 70% (n_rows * 0.7) indices in the ranges 1:nrows
training_idx <- sample(n_rows, n_rows * 0.7)
# filter the data frame with the training indices (and the complement)
training_country_minmax <- country_minmax[training_idx,]
test_country_minmax <- country_minmax[-training_idx,]

######################################################################
# 3. Neural network training

# define a formula for predicting strength
country_formula = child_mort ~ exports + health + imports + income + inflation +
  life_expec + total_fer + gdpp 

# train a neural network with 1 hidden node
country_nn_1 <- neuralnet(country_formula, data = training_country_minmax)

# train a neural network with 5 nodes on one hidden layer
#   note: the number of layers is set with the hidden option parameter
country_nn_5 <- neuralnet(country_formula, hidden = 5, data = training_country_minmax)

# train a neural network with 5 nodes on each of two hidden layers
country_nn_55 <-neuralnet(country_formula, hidden = c(5,5), data=training_country_minmax)

# plot the three neural networks and compare their structure
plot(country_nn_1)
plot(country_nn_5)   # Best model with least error
plot(country_nn_55)

# 4. Neural network prediction
# compute the prediction for each neural network
#   note: the child motility attribute (column 1) is excluded from the test data set
pred_country_nn_1 <- compute(country_nn_1, test_country_minmax[,-1])
pred_country_nn_5 <- compute(country_nn_5, test_country_minmax[,-1])
pred_country_nn_55 <- compute(country_nn_55, test_country_minmax[,-1])

# create a table with actual values and the three predictions
#   note: predicted values are stored as net_result attribute of the prediction object
country_results <- data.frame(
  actual = test_country_minmax$child_mort,
  nn_1 = pred_country_nn_1$net.result,
  nn_5 = pred_country_nn_5$net.result,
  nn_55 = pred_country_nn_55$net.result
)

##############################################################
# Calculate accuracy of the best network
### create a contingency table of the actual VS predicted
#Countr<-country[ ,c(2, 6:14)]

pred_country_nn5<-compute(country_nn_5, test_country_minmax[,-1])

country_results5 <- data.frame(
  actual = test_country_minmax$child_mort,
  predicted = round(pred_country_nn_5$net.result)
)
table_country_results5 <- table(country_results5) 


######################################################################

# calculate the correlation between actual and predicted values to identify the best predictor
cor(country_results[,'actual'], country_results[,c("nn_1","nn_5", "nn_55")])

# plot actual vs predicted values for the worst (blue) and best predictor (orange2)
#   note: points is used to add points on a graph
plot(
  country_results$actual,
  country_results$nn_5,
  col = 'blue',
  xlab = 'actual child mortality',
  ylab = 'predicted child mortality',
  xlim = c(0,1),
  ylim = c(0,1)
)
points(
  country_results$actual,
  country_results$nn_1,
  col = 'orange2'
)
abline(a = 0, b = 1, col = 'red', lty = 'dashed')
legend(
  'topleft',
  c('nn_5', 'nn_1'),
  pch = 1,
  col = c('blue', 'orange2'),
  bty = 'n'
)

