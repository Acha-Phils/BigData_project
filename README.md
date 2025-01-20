# BIG DATA PROJECT PART A

## Table of contents 
- [Aim of the project](#aim-of-the-project)
- [Data sets choosen for the project](#data-sets-choosen-for-the-project)
- [Data preparation and exploratory data analysis](#data-preparation-and-exploratory-data-analysis)
- [k-Means Clustering ](#k-means-clustering )
- [Hierarchical Clustering](#hierarchical-clustering)
- [Evaluation of the cluster methods discussed Conclusion](#evaluation-of-the-cluster-methods-discussed-and-conclusion)

# Aim of the project 

1. Use at least two data sets from publicly available open data source, Perform exploratory data analysis on the larger set, merge them and perform unsupervised machine learning. 
2. Perform supervised learning on each choosen data set.

## Part I: Choose data sets 

### Data sets choosen for the project.

1. “World_population.csv” data from (https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset). The data set contains historic information of the population for every country/Territory in the world from 1970 to 2022 and other features such as name of continent, capital city, population growth rate, country’s rank based on population, world population. The data has 17 variables and 234 observations. 

![World popu](https://github.com/user-attachments/assets/6f2d4934-7dcc-47fc-b22d-d5f4d5f87b5b)

2. The second data set is the “Country-data.csv” made up of 167 observations of 10 variables from (https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data/data). The data set contains information from different countries of child mortality for kids under 5 years per 1000 live birth (child_mort), exports of goods per capita (exports), total health spending per capita (health), imports of goods and services per capita (imports), net income per person (income), inflation, life expectancy (life_expec), number of children that would be born with current age-fertility rates remain the same (total_fer) and GDP per capita (gdpp).

![country](https://github.com/user-attachments/assets/fc1b6ab5-be91-48a2-93b1-3c26fa091b35)

3. The final data set “penguin.csv” from (https://www.kaggle.com/datasets/larsen0966/penguins) is made up of 344 observations of 9 variables. The data is a subset of the penguin data set which record sexual dimorphism and environmental variability within Antarctic penguin community.

![penguin](https://github.com/user-attachments/assets/701a5080-d5f1-4f1d-bc90-78095346288b)

## Part II: Data preparation, cleaning and exploratory data analysis

## Data preparation and exploratory data analysis

The “World_population.csv” data set was reduced down to 5 variables that are useful for this project which include; country, continent, rank, growth rate and density per kilometre. The two data sets (“World_population.csv” and “Country-data.csv”) were merged by country resulting in a data set with 185 observations and 14 variables.

![Merged worldpop country](https://github.com/user-attachments/assets/4a84dd30-f402-42b7-8b3e-bdcee365f8ee)

### Summary statistics of the merged data

The population growth rate ranged from 0.91% to 1.1% with a mean of 1% and standard deviation of 1.4%. The average population density per km2 was 216 with a minimum of 2.2 and maximum of 8416. Child mortality recorded a minimum of 2.6 and maximum of 208 with a mean of 37 per 1000 live birth. The mean export of the different countries was 41% of their GDP with a minimum of 0.11% and maximum of 51%. The health spending of the various countries per capita of GDP ranged from 1.8% to 18% with a mean of 6.8%. The imports for goods and services of the different countries ranged from 0.066% to 174% (looks like outlier) with a mean of 46% of the GDP per capita. The net income per person of the different countries was on average 17709 with a minimum of 700 to a maximum of 125000. The annual growth rate of the total GDP of the different countries (inflation) ranged from -4.2% to 104|% (possible outlier) with a mean of 7.7%. The average number of years a new born child would live if the current mortality patterns are to remain the same (life_expec) ranged from 32 to 83 with a mean of 71. The number of children that would be born to each woman if the current age-fertility rates remain the same in the different countries was 2.9 on average with a minimum of 1.1 to 7.5. The average GDP per capita of the different countries was 13471 with a minimum of 231 to maximum of 105000 (table 5).

![Summary](https://github.com/user-attachments/assets/50ffeafe-9e43-4567-af34-362c61ea5844)

Figure one shows the distribution of the various variables. All variables have potential outliers present except for rank.

![Distri](https://github.com/user-attachments/assets/9edc5433-2678-42b5-af31-94ecc0d542d6)

### Correlation of the country’s characteristics

![Corr](https://github.com/user-attachments/assets/25982408-d723-451a-b374-8ea525f26bbf)

Results from the correlation matrix showed that child mortality, exports, imports, total fertility and GDP per capita are high correlated with growth rate, density per km2, exports, income, child mortality and income respectively. Imports is also positively correlated with the country’s rank, the country’s total fertility is positively correlated with growth rate. The country’s GDP per capita is also positively correlated with life expectancy. On the other hand, the country’s income is negatively correlated with child mortality, life expectancy is highly negatively correlated with child mortality. Total fertility is negatively correlated with both income and life expectancy of the country. Other positive and negative correlation exist and con be found on figure 2.

### Summary of the penguin data
The penguin data set was checked for NAs. A frequency of the NA values for penguin sex show males to be more frequent than females, so NA values for sex where replaced with female. The rows still having NA were imputed by replacing the NAs with their respective means before analysis. The resulting data contain characteristics on 344 penguins 44%, 20% and 36% of which were of the species Adelle, Chinstap and Gentoo respectively. The sampled penguins, 49%, 36% and 15% were from the islands, Biscoe, Dream and Torgersen respectively. The penguin’s bill length ranged from 32mm to 60mm with a mean of 44mm. The bill depth had a mean of 17mm and ranged from 13mm to 22mm. The penguin’s flipper length had an average of 201mm and ranged from 172mm to231mm. Finally, their body mass had an average of 4202g and ranged from 2700g to 6300g (table 6).

![peng stats](https://github.com/user-attachments/assets/97bd48ae-fbd8-4e94-8ecc-49d306fccf07)

## Part III: Some Unsupervised machine learning algorithms 

### k-Means Clustering 
 
**Objective**: To partition the merged dataset into best number of clusters in a way that each data point belong to the cluster within the nearest mean.

**Data preparation**: Outliers were removed from the merged dataset before cluster analysis.
I determined the number of groups desired in advance using the Elbow method and display the results from the fviz_nbclust() library which contain within sum of square (WSS) values Figure 4.

![Elbow](https://github.com/user-attachments/assets/56bdce11-705d-4a7f-b209-dd87fb52167d)

Results from figure 4 shows that we select k=4 because after 4, increasing k does not result in a considerable decrease of the total within sum of squares.

![Kmeans table](https://github.com/user-attachments/assets/79f27e36-969c-4898-aef9-5194c016f555)

There is noticeable overlap between some clusters (e.g., clusters 1 and 2), suggesting potential similarities or ambiguity in separating these groups figure 5.

![k-means cluster](https://github.com/user-attachments/assets/aa9f4075-58be-4503-ac7c-573d3faf3ce8)

### Hierarchical Clustering

Unlike k-Means clustering that group observations into k clusters in which each observation belongs to the cluster with the nearest mean, hierarchical clustering builds a tree-like structure (dendrogram) to represent the relationships between data points. Unlike K-Means clustering, hierarchical clustering It does not require a predefined number of clusters and provides a visual representation of the clustering process.

**Objective**: Build a hierarchy of clusters using the merged dataset by iteratively splitting clusters based on their proximity using the divisive method.

**The algorithm of hierarchical cluster follows**; initialization where all data points are in a single cluster. The split step, where the data is split into clusters based on their proximity until each point is in its own cluster. The stopping step where the splitting process is stopped based on the distance threshold. Figure 6 display the result of the clustering analysis performed using the above mentioned clustering
algorithm.

![hcluster](https://github.com/user-attachments/assets/9c732814-c1e1-4eac-925c-fd649d5b1cc8)

### Evaluation of the cluster methods discussed and Conclusion

![Silhouette](https://github.com/user-attachments/assets/2c712b86-d9d8-4a25-95fa-1d91b8e5fa47)  

As shown on figure 7, both clustering technique performed on the merged data set separated the data reasonably well into four clusters as shown by Silhouette score. 
It could be concluded that some characteristic are common between countries resulting in some overlap in the different cluster as shown on figure 5.














