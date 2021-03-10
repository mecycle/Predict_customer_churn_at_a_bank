# Predicting Customer Churn at a Bank
A data analysis program from Kaggle

## Project Flow
<img src="results/project flow.png" width="80%" height="80%" alt = "where"> 


## Exploratory Data Analysis
<img src="results/box plots.png" width="40%" height="40%" alt = "where"> 
<ul>
  <li> No significant difference in the credit score distribution.</li>
  <li> Older customers are churning at higher rate than the younger ones.</li>
  <li> No significant difference in products and salary.</li>
</ul>

### Some other feathers were added to the variables to increase the dimension of the data, including:
<ul>
  <li> Balance salary ratio (Balance/EstimatedSalary)</li>
  <li> Tenure by age (Tenure/Age)</li>
  <li> Credit score by given age(CreditScore/Age)</li>
</ul>

## Result for training set:
<img src="results/model score of training set.png" width="40%" height="40%" alt = "where"> 

 <ul>
  <li> The model score of random forest is the highest.</li>
</ul>


## Result for testing set:
<img src="results/model score of testing set.png" width="40%" height="40%" alt = "where"> 

 <ul>
  <li> A prediction rate of 70% for churned customers.</li>
</ul>

More infromation you can find: https://www.kaggle.com/kmalit/bank-customer-churn-prediction
