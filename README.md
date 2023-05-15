# Reducing Telco Customer Churn Through Statistical Testing and Machine Learning Modeling Techniques

**Official Project Report**
Link: https://docs.google.com/document/d/1ke1jW6u6KTLUWF2Tj97lp8qvR6g6VyeQjXJeEW2VTlI/edit?usp=sharing

This report provides an examination of the Telco Customer Churn dataset through both statistical testing and machine learning model techniques to better predict whether customers would churn or not and construct a solution to reduce churn.

Yesenia Garcia
Marisol Guel
Anoosha Valliani
Minji Kim

**Introduction
Background**
The Telco Customer Churn Dataset contains information regarding customers who had received services from a fictional phone and internet company specifically highlighting whether customers had either left, stayed, or enrolled in their services. The dataset details several values or factors of each customer concluding to groupings of demographics, location, population, services, and status. The goal of this project was to analyze the IBM fake Telco Customer Churn dataset to identify the customers at risk of churn and propose a solution to prevent/limit churn. For context, churn is defined as the act of a customer no longer paying for a service. Therefore if there is no occurrence, the customer remains a customer of Telco. By identifying customers who are at risk of no longer paying for a service at a company (in this case, Telco), companies can identify what makes their customers disconnect from their services and combat that. This may result in increased engagement and subscription by providing solutions to these issues and potentially unlocking more growth and scaling a business. However, not only does this project benefit companies, it also benefits the consumers. An increase in subscription and engagement means that consumers are being attended to, which in return increases customer loyalty, they find the services satisfactory in that it focuses on the customer's needs. Therefore, trusting in the company and being more lenient if churn increases later on from newer customers.
Objective
As previously stated, the goal of this project was to analyze the IBM fake Telco Customer Churn dataset to identify the customers at risk of churn and propose a solution to prevent/limit churn. After observing the data, we constructed three research questions to help guide us through our methods: How can we construct a solution to reduce churn at Telco and is it possible for it to be applicable to all companies? First, is there a pattern in column values within each customer who churned to see if there is a specific customer group to address at Telco? (we believed this could be answered through statistical testing) If not, is there a way we can accurately predict if a customer will churn or not without a score to then address them? (we believed this could be answered through a machine learning model)
In order to help Telco minimize the churn rate of their customers, we acquired relevant data, applied appropriate analyses to discover potential causes or correlations between users who decide to leave the service, and made a final recommendation. Our group did this by uncovering insights in customer behaviors towards Telco’s telecommunications services by conducting statistical testing on the dataset and then finally training and evaluating a machine learning model to ultimately come to a well-rounded solution for churn.

Data Description
The Telco Customer Churn Dataset details information regarding “home phone and Internet services to 7043 customers in California.” This data came from Kaggle and the IBM Samples Team and we pre-processed it by cleaning it from unnecessary variables. This data consisted of 33 variables and we deemed 10 of them unnecessary considering that they either already state given information that applies to all data points (this being all location variables except “City” being “Country,” “State,” “City,” “Zip Code,” “Lat Long,” “Latitude,” and “Longitude” considering that the entire dataset is known to be regarding customers in California without having to analyze each data point) or have irrelevant labeling that doesn’t serve a purpose in testing (“Customer ID” and “Count”). We also got rid of “Churn Value” since churn value only states if the customer left or stayed at the company, whereas when we separated the customers who left into their own CSV, this variable became unnecessary. And we also got rid of the “Churn Label” when conducting the bivariate analysis since it says the same thing as the churn value but just presented with 1s and 0s instead of yes’s and no’s, but we still kept it when conducting machine learning since it serves as the target variable for our classification model. However, the variables “City,” “Churn Score,” and “Churn Reason” were used during statistical testing for analysis to see if we could derive any helpful insights, but not during the machine learning model testing in order to assist our accuracy score since it would allow the model to learn if customers did ultimately churn or not, which is what we are attempting to predict. Then lastly, the rest of the variables listed were used in either our statistical testing and/or machine learning model techniques and mainly fall within demographics, population, services, and status of customers at Telco.
Feature Detailing
Feature/Variable, DataType, Description

City, String, The city of the customer’s primary residence.
Gender, String, The customer’s gender: Male, Female
Senior Citizen, String, Indicates if the customer is 65 or older: Yes, No
Partner, String, Indicate if the customer has a partner: Yes, No
Dependents, String, Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
Tenure Months, Int, Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
Phone Service, String, Indicates if the customer subscribes to home phone service with the company: Yes, No
Multiple Lines, String, Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
Internet Service, String, Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
Online Security, String, Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
Online Backup, String, Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
Device Protection, String, Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
Tech Support, String, Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
Streaming TV, String, Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
Streaming Movies, String, Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
Contract, String, Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
Paperless Billing, String, Indicates if the customer has chosen paperless billing: Yes, No
Payment Method, String, Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
Monthly Charges, Float, Indicates the customer’s current total monthly charge for all their services from the company.
Total Charges, Float, Indicates the customer’s total charges, calculated to the end of the quarter specified above.
Churn Label, String, Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.
Churn Score, Int, A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
CLTV, Int, Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.
Churn Reason, String, A customer’s specific reason for leaving the company. Directly related to Churn Category.
Methodology
We decided our methodological approach into two categories in order to conclude a collective solution serving insights from both methods.

Statistical Testing Method
Our first methodological approach consisted of primarily identifying the customers at risk of churn and seeing if they followed similar patterns. From this, we expected to find a specific group of customers that would most likely contribute to churn. To begin with statistical testing for bivariate analysis, we first started by separating our data set into customers who did leave the services at Telco & those who didn’t. As you can see in our code, we did this by taking the churn label columns that equaled yes and no and putting them in their own CSV.

Next, we searched for similarities or patterns in other feature columns through numerical and categorical data analysis. As you can see in our code, we did this by .unique to first view all the reasons a customer could have potentially churned and then using .value_counts() and .nlargest() we were able to discover what were the top 3 leading reasons being the attitude of the support person, the competitor offering higher download speeds, and the competitor offering more data.

And throughout other columns, we did similar tests like these but we decided to tie them into our third phase of our statistical testing, which was to visualize the results as shown below to get a better inside look on what can be addressed in our final solution.

Machine Learning Model Method
Random Forest Classification:
We created a supervised machine learning model through Random Forests Classification from Scikit Learn because our goal was to classify the likelihood of customers to churn based on the demographic and service details we selected. We believed this model would allow us to better work with the categorical data that we encoded and would allow us to evaluate the importance of the features to inform our feature selection for future model iterations.

Random Forest is a classification model that utilizes decision trees of n_estimators that select random groupings of data from the training dataset. The decision trees consider random subsets of features when parsing through the subset of data which ensures that correlation among the decision trees is reduced. Once each decision tree has made a label prediction, the model combines all the predictions through majority voting and returns the most frequent label as the final output.

We created 3 machine learning models to predict customer churn. The first was a base model that took in all 27 feature columns and the original dataset of 7043 data points. The second model was trained in response to the low precision and recall scores of the initial model. To improve these scores, we used a resampling technique known as ‘undersampling’ in order to randomly remove instances from the majority class (customers who did not churn) in order to match the number of instances for the minority class (customers who did churn).

The third model took in only 5 columns which showed to have the highest impact scores from our previous model. These 5 columns were 'Total Charges', 'Monthly Charges', 'Tenure Months', 'Contract', 'Internet Service', and 'Churn Label'. Similar to the previous model, our third model employs undersampling to balance the dataset's class distribution. To match the size of the minority class (customers who did churn), it chose a portion of the majority class (customers who did not churn). We added the third model in a different ipynb file because the code was self-contained and might not run with the second model, so we split it for better differentiation and code structure to ensure everything worked well.
Our methods for creating the model are as follows:

Preprocessing

Data Cleaning
Feature Selection
Feature Engineering

One hot encode categorical variables with multiple unique values
Label encode categorical variables with 2 unique values
Reduce the dimensionality of the data by reducing column number
Split the data into training and testing sets

Train the model on the testing set
Utilize a 70:30 ratio for testing and training
Define and test the model

Utilize (n_estimators = 100, random_state=42) parameters
Evaluate the model

Use confusion_matrix, classification_report, accuracy_score
Evaluation:
For our evaluation of the models we created, we utilized accuracy score, confusion matrix, classification report, and feature importance methods from scikit-learn.metrics module.
The accuracy score provides a general score for the performance of a model by comparing predicted labels to the actual labels in a dataset. A higher accuracy would represent improved performance but may be misleading if the dataset is imbalanced or contains noisy data that detracts from the model’s ability to read important features.
A confusion matrix provides a detailed view of the model’s performance and can be used to calculate other evaluation metrics. It also lets you view the number of TP, TN, FP, and FN the model predicted. True Positives - The model correctly predicts customers who churn True Negatives - The model correctly predicts customers who didn’t churn False Positives - The model incorrectly predicts customers who didn’t churn as customers who churned False Negatives - The model incorrectly predicts customers who did churn as customers who did not churn.

The classification report summarizes precision, recall, and F1-score for the different classes (Churn == ‘yes’ and Churn == ‘no’). Precision - measures the ratio of true positives for a class over the total number of positive predictions made by the model Recall - measures the ratio of true positives for a class to the total number of instances that belong to that class F1 - measures a model’s accuracy by combining the precision and recall scores.

Feature Importances is an attribute available for Random Forest models, and it computes the relative importance of feature inputs in determining the final prediction of a label. It completes this by calculating the gini importance, or the mean decrease in impurity. This measures how effective a feature is at reducing uncertainty or variance when creating decision trees. This score is then averaged among all decision trees to return a final ‘feature importance’ score for that feature. We can use the results of the feature importance to decide which features to drop to simplify our model and increase interpretability.

Results
Before conducting our analysis, we expected customers that did churn to perhaps have the basic factors that usually make customers disconnect from the service, including higher monthly or total charges, better offers from leading companies, or basic dissatisfaction with the service itself. Therefore, we expected most, if not all, variables or features that fell under demographics, population, services, and status to be of significant importance or influence, specifically highlighting “Churn Reason.” Although we did not have a fully developed hypothesis, this lead us to believe that the customers who did churn would all fall in a single group of similar variables that correlate with each other and provide a reasonable conclusion for the causation to churn, specifically at Telco to then attend to that group every time churn arose even if this method wouldn’t be applicable to all companies. On the other hand, we expected for factors regarding “Customer ID,” “Count,” “Country,” “State,” “Zip Code,” “Lat Long,” “Latitude,” and “Longitude” to be present but of little importance considering they either already state given information that applies to all data points (this being all location variables except “City” considering that the entire dataset is known to be regarding customers in California without having to analyze each data point) or have irrelevant labeling that doesn’t serve a purpose in testing (“Customer ID” and “Count”). We chose our methodology as our analysis tool because, first, we wanted to see what we could do with our data after viewing the results of statistical testing. Then if there were more diverse groupings than expected, that’s where machine learning would better contribute to a more well-rounded solution to reduce churn for the expected customer base.

Statistical Testing Findings
From our first methodological approach, the “Statistical Testing” section, we derived a range of insights regarding the customers who did churn compared to those who didn’t. However, the most significant finding regarded the top 3 reasons customers churned at Telco being the attitude of customer support at 192 data points, the competing company offered higher download speeds at 189 data points, and then the competing company offered more data at 162 data points. This gives us an idea of what we should address head on before looking into similar column elements each row contains for churned customers. In this case prioritizing addressing reasons "Attitude of support person" and "Competitor offered higher download speeds" considering that they are very close alongside each other unlike the third runner up which has about a 27 point difference. Alongside this, we noticed that many customers who didn’t churn still occasionally had churn scores of high 70s and therefore we calculated the average churn score to see if there was any significance. As a result, we discovered that customers who did churn had an average churn score of 50.1, whereas those who did had a churn score of 82.5. Based on the average churn scores alongside whether a customer actually churned or not, we can predict that customers who didn't churn but have churn scores of high 70s and low 80s (and of course above) are most likely at risk of churning and therefore gives us a better idea of who to focus on regarding addressing the needs of customers who churned as well as those who are most likely to churn later on. However, this arose the question of “What values were weighed for the customers who didn’t churn for them to be set with a churn score near the average of those who did churn?” Based on this question we decided to test other averages such as customer charges where we found the total average charge for customers who churned was around $1532.80 whereas monthly charges were at $74.40. Although informational, these statistics didn’t provide significant insight and we concluded we could reconvene this question in part of our second methodological approach in evaluating a machine learning model. Alongside the top reasons customers churned, we also noticed a wide pool of customers who churned resided in similar demographics. Due to this, we discovered the top 2 cities that recurred the most within customers who churned were in the lower SE area of California, highlighting Los Angeles at 90 data points and San Diego and 50 data points. Based on these similarities, we decided to look at what other columns had similar or the same values for every customer who churned. As a result of this, we discovered that churn was more likely to occur with customers without a partner and customers without dependents. However, one thing that we did note from these insights was that correlation does not equal causation. Although we could find groupings based on similar/same variables within customers who churned data, there wasn’t a single table (Demographics, Location, Population, Services, and Status) that the groupings all fell in. For example, just because most customers who churned didn’t have a partner and paid by electronic check doesn’t mean that this is what caused them to, or the cause for, churn. Therefore, running this data through a machine learning model would better help us better confirm a customer grouping based on all 5 folds of the data. In conclusion, our first out of two-part methodological approach served as a strong foundational basis to steer our second approach giving us a more clear direction to understand what would best contribute to the most effective solution to reduce churn not only at Telco, but at a range of companies where applicable.

Machine Learning Model Findings
For our second methodological approach, being the “Machine Learning Model” section, we found the accuracy, recall, precision, F1 scores for all models and the feature importances scores for our model made on balanced data.

We found that while the first model had a higher accuracy score, the second model had much better performance in identifying class 1 as shown by the increase in precision, recall, and f1 scores. Additionally, the 2nd model had similar scores of around 0.76 in both class 1 and class 0 metrics, revealing similar performance and lack of bias towards a single class. For Telco, the second model would be most beneficial to implement since the cost of false negatives, or not catching customers who are likely to churn, would be high and would not provide Telco the opportunity to target those customers with preventive measures in order to retain them in the service.

However, based on our model accuracy of 76% and the fact that it was trained on a balanced dataset containing almost ½ of the original size, we would not recommend that this model be used without further tuning and improvement. With further feature selection, cross-validation, and a larger dataset to capture the details that may have been overlooked in the balanced dataset, we could improve the performance of our model to a stage in which it is ready for effective business-level implementation. Ultimately, the process of developing a machine learning model for business use is iterative and requires continuous testing, improvement, and evaluation to ensure it is able to address the business problem at hand.

In our third model iteration, we found that the accuracy score decreased to 71%, as we implemented features with the highest impact scores on the model. We assume this is due to the decrease in the amount of features used which can lead to underfitting and the inability to capture trends in the data. But using too many features can also result in overfitting, where the model becomes overly complex and does poorly on fresh, untried data but well on training data. This is why iteration and finding the right balance between the number of features and model accuracy is crucial for creating a reliable predictive model.

Classification Reports:

Model 1:
category, precision, recall, f1-score, support

Class 0, 0.83, 0.90, 0.87, 1522
Class 1, 0.68, 0.52, 0.59, 588
Accuracy, ---, ---, 0.80, 2110
Macro avg, 0.75, 0.71, 0.73, 2110
Weighted avg, 0.79, 0.80, 0.79, 2110
Model 2:
category, precision, recall, f1-score, support

Class 0, 0.76, 0.78, 0.77, 561
Class 1, 0.77, 0.75, 0.76, 561
Accuracy, ---, ---, 0.76, 1122
Macro avg, 0.76, 0.76, 0.76, 1122
Weighted avg, 0.76, 0.76, 0.76, 1122
Model 3:
category, precision, recall, f1-score, support

Class 0, 0.71, 0.73, 0.72, 561
Class 1, 0.72, 0.70, 0.71, 561
Accuracy, ---, ---, 0.72, 1122
Macro avg, 0.72, 0.72, 0.72, 1122
Weighted avg, 0.72, 0.72, 0.72, 1122
Confusion Matrix:
(view official report for better demonstration)
Model 1:
Predicted 0 Predicted 1 Actual 0 11376 146 Actual 1 281 307

Model 2:
Predicted 0 Predicted 1 Actual 0 435 126 Actual 1 139 422

Model 3:
Predicted 0 Predicted 1 Actual 0 410 151 Actual 1 167 394

Feature Importances
Model 2:
Top 5 features with highest importance in contributing to label decision: Model 2
Feature, Importance Score

Tenure months, 0.160
Total Charges, 0.159
Monthly Charges, 0.145
Contract_Month-to-month, 0.075
Internet Service _Fiber optic, 0.048
Model 3:
Top 5 features with highest importance in contributing to label decision: Model 3
Feature, Importance Score

Total Charges, 0.288
Monthly Charges, 0.275
Tenure Months, 0.193
Contract_Month-to-month, 0.104
Internet Service _Fiber optic, 0.046
After evaluating the feature importances for our second model, we confirmed that demographic features such as ‘Dependents’, ‘Gender’, and ‘Partner’ did not seem to be as important to predicting whether churn would occur or not. However, as seen by our third model, which contained only the 5 most important features, the performance did not improve. This suggests that further exploration and experimentation with different sets of features, parameters, and models may be necessary to improve the predictive performance.

Conclusion
Solution and Implications
In conclusion, we were able to collectively identify potential causes of churn through statistical testing, predict which customers will churn or not through a machine learning model, and discover that time and customer’s service charges contribute the most to their decision to churn while demographic data plays a minimal role in that decision. In regards to if our findings are generalizable to other, similar contexts, we believe that they can be to a certain extent. Two of our models were built using only a very small portion of the original dataset which might not be representative of the telecommunications industry and customer base. This suggests that there should be a level of caution when using it in other contexts. We believe that further research, data gathering on customer service details and of customers in different locations would be necessary to strengthen the confidence in our findings. However, certain time-related factors such as tenure months and financial considerations like total and monthly charges can be applied more broadly to other telecommunications businesses looking to retain customers. This is because they are primary factors customers use when deciding whether to switch to a different provider or remain loyal as well as if a service is the most financially advantageous choice for them.

As a result of overall testing and findings, we concluded that our proposed solution for Telco would be to preemptively target users predicted to churn since our model will only give us a label of whether they will churn or not and not a specific churn score. Additionally, we would utilize customer or user surveys to determine if and why they are dissatisfied with their service to then address those concerns in a timely manner to prevent churn before it occurs. This solution could also include reducing costs for customers and offering discounts or other incentives to stay with Telco, if a customer is labeled as likely to churn. Considering that we discovered the main reasons for churn through our statistical analysis, the solution would also include providing better training for customer support personnel to handle customer concerns. We choose this solution because instead of giving the company a score, like that of what IBM had already implemented, they are given a simpler answer that uses only the most important features to give an accurate score. This can free up resources that the company has to target those customers. Additionally, the interpretation of the score is not as clear and thus less actionable. The churn label allows for immediate action to try to retain the customers through discounts and offers, etc. Overall, the churn score is less understandable especially without information on what threshold is being set and what factors it takes into account. However, we believe that this solution maybe be both fair and unfair based on different aspects because while prioritizing attending to customers who are more likely to churn and therefore may be more unhappy with their services at the company may be fair in comparison to those who may be doing fine or are happy with their services, it can also be unfair to be offering better customer attention to those who are more likely to churn rather than all customers in general. We can tell this because it should be a company's priority to address their entire customer at the best service possible. However, this may be unrealistic depending on the company's size since this takes a lot of resources and therefore our solution is a more realistic approach even if it is not ideal. Additionally, there may be limitations considering altering our solution to external companies outside of Telco considering factors like locations outside of California. Therefore, our next steps would involve the retraining of the models with most important features, implementing different locations, and gathering more service details and financial changes.

References
TANKY. (2020, December 7). Telco customer churn: IBM dataset. Kaggle. Retrieved April 24, 2023, from https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

