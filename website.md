# Is it Worth Making Long Recipes?

Dylan Pham (dpham@umich.edu), Andrew Lezaja (alezaja@umich.edu)

## Introduction

-Our dataset pulls recipes from food.com and gives a handful of useful values for each. The question we want to answer is, "Is a recipe's average rating predictable by its 'length'?" In other words, 'length' refers to the number of tags the recipe has, the description length, and the number of steps.

-This dataset and question are really important because as college students, we don't necessarily have the time to go through a lengthy process for food, we would prefer to make something simple so we can focus on homework, exams, clubs, etc.

-Our dataset has 83782 rows. We will be suing the columns tags (categorizing words for the recipe), n_steps (number of steps the recipe has), and description (description of the recipe).


## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
-First, we accumulated the ratings of each recipe. We wanted the average rating of each recipe, but realized that some ratings never gave a numerical rating, thus the data had null values. To clean this, we replaced these with 0 and then summed the ratings, and divided them by the count to get the averages. 

-Next, we manipulated the tags column to go from a list to a numerical value by simply taking the length of it.

-We used the same process for the description, but used the length in characters rather than list objects.

-This is the head of our DataFrame:

| name                                 |   avgRatingPerRecipe |   n_steps |   n_tags |   desc_word_count |
|:-------------------------------------|---------------------:|----------:|---------:|------------------:|
| 1 brownies in the world    best ever |                    4 |        10 |       14 |                41 |
| 1 in canada chocolate chip cookies   |                    5 |        12 |        9 |                42 |
| 412 broccoli casserole               |                    5 |         6 |       10 |                64 |
| millionaire pound cake               |                    5 |         7 |       20 |                34 |
| 2000 meatloaf                        |                    5 |        17 |       10 |                29 |


### Exploratory Data Analysis

 <iframe
 src="assets/univariate_n_steps.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

This is a bar graph that shows the distribution of the number of steps. It follows a slight-right skew, and has a median of 7-8 steps, as shown by the 'peak' of the bell curve, denoting that most recipes are relatively simple, but we have some outliers that are very lengthy.

 <iframe
 src="assets/rate_across_n_steps.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

 This is another bar graph that shows the sum of average ratings for 5 interval ranges of numbers of steps in the recipe. We can clearly see a drop-off in average ratings once you hit the 70-step mark, intuitively so as the average cook won't have the free time to make a lengthy meal.

### Interesting Aggregates

Below is our merged table:

| name                                    |   avgRatingPerRecipe |   n_steps |   n_tags |   desc_word_count |
|:----------------------------------------|---------------------:|----------:|---------:|------------------:|
| 1 brownies in the world    best ever    |                 4    |        10 |       14 |                41 |
| 1 in canada chocolate chip cookies      |                 5    |        12 |        9 |                42 |
| 412 broccoli casserole                  |                 5    |         6 |       10 |                64 |
| millionaire pound cake                  |                 5    |         7 |       20 |                34 |
| 2000 meatloaf                           |                 5    |        17 |       10 |                29 |
| 5 tacos                                 |                 4    |         5 |       26 |                 5 |
| 50 chili   for the crockpot             |                 5    |         4 |       22 |               125 |
| blepandekager   danish   apple pancakes |                 5    |        10 |       10 |                23 |
| lplermagrone                            |                 5    |        10 |       11 |               108 |
| lplermagrone  herdsman s macaroni       |                 5    |        14 |       13 |                99 |
| rter med flsk   pea soup with pork      |                 5    |         5 |       23 |                29 |
| rtsoppa  swedish yellow pea soup        |                 4    |        14 |       11 |                46 |
| pinards en branche  sauted spinach      |                 5    |        13 |       24 |                 5 |
| go to bbq sauce for ribs                |                 5    |         3 |       11 |                46 |
| bbq spray recipe    it really works     |                 4.75 |         5 |       18 |                32 |
| berry french toast  oatmeal             |                 4.75 |         5 |       12 |                32 |
| big easy   gumbo                        |                 4.5  |        11 |       28 |                42 |
| burek  or feta cheese  phyllo pie       |                 4.5  |        38 |       12 |               101 |
| california roll   salad                 |                 5    |         8 |        5 |                55 |
| cheeeezy  potatoes                      |                 5    |         8 |       22 |                 8 |

To create this table, we needed to use two spreadsheets. First, we had a spreadsheet of the recipes, that had general recipe information such as ingredients, tags like baked, fried, etc., and duration. Second, we had a spreadsheet of the "interactions", or the ratings that users optionally give out of 5 stars, as well as any additional comments.

To merge these into one DataFrame, we mapped the ID of each rating to the ID of the recipe, and thus were able to create the "avgRatingPerRecipe" feature. This was very simple using Python's merge() function.

### Imputation

An imputation we had to use was for ratings that were given no numerical value, thus receiving a Not-a-Number (or NaN) value in our original spreadsheet. To solve this, we replaced these values with 0 in order to do mean calculations without turning the entire calculation into NaN. Next, after placing all of our average ratings back, we had average ratings that only considered ratings that came with numerical values.

In doing so, we were then able to plot our findings, as on an XY plane, it would've been impossible to plot a value of NaN. Therefore, we are unfortunately unable to show the differences between our plots before and after cleaning, as Python doesn't even allow you to plot a NaN value, justifiably.


## Framing a Prediction Problem

Our prediction problem is to estimate an arbitrary recipe's average rating given their 'length', or number of steps, description length, and number of classifying tags. This is a regression problem, as the average rating is a numerical variable. Average rating is our response variable. We chose it because as college students, we want to eat well but we also want simple recipes. Thus, intuitively, we will look for recipes with high ratings and low 'lengths'.

Validation accuracy, training accuracy, and test accuracy are the three metrics we decided to use for this analysis. These were suitable because we want to fine-tune hyperparamters to prevent overfitting, ensure our model accurately fits the actual data we give it, and generalize a set of unseen data predictably well. Validation, training, and test accuracies cover each of these key points respectively.

## Baseline Model

Our model utilizes a standard scaler to standardize the number of steps, description length, and number of classifying tags. These are all quantitative variables, which we standardized because we know that the description length will likely always be longer than our other two variables, causing an imbalanced weighing scale. Our baseline model recorded a Mean Squared Error of 0.418. Our current model isnt great, but also isn't horrible, as we are trying to judge how much people like a recipe based on how descriptive the website is for them, rather than actually analyzing any of the actual parts of the recipe, such as ingredients, preparation time, etc. Therefore, by default, it would make sense that what we're analyzing can't give a perfectly accurate representation. Additionally, we decided to use a Linear Regression Model to keep it simple at first and improve later on by iterating through different models later. 

## Final Model

'One' new feature we added was the nutrition column, which we parsed to introduce 7 new features: calories, total_fat, sugar, sodium, protein, sat_fat, and carbs. We handled the parsing in the baseline model cell, and used the apply function to add each of these columns respectively to our new recipe data frame. Our rationale was that before our features were examining only a recipe's public facing appearance, while the nutrition column would hopefully reveal more insightful information about the actual food itself and possibly what kinds of food succed more than others, which proved to help even the baseline model by a small portion - approximately -0.01 MSE. 

We also decided to engineer 4 new features: desc_per_step (the ratio of a recipe's description word count to its number of recipe steps), protein_per_calorie (a measure of how dense a recipe is with protein compared to its calories), sugar_to_fat (a measure of how dense a recipe is with sugar compared to its fat content), and fat_to_calorie (the ratio of a recipe's fat content with its calories). We added each of these columns to our data frame and handled imputation appropriately by replacing NaN values with 0. We hoped that these new features would introduce even more intimte data about the food, specfically how foods with similar 'macros' may differ from each other in terms of how dense each one is with protein, sugar, fat, etc. (a more apt example may be how a chicken varies from a steak as both may contain similar calorie, protein, and sugar attributes, however, they deviate greatly in fat contant)

Once our new recipe data frame had been completed, we proceeded to drop all rows containing NaN before moving onto training/testing.

We decided upon a standard train/test split and fed our feature columns into our X/Y_train and X/Y_test df's respectively. After iterating through about 10 models before introducing other model enhancements, we found that a Random Forest Regressor proved to yield the best results among other models like Linear Regression, k-Nearest Neighbors (k=10), Naive Bayes, MLP, etc. The RFR likely performed the best due to its robustness to outliers, strong ability to handle non-linear relationships, and resistance to overfitting, which is a problem we ran into from other models.

Once our model algorithm and features were decided upon, we shifted our focus to modifying our features with standard scalar, for the same reasons as in the baseline model, and also a Quantile Transformer mapping calories, total_fat, sugar, and sat_fat to normal distributions. We did this all of this using the preprocessing module from sklearn, although we were tempted to switch over to PyTorch as preprocessing took a considerable amount of time, around 3 minutes, per cell run. 

Our model's build had finally been completed and we were ready to move onto the final steps of the model, being tuning hyperparameters and testing/training. We used GridSearchCV leveraging gradient descent to find the best hyperparameters for our model, avoiding saddle points and local minima. We found that the RFR performed optimally with 5 levels, 100 decision trees, and a minimum sample split of 2. Our train MSE was 0.413 while out test MSE yielded an improvement of -0.022 at 0.396 Test MSE. Our cross-validation accuracy was not as good as we expected, but still lower with a mean CV MSE of 0.417 and a standard devisation of 0.0106, showing consistent performance across all folds. 

There was considerably low overfitting in a model with a train, test, and CV MSE all very familiar to each other. Our very low standard deviation also implies very stable and consistent predictions. Overall, our final model was defintely and improvement, however, the added complexity did not show the considerable improvements we may have hoped for. While an MSE of only 0.396 is good, our features may not have been as insightful as we previously thought. Perhaps a food's average rating is more heavily influenced it cook time or recency or even the user who posted the recipe, all variables we had not account for. Regardless, a food's nutritional value and recipe length are decent indicators of average rating. 