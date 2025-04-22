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

Our model utilizes a standard scaler to standardize the number of steps, description length, and number of classifying tags. These are all quantitative variables, which we standardized because we know that the description length will likely always be longer than our other two variables, causing an imbalanced weighing scale. Our baseline model recorded a Mean Squared Error of 41%. Our current model isnt great, but also isn't horrible, as we are trying to judge how much people like a recipe based on how descriptive the website is for them, rather than actually analyzing any of the actual parts of the recipe, such as ingredients, preparation time, etc. Therefore, by default, it would make sense that what we're analyzing can't give a perfectly accurate representation. 

## Final Model

One new feature we added was the number of words per tag, which we calculated by directly dividing the description word count we created by the number of tags. 