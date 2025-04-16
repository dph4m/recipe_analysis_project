# Recipe Analysis Project

## Introduction

-Our dataset pulls recipes from food.com and gives a handful of useful values for each. The question we want to answer is, "Is a recipe's average rating predictable by its 'length'?" In other words, 'length' refers to the number of tags the recipe has, the description length, and the number of steps.

-This dataset and question are really important because as college students, we don't necessarily have the time to go through a lengthy process for food, we would prefer to make something simple so we can focus on homework, exams, clubs, etc.

-Our dataset has 83782 rows. We will be suing the columns tags (categorizing words for the recipe), n_steps (number of steps the recipe has), and description (description of the recipe).


## Data Cleaning and Exploratory Data Analysis
-First, we accumulated the ratings of each recipe. We wanted the average rating of each recipe, but realized that some ratings never gave a numerical rating, thus the data had null values. To clean this, we replaced these with 0 and then summed the ratings, and divided them by the count to get the averages. 

-Next, we manipulated the tags column to go from a list to a numerical value by simply taking the length of it.

-We used the same process for the description, but used the length in characters rather than list objects.

## Framing a Prediction Problem


## Baseline Model
## Final Model