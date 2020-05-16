# =====================================================================================================================
#
# Car Insurance Cold Calls
#
# Megan Beckett                                  Andrew Collier
# megan@exegetic.biz                             andrew@exegetic.biz
# https://www.linkedin.com/in/meganbeckett/      https://www.linkedin.com/in/datawookie/
# https://twitter.com/mbeckett_za                https://twitter.com/datawookie
#
# =====================================================================================================================

# This is based on the data from https://www.kaggle.com/kondla/carinsurance.
#
# The data were gathered by a bank which was trying to sell car insurance to potential clients.

# STAGES --------------------------------------------------------------------------------------------------------------

# Astrid Radermacher11:19 AM
#' @Maria to recap: We want to predict whether or not people phoned by
#' insurance companies will accept a new product telephonically based on
#'  previous data points captured, such as how many previous calls were made,
#'what kind of insurance they currently have
#' etc
#' Andrew is currently taking us through the data wrangling step


# 1. Acquire data.
# 2. Load data.
# 3. Wrangle data.
# 4. Exploratory analysis.
# 5. Build models.

# ACQUIRE DATA --------------------------------------------------------------------------------------------------------

# 1. Create an account on Kaggle.
# 2. Login to Kaggle.
# 3. Navigate to the URL above.
# 4. Click on the "Download" button (next to the "New Notebook" button).
# 5. Unpack the ZIP archive into a data/ folder.

# LIBRARIES -----------------------------------------------------------------------------------------------------------

library(rpart)               # Decision Tree models
library(rattle)              # Visualising Decision Tree models (can be a problem to install on Windows)
library(rpart.plot)          # Visualising Decision Tree models
library(Metrics)             # Evaluating model performance
library(MASS)                # Stepwise Algorithm
library(caret)               # Swiss Army Knife

library(readr)               # Reading CSV files
library(dplyr)               # General data wrangling
library(forcats)             # Working with factors
library(janitor)             # Cleaning column names
library(naniar)              # Missing data
library(ggplot2)             # Wicked plots
library(lubridate)           # Handling date/time data

# NOTE: It's important to load dplyr after MASS, otherwise MASS::select will mask dplyr::select.

# LOAD DATA -----------------------------------------------------------------------------------------------------------

PATH_TRAIN <- file.path("data", "carInsurance_train.csv")
#
# There's also a "test" dataset but we're not going to load that because we're not entering a competition.

# Read in the training data.
#
insurance <- read_csv(PATH_TRAIN)

# Take a look at the column names.
#
names(insurance)

# Improve column names (using snake case).
#
insurance <- insurance %>% clean_names()
#
names(insurance)

# Look at structure of data.
#
str(insurance)

# Look at a "spreadsheet" view of data.
#
View(insurance)
#
# What do all of those fields mean?

# Data dictionary:
# 
# id                 — Unique identifier
# age                — Age (years)
# job                — Job type (11 levels; missing data)
# marital            — Marital status (3 levels)
# education          — Highest level of education (3 levels; missing data)
# default            — History of credit default
# balance            — Average annual balance (presumably in bank account, USD)
# hh_insurance       — Is household insured?
# car_loan           — Is there a loan on car?
# communication      — Contact communication type (2 levels; missing data)
# last_contact_day   — Day of last contact
# last_contact_month — Month of last contact
# no_of_contacts     — Number of times contact during current campaign
# days_passed        — Number of days since contact on previous compaign (-1 -> not previously contacted)
# prev_attempts      — Number of contacts before current campaign
# outcome            — Outcome of previous marketing campaign (3 levels; missing data)
# call_start         — Start time of last call
# call_end           — End time of last call
# car_insurance      — Did the client buy car insurance? (target)

# We can immediately drop the ID column since this cannot have any predictive value.
#
insurance <- insurance %>% select(-id)

# WRANGLE -------------------------------------------------------------------------------------------------------------

# Convert columns to factor.
#
#
insurance <- insurance %>% mutate(
  job                = factor(job),
  marital            = factor(marital),
  education          = factor(education),
  communication      = factor(communication),
  outcome            = factor(outcome),
  #
  # Numeric columns are first translated into strings (because these are ultimately converted into dummy variables).
  #
  default            = ifelse(default == 1, "yes", "no") %>% factor(),
  hh_insurance       = ifelse(hh_insurance == 1, "yes", "no") %>% factor(),
  car_loan           = ifelse(car_loan == 1, "yes", "no") %>% factor(),
  #
  # Months:
  #
  # - ordered factor and
  # - ensure that order of levels is not alphabetical!
  #
  last_contact_month = factor(last_contact_month, levels = tolower(month.abb), ordered = TRUE),
  car_insurance      = factor(car_insurance)
)

# Let's take a look at a couple of the categorical variables.
#
insurance$default
insurance$last_contact_month

# MISSING DATA --------------------------------------------------------------------------------------------------------

# Which features have missing entries?
#
vis_miss(insurance)

# Do the missing entries occur together?
#
gg_miss_upset(insurance)

# The outcome feature seems to have a lot of missing entries.
#
# Can we understand what's going on there?
#
insurance %>% filter(is.na(outcome))
#
# It's always missing when the number of previous attempts is zero.

# We could try to impute the missing values, but in this case the fact that these data are simply "missing" is
# probably informative in itself.
#
# Make missing levels explicit.
#
insurance <- insurance %>% mutate(
  job           = fct_explicit_na(job, na_level = "missing"),
  education     = fct_explicit_na(education, na_level = "missing"),
  communication = fct_explicit_na(communication, na_level = "missing"),
  outcome       = fct_explicit_na(outcome, na_level = "missing")
)

# FEATURE ENGINEERING -------------------------------------------------------------------------------------------------

# Create a "call duration" column.
#
insurance <- insurance %>% mutate(
  call_end   = as.POSIXct(call_end, format = "%H:%M:%S"),
  call_start = as.POSIXct(call_start, format = "%H:%M:%S"),
  call_duration = as.numeric(difftime(call_end, call_start, units = "mins"))
)

# Convert the call start and end times to numeric time of day.
#
insurance <- insurance %>% mutate_at(
  c("call_end", "call_start"),
  function(time) hour(time) + minute(time) / 60
)

# EDA: TABLES ---------------------------------------------------------------------------------------------------------

# What proportion of the calls resulted in a sale?
#
prop.table(table(insurance$car_insurance))
#
# So by simply guessing we'd be right around 40% of the time. Can we do better than this with a model?

# EDA: PLOTS ----------------------------------------------------------------------------------------------------------

scale_fill_palette <- scale_fill_manual(values = c("#AAAAAA", "#5D9D3F"))

# Age
#
ggplot(insurance, aes(x = age, fill = car_insurance)) +
  geom_histogram() +
  xlab("Age") +
  scale_fill_palette +
  theme_classic()

# Age (improved)
#
ggplot(insurance, aes(x = age, fill = car_insurance)) +
  geom_histogram(binwidth = 5, position = "fill") +
  xlab("Age") +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Job
#
ggplot(insurance, aes(x = job, fill = car_insurance))+
  geom_bar() +
  xlab(NULL) +
  scale_fill_palette +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1.0))

# Job (improved)
#
# - by wrangling data.
#
insurance %>% 
  count(job, car_insurance) %>%
  ungroup() %>%
  group_by(job) %>%
  mutate(prop = n/sum(n)) %>%
  ggplot(aes(x = job, y = prop, fill = car_insurance)) +
  geom_bar(stat = "identity") +
  xlab("") +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()  +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1.0))

# - by using ggplot functionality to do it for us.
#
ggplot(insurance, aes(x = job, fill = car_insurance))+
  geom_bar(position = "fill") +
  xlab(NULL) +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1.0))

# Education
#
ggplot(insurance, aes(x = education, fill = car_insurance))+
  geom_bar(position = "fill") +
  xlab("Education") +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Balance
#
ggplot(insurance, aes(x = balance, fill = car_insurance)) +
  geom_density(alpha = 0.75) +
  scale_fill_palette +
  theme_classic() +
  scale_x_continuous(limits = c(-5000, 20000))

# Credit Default
#
ggplot(insurance, aes(x = default, fill = car_insurance)) +
  geom_bar(position = "fill") +
  xlab("Credit Default") +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Household Insurance
#
ggplot(insurance, aes(x = hh_insurance, fill = car_insurance)) +
  geom_bar(position = "fill") +
  xlab("Household Insurance") +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Car Loan
#
ggplot(insurance, aes(x = car_loan, fill = car_insurance)) +
  geom_bar(position = "fill") +
  xlab("Car Loan") +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Month of last contact
#
ggplot(insurance, aes(x = last_contact_month, fill = car_insurance)) +
  geom_bar(position = "fill") +
  xlab(NULL) +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Day of last contact
#
# Perhaps more likely to buy insurance towards beginning of month?
#
ggplot(insurance, aes(x = last_contact_day, fill = car_insurance)) +
  geom_bar(position = "fill") +
  xlab(NULL) +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# Call duration
#
ggplot(insurance, aes(x = call_duration, fill = car_insurance)) +
  geom_density(alpha = 0.75) +
  xlab(NULL) +
  ylab("proportion") +
  scale_fill_palette +
  theme_classic()

# This EDA has only considered univariate relationships.
#
# It could be extended with a multivariate analysis.

# TRAIN/TEST SPLIT ----------------------------------------------------------------------------------------------------

# In order to assess how well our model performs we need to split it into two components:
#
# - training and
# - testing.
#
# There are a number of ways to do this split.

# Set the RNG seed so that everybody gets the same train/test split.
#
set.seed(13)

# Generally you want to have around 80:20 split.
#
index <- sample(c(TRUE, FALSE), nrow(insurance), replace = TRUE, prob = c(0.8, 0.2))

train <- insurance[index,]
test <- insurance[!index,]

# Check that the proportions are correct (should be roughly 4x as many records in training data).
#
nrow(train)
nrow(test)

# What about the proportions of the target variable?
#
prop.table(table(train$car_insurance))
prop.table(table(test$car_insurance))
#
# These look fairly close. Good enough for the moment.

# MODEL #1 ------------------------------------------------------------------------------------------------------------

# A Decision Tree model is a great place to start. It should produce reasonable result and it also gives a model which
# is simple to interpret.

model_rpart <- rpart(car_insurance ~ ., data = train)
#
# This is a "kitchen sink" model: we're throwing in all of the features.
#
# Fortunately a Decision Tree works rather well for this because it'll perform implicit feature selection, retaining
# only those features which do actually contribute.

# Take a quick look at the model.
#
model_rpart
#
# What's the most important feature?

# Plot the tree.
#
fancyRpartPlot(model_rpart)
rpart.plot(model_rpart, cex = 0.75)
#
# Interpretations:
#
# - The longer the call, the more likely the sale.
# - Long calls less effective for elderly. Simply enjoy the chat?

# MODEL ASSESSMENT ----------------------------------------------------------------------------------------------------

# Make predictions on the testing data.
#
test_predictions <- predict(model_rpart, test)
head(test_predictions)
#
# These are the predicted probabilities of the outcome classes.
#
# Can we get a class prediction?
#
?predict.rpart
#
# We need to specify the 'type' parameter.

# Specify that we want to predict classes.
#
test_predictions <- predict(model_rpart, test, type = "class")
head(test_predictions)
#
# Bingo!

# So how well does the model perform? Need to compare the known to the predicted classes.
#
head(test$car_insurance)
#
# Looks reasonable. But let's be more rigorous!

# Compare predictions to known values.
#
test_predictions == test$car_insurance

# What proportion of these are correct?
#
mean(test_predictions == test$car_insurance)

# There's a function for this!
#
accuracy(test$car_insurance, test_predictions)
#
# This is the proportion of predictions that are correct.
#
# There's a problem with this though: if our model is really good at predicting the negative class then it will still
# have good accuracy.

# MODEL #2 ------------------------------------------------------------------------------------------------------------

# Unfortunately our model appears to be better than it actually is.
#
# Why?
#
# Because it's reliant on a feature that we could not have prior to contacting the potential client: call duration!

# 🚨 Exercise
#
# 1. Remove the columns relating to the call.
# 2. Repeat the train/test split.
# 3. Build a Decision Tree model.
# 4. Plot the tree. What's the most important feature?
# 4. Make predictions on the testing set.
# 5. Calculate the accuracy.

# === -> YOUR CODE ===
# === <- YOUR CODE ===

# DAY OF WEEK ---------------------------------------------------------------------------------------------------------

# 🚨 Exercise (BONUS)
#
# We have the month and day of last contact. Is it possible to guess the year?
#
# It's possible that not all of the contacts happened in the same year, but you should be able to get an idea of a
# likely year.

# === -> YOUR CODE ===
# === <- YOUR CODE ===