library(recipes)
library(modelgrid)
library(caret)
library(purrr)
library(leaps)
library(cellWise)
library(rrcovHD)
library(mltools)


# Set seed for reproducibility 
set.seed(1)

# Set working directory to where the data set files are located
setwd("D:\\Master\\Cassino\\Statistical Learning\\final-assignment")

## ------------------------------------------------------------------------------------------------------------
#' ** TASK 3 : Control Vs Mild Cognitive Impairment **
#' In this task we will classify patients to two classes: 
#' --> CTL : No deficit 
#' --> MCI : Mild Cognitive Impairment
#' 
#' ** 1 - Data loading ** 
# Read files as Data Frames
train_data<- read.csv("MCICTLtrain.csv") # Training set
test_data<- read.csv("MCICTLtest.csv") # Test set

# Remove patient ID columns:
test_ids = test_data$ID
train_data <- subset(train_data, select = -ID )
test_data <- subset(test_data, select = -ID )

# Separating features from labels in the training data
train_labels <- train_data$Label
train_predictors <- subset(train_data, select = 1:593)

# Encoding labels using a 2 level factor
train_data$Label <- factor(train_data$Label)
train_labels <- factor(train_labels)


#'  ** 2 - Data Analysis ** 

#' Check the dimension of the problem
dim(train_data)
#' we can see that we have a very high dimensional problem where the number of 
#' predictors (p=593) is way higher than the number of samples (n=172) => p >>> n
#' which will introduce a curse of dimensionality 

#' Check data balance 
summary(train_labels)
#' we can see that we have almost equal number of samples in each class (82/90)
#' so the dataset is balanced

#' Correlation 
m <- cor(train_predictors)
image(m) # Draw heatmap of the correlation matrix
#' we can see from the heat map that there are highly correlated predictors (dark red)
#' To avoid the curse of dimensionality and eliminate correlated predictors we will use 
#' dimensionality reduction methods.

#' Check range
summary(train_predictors)
#' from the summary of the data-frame we can see that some of the predictors are normalized to [0-1]
#' range whereas others are not. To avoid bias towards predictors with higher values we will apply 
#' a pre-processing techniques -normalization- to map all predictors to the same distribution 
#' (mean = 0, std = 1) 
#' 
#' 
#' Checking for outliers using Cell-Wise 
#' 
  # Default options for DDC:
  DDCpars = list(fastDDC = FALSE)
  DDC_train_data = DDC(train_data,DDCpars)
  remX = DDC_train_data$remX
  dim(remX)
  cellMap(D=remX, R=DDC_train_data$stdResid, rowlabels = 1:nrow(remX), columnlabels = colnames(remX))
#' We can see from the map that there exist ~ 7 samples that have multiple cell outliers 
#' especially in the gene expression predictors. 
  pca <- PcaHubert(train_predictors, k=64)
  plot(pca)

#' 
  #' ** 3 - Training and Evaluation **
  #' 
  exp.models <- c("LR", "LDA", "QDA", "k-NN", "SVM - Linear", "SVM - RBF", "RF") # models to be trained and compared
  exp.roc_scores <- numeric(0) # array to store AUC values
  exp.mcc_scores <- numeric(0) # array to store MCC values
  exp.feat_select = character(length = 7) # Array to store FS methods
  exp.n_predictors <- numeric(0) # array to store the number of predictors used
  exp.model_params = character(length = 7) # Array to store the optimal model params if any
  
  #' We will create a custom summary function to include mcc with AUC 
  calculate_mcc <- function (data, lev = NULL, model = NULL) {
    mcc_metric <- mcc(data$obs, data$pred)  
    names(mcc_metric) <- "MCC"
    mcc_metric
  }
  
  newSummary <- function(...) c(twoClassSummary(...), calculate_mcc(...))
#' 
#' **________________________________Logistic Regression____________________________________**
#' 
#' ## **Feature selection**
#' We will use recipes and modelGrid to fine-tune parameters of the pre-processing
#' pipeline to perform feature selection. First, we will try  Recursive Feature 
#' Elimination with the RFE function of Caret. Next, we will try removing correlated 
#' predictors. In a final approach, we try to apply Principal Component Analysis (PCA).
#' 
#' 
#' __Recursive Feature Elimination____________________
#' 
    caretFuncs$summary <- twoClassSummary
    
    rfe_ctrl <- rfeControl(functions=caretFuncs, 
                       method = "repeatedcv", # Apply repeated CV
                       number = 5, # use 5 folds
                       repeats =5) # repeat 5 times
    
    train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                               summaryFunction = twoClassSummary) # needed to calculate UC/ROC 
    set.seed(1)
    rfe_result <- rfe(train_predictors, 
                      train_labels,
                      sizes=c(1, 5, 10, 20, 30, 40, 50, 60, 100, 250), # number of features in the subsets to be tested
                      rfeControl= rfe_ctrl,
                      trControl = train_ctrl,
                      preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                      method = "glm", # Logistic Regression 
                      family = binomial(link = "logit")
                    )
    # Check the results 
    rfe_result
#' from the result we see that the RFE suggests using 20 predictors.
#' print the predictors  
   predictors(rfe_result)
   
   # Create new train data to be used later with only the predictors 
   # suggested by the RFE method
   optimal_preds <- append(predictors(rfe_result), 'Label')
   optimal_train_data <- train_data[ ,optimal_preds]
   
   # Visualize variable importance
   varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:20],
                             importance = varImp(rfe_result)[1:20, 1])
   
   ggplot(data = varimp_data, 
          aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
     geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
     geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
     theme_bw() + theme(legend.position = "none")
   
   # Visualize the RFE results
   ggplot(data = rfe_result, metric = "ROC") + theme_bw()
   
   # save in case needed later
   lr_predictors_rfe <- predictors(rfe_result)
   
#' We can see from the plots that the model that uses 20 predictors yields 
#' the best re-sampled AUC/ROC
#' 
   # Create the pre-processing pipeline to be applied to all models
   initial_recipe <- recipe(train_data) %>%
     update_role(Label, new_role = "outcome") %>%
     update_role(-Label, new_role = "predictor") %>%
     step_center(all_predictors()) %>% # center to mean = 0
     step_scale(all_predictors()) # scale to std = 1
   
   # Pre-processing pipeline for new train data suggested by RFE
   rfe_recipe <- recipe(optimal_train_data) %>%
     update_role(Label, new_role = "outcome") %>%
     update_role(-Label, new_role = "predictor") %>%
     step_center(all_predictors()) %>% # center to mean = 0
     step_scale(all_predictors()) # scale to std = 1
   
   # Container for models to be trained and compared
   models <- 
     # create empty model grid with constructor function.
     model_grid() %>%
     # set shared settings, that will apply to all models by default.
     share_settings(
       data = train_data,
       trControl = trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 5,
                                summaryFunction = newSummary,
                                classProbs = TRUE),
       metric = "ROC",
       method = "glm",
       family = binomial(link = "logit")
     )
   
   #' Create baseline model with no feature selection to compare the results with
   #' And add the model suggested by RFE to test on the same split with other models
   #' and other models to be tested
   models <- models %>%
     add_model(model_name = "rfe", x = rfe_recipe) %>%
     add_model(model_name = "baseline", x = initial_recipe) %>%
     add_model(model_name = "corr_.6",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .6)) %>%
     add_model(model_name = "corr_.7",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .7)) %>%
     add_model(model_name = "corr_.8",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.75",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .75)) %>%
     add_model(model_name = "pca_.8",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.85",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .85)) %>%
     add_model(model_name = "pca_.9",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .9)) %>%
     add_model(model_name = "pca_.95",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .95))
     
   
   
   # train the models and compare the results 
   set.seed(1)
   models <- models %>% train(.)
   # Display re-sampled performance statistics of the fitted models using standard 
   # functionality from the 'caret' package.
   models$model_fits %>% resamples(.) %>% bwplot(.)
   # Show the number of predictors used in each model
   models$model_fits %>%
     map(pluck(c("recipe", "term_info", "role"))) %>%
     map_int(~ sum(.x == "predictor"))
#' We can see from the plots that the model that uses PCA with threshold = 0.85
#' achieves the best performance in terms of median AUC/ROC. 
#' Number of components = 16
#' 
#' 
#' ** Summary of ROC of all LR models**
  resamps <- caret::resamples(models$model_fits)
  summary(resamps)
  
#' based on the results obtained, we will use the model pca_.85 as a best result 
#' for Logistic Regression => __LR: Median AUC/ROC = 0.8888889__
   exp.roc_scores <- c(exp.roc_scores, 0.8888889)
   exp.mcc_scores <- c(exp.mcc_scores, 0.6017536)
   exp.feat_select[1] <- "PCA 0.85"
   exp.n_predictors <- c(exp.n_predictors, 16)
   exp.model_params[1] <- "None"
#' 
#' **____________________________Linear Discriminant Analysis________________________________**
#' 
#' We repeat the same analysis with LDA 
#' ** Feature Selection **
#' 
  caretFuncs$summary <- twoClassSummary
   
  rfe_ctrl <- rfeControl(functions=caretFuncs, 
                          method = "repeatedcv", # Apply repeated CV
                          number = 5, # use 5 folds
                          repeats =5) # repeat 5 times
   
  train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                              summaryFunction = twoClassSummary) # needed to calculate ROC
  set.seed(1)
  rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1, 5, 10, 20, 30, 40, 50, 100, 250), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     metric = "ROC",
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "lda" # Linear Discriminant Analysis
                    )
                    
  rfe_result
#' Interestingly, from the result we see that the RFE suggests using 1 predictor.
#' print the predictor  
  predictors(rfe_result)
  lda_rfe_pred <- predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(lda_rfe_pred, 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[2],
                            importance = varImp(rfe_result)[2, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Visualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  lda_predictor_rfe <- predictors(rfe_result)
  
#' We can see from the plots that the model that use only 1 predictor, namely
#' "NDUFA1" yields the best re-sampled AUC/ROC
#' 
#' Create the pre-processing pipeline to be applied to all models
  initial_recipe <- recipe(train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  
  # Pre-processing pipeline for new train data
  rfe_recipe <- recipe(optimal_train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
   # Container for models to be trained and compared
  models <- 
     # create empty model grid with constructor function.
     model_grid() %>%
     # set shared settings, that will apply to all models by default.
     share_settings(
       data = train_data,
       trControl = trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 5,
                                summaryFunction = newSummary,
                                classProbs = TRUE),
       metric = "ROC",
       method = "lda"
     )
   
#' Add the different models to be trained
   
   models <- models %>%
     add_model(model_name = "rfe", x = rfe_recipe) %>%
     add_model(model_name = "baseline", x = initial_recipe)  %>%
     add_model(model_name = "corr_.6",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .6)) %>%
     add_model(model_name = "corr_.7",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .7)) %>%
     add_model(model_name = "corr_.8",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.75",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .75)) %>%
     add_model(model_name = "pca_.8",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.85",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .85)) %>%
     add_model(model_name = "pca_.9",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .9)) %>%
     add_model(model_name = "pca_.95",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .95))


   set.seed(1)
   models <- models %>% train(.)
   # Display re-sampled performance statistics of the fitted models using standard 
   # functionality from the 'caret' package.
   models$model_fits %>% resamples(.) %>% bwplot(.)
   # Show the number of predictors used in each model
   models$model_fits %>%
     map(pluck(c("recipe", "term_info", "role"))) %>%
     map_int(~ sum(.x == "predictor"))
   
#' ** Summary of all LDA models**
   resamps <- caret::resamples(models$model_fits)
   summary(resamps)
   
#' we can see from the plots and the summary that the LDA model that gives the 
#' best median AUC/ROC is the one that uses only the predictor NDUFA1 as suggested
#' by RFE
#' 
#' based on the results obtained, we will use the model rfe as a best result 
#' for Linear Discriminant Analysis => __LDA: Median AUC/ROC = 0.8750000__ 
#' 
   exp.roc_scores <- c(exp.roc_scores, 0.8750000)
   exp.mcc_scores <- c(exp.mcc_scores, 0.5493503)
   exp.feat_select[2] <- "RFE"
   exp.n_predictors <- c(exp.n_predictors, 1)
   exp.model_params[2] <- "None"
   
#' **_________________________Quadratic Discriminant Analysis_______________________________**
#' 
#' We repeat the same analysis with QDA 
#' ** Feature Selection **
#' 
   caretFuncs$summary <- twoClassSummary
   
   rfe_ctrl <- rfeControl(functions=caretFuncs, 
                          method = "repeatedcv", # Apply repeated CV
                          number = 5, # use 5 folds
                          repeats =5) # repeat 5 times
   
   train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                              summaryFunction = twoClassSummary) # needed to calculate ROC
   set.seed(1)
   rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1, 5, 10, 20, 30, 40, 50, 100, 250), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "qda", # Quadratic Discriminant Analysis
   )
  # We can see that QDA was not trained using the original predictors with RFE. 

#' Other methods will be tried

   initial_recipe <- recipe(train_data) %>%
     update_role(Label, new_role = "outcome") %>%
     update_role(-Label, new_role = "predictor") %>%
     step_center(all_predictors()) %>% # center to mean = 0
     step_scale(all_predictors()) # scale to std = 1
   
   # Container for models to be trained and compared
   models <- 
     # create empty model grid with constructor function.
     model_grid() %>%
     # set shared settings, that will apply to all models by default.
     share_settings(
       data = train_data,
       trControl = trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 5,
                                summaryFunction = newSummary,
                                classProbs = TRUE),
       metric = "ROC",
       method = "qda"
     )
  
   # For this classifier, some of the combinations were not suitable for training a QDA therefore they were removed
   models <- models %>%
     add_model(model_name = "corr_.6",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .6)) %>%
     add_model(model_name = "pca_.75",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .75)) %>%
     add_model(model_name = "pca_.8",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.85",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .85)) %>%
     add_model(model_name = "pca_.9",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .9)) %>%
     add_model(model_name = "pca_.95",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .95))
   
   set.seed(1)
   models <- models %>% train(.)
   # Display re-sampled performance statistics of the fitted models using standard 
   # functionality from the 'caret' package.
   models$model_fits %>% resamples(.) %>% bwplot(.)
   # Show the number of predictors used in each model
   models$model_fits %>%
     map(pluck(c("recipe", "term_info", "role"))) %>%
     map_int(~ sum(.x == "predictor"))
   
   #' ** Summary of all QDA models**
   resamps <- caret::resamples(models$model_fits)
   summary(resamps)
   
#' we can see from the plots and the summary that the QDA model that gives the 
#' best median AUC/ROC is the one that uses PCA with threshold = 0.85. resulting
#' in 16 components
#' 
#' based on the results obtained, we will use the model pca_.85 as a best result 
#' for Quadratic Discriminant Analysis => __QDA: Median AUC/ROC =  0.8472222__ 
   exp.roc_scores <- c(exp.roc_scores, 0.8472222)
   exp.mcc_scores <- c(exp.mcc_scores, 0.5277778)
   exp.feat_select[3] <- "PCA 0.85"
   exp.n_predictors <- c(exp.n_predictors, 16)
   exp.model_params[3] <- "None"
#'
#' 
#' **___________________________________K-NN_______________________________________**  
   
#' We repeat the same analysis with k-Nearest Neighbors 
#' Together with the feature selection parameters, we will fine-tune the parameter k 
#' of the classifier with GridSearch algorithm using the tuneLength parameter of Caret's train() function
#' 
#' ** Feature Selection **
   caretFuncs$summary <- twoClassSummary
   
   rfe_ctrl <- rfeControl(functions=caretFuncs, 
                          method = "repeatedcv", # Apply repeated CV
                          number = 5, # use 5 folds
                          repeats =5) # repeat 5 times
   
   train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                              summaryFunction = twoClassSummary) # needed to calculate ROC
   set.seed(1)
   rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1, 5, 10, 20, 30, 40, 50, 100, 250), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     metric = "ROC",
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "knn", # k-NN
                     tuneLength = 3 # try different values of k
   )
   # Check the results 
   rfe_result
   rfe_result$fit
   #' from the result we see that the RFE suggests using 5 predictors
   #' print the predictors  
   predictors(rfe_result)
   # save predictor for later use 
   knn_rfe_preds <- predictors(rfe_result)
   
   # Create new train data to be used later 
   optimal_preds <- append(knn_rfe_preds, 'Label')
   optimal_train_data <- train_data[ ,optimal_preds]
   
   # Visualize variable importance
   varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:5],
                             importance = varImp(rfe_result)[1:5, 1])
   
   ggplot(data = varimp_data, 
          aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
     geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
     geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
     theme_bw() + theme(legend.position = "none")
   
   # Visualize the RFE results
   ggplot(data = rfe_result, metric = "ROC") + theme_bw()
   
   #' We can see from the plots that the model that uses 5 predictors yields 
   #' the best re-sampled AUC/ROC
   #' 
   #' Create the pre-processing pipeline to be applied to all models
   initial_recipe <- recipe(train_data) %>%
     update_role(Label, new_role = "outcome") %>%
     update_role(-Label, new_role = "predictor") %>%
     step_center(all_predictors()) %>% # center to mean = 0
     step_scale(all_predictors()) # scale to std = 1
   
   # Pre-processing pipeline for new train data
   rfe_recipe <- recipe(optimal_train_data) %>%
     update_role(Label, new_role = "outcome") %>%
     update_role(-Label, new_role = "predictor") %>%
     step_center(all_predictors()) %>% # center to mean = 0
     step_scale(all_predictors()) # scale to std = 1
   # Container for models to be trained and compared
   models <- 
     # create empty model grid with constructor function.
     model_grid() %>%
     # set shared settings, that will apply to all models by default.
     share_settings(
       data = train_data,
       trControl = trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 5,
                                summaryFunction = newSummary,
                                classProbs = TRUE),
       metric = "ROC",
       method = "knn",
       tuneLength = 10
     )
   
   #' Add the different models to be trained
   
   models <- models %>%
     add_model(model_name = "rfe", x = rfe_recipe) %>%
     add_model(model_name = "baseline", x = initial_recipe)  %>%
     add_model(model_name = "corr_.6",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .6)) %>%
     add_model(model_name = "corr_.7",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .7)) %>%
     add_model(model_name = "corr_.8",
               x = initial_recipe %>%
                 step_corr(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.75",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .75)) %>%
     add_model(model_name = "pca_.8",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .8)) %>%
     add_model(model_name = "pca_.85",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .85)) %>%
     add_model(model_name = "pca_.9",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .9)) %>%
     add_model(model_name = "pca_.95",
               x = initial_recipe %>%
                 step_pca(all_predictors(), threshold = .95))
   
   
   set.seed(1)
   models <- models %>% train(.)
   # Display re-sampled performance statistics of the fitted models using standard 
   # functionality from the 'caret' package.
   models$model_fits %>% resamples(.) %>% bwplot(.)
   # Show the number of predictors used in each model
   models$model_fits %>%
     map(pluck(c("recipe", "term_info", "role"))) %>%
     map_int(~ sum(.x == "predictor"))
   models$model_fits
   
#' ** Summary of all K-NN models**
#' 
   resamps <- caret::resamples(models$model_fits)
   summary(resamps)
   
#' we can see from the plots and the summary that the k-NN model that gives the 
#' best median AUC/ROC is the result of RFE that uses the best 5 predictors and k = 9
#' 
#' based on the results obtained, we will use the model rfe as a best result 
#' for k-Nearest Neighbors => __k-NN: Median AUC/ROC =0.8726852__
  exp.roc_scores <- c(exp.roc_scores, 0.8726852)
  exp.mcc_scores <- c(exp.mcc_scores, 0.5493503)
  exp.feat_select[4] <- "RFE"
  exp.n_predictors <- c(exp.n_predictors, 5)
  exp.model_params[4] <- "k=9"
#'
#' 
#' **__________________________________SVM, Linear_______________________________________**  
  
#' We repeat the same analysis with Support Vector Machine using a Linear kernel
#' Together with the feature selection parameters, we will fine-tune the parameter C of SVM
#'with GridSearch algorithm using the tuneGrid parameter of Caret's train() function
#'
#' ** Feature Selection **
#' 
  caretFuncs$summary <- twoClassSummary
  
  rfe_ctrl <- rfeControl(functions=caretFuncs, 
                         method = "repeatedcv", # Apply repeated CV
                         number = 5, # use 5 folds
                         repeats =5) # repeat 5 times
  
  train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                             summaryFunction = twoClassSummary) # needed to calculate ROC
  
  set.seed(1)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 5, 10, 20, 30, 40, 50, 100, 250), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    metric = "ROC",
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "svmLinear",
                    tuneLength = 3 # try for different values of C
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using 5 predictors

  # print the suggested predictors
  predictors(rfe_result)
  # save predictor for later use 
  svm_rfe_preds <- predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(svm_rfe_preds, 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:5],
                            importance = varImp(rfe_result)[1:5, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Visualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  
  #' We can see from the plots that the model that uses 5 predictors yields 
  #' the best re-sampled AUC/ROC
  #' 
  #' Create the pre-processing pipeline to be applied to all models
  initial_recipe <- recipe(train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  
  # Pre-processing pipeline for new train data
  rfe_recipe <- recipe(optimal_train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  
  
  # Container for models to be trained and compared
  models <- 
    # create empty model grid with constructor function.
    model_grid() %>%
    # set shared settings, that will apply to all models by default.
    share_settings(
      data = train_data,
      trControl = trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 5,
                               summaryFunction = newSummary,
                               classProbs = TRUE),
      metric = "ROC",
      method = "svmLinear",
      tuneLength = 10
    )
  
  #' Add the different models to be trained
  
  models <- models %>%
    add_model(model_name = "baseline", x = initial_recipe)  %>%
    add_model(model_name = "rfe", x = rfe_recipe)  %>%
    add_model(model_name = "corr_.6",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .6)) %>%
    add_model(model_name = "corr_.7",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .7)) %>%
    add_model(model_name = "corr_.8",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .8)) %>%
    add_model(model_name = "pca_.75",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .75)) %>%
    add_model(model_name = "pca_.8",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .8)) %>%
    add_model(model_name = "pca_.85",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .85)) %>%
    add_model(model_name = "pca_.9",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .9)) %>%
    add_model(model_name = "pca_.95",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .95))
  
  
  set.seed(1)
  models <- models %>% train(.)
  models$model_fits
  # Display re-sampled performance statistics of the fitted models using standard 
  # functionality from the 'caret' package.
  models$model_fits %>% resamples(.) %>% bwplot(.)
  # Show the number of predictors used in each model
  models$model_fits %>%
    map(pluck(c("recipe", "term_info", "role"))) %>%
    map_int(~ sum(.x == "predictor"))
  models$model_fits
  
#' ** Summary of all Linear SVM models**
  resamps <- caret::resamples(models$model_fits)
  summary(resamps)
  
#' we can see from the plots and the summary that both models pca_.85 and pca_95
#' achieve equal ROC/AUC scores. therefore we will choose the model that achieves
#' the highest MCC which is pca_.95 resulting in 66 components with C = 1
#' 
#' based on the results obtained, we will use the baseline model as a best result 
#' for Linear SVM => __SVM Linear: Median AUC/ROC = 0.8750000__
  exp.roc_scores <- c(exp.roc_scores, 0.8750000)
  exp.mcc_scores <- c(exp.mcc_scores, 0.6042610)
  exp.feat_select[5] <- "pca_.95"
  exp.n_predictors <- c(exp.n_predictors, 66)
  exp.model_params[5] <- "C = 1"
  
  
#' **__________________________________SVM, RBF_______________________________________**  
  
#' We repeat the same analysis with Support Vector Machine using an RBF kernel
#' Together with the feature selection parameters, we will fine-tune the parameter C of SVM
#'with GridSearch algorithm using the tuneLength parameter of Caret's train() function
#'
#' ** Feature Selection **
  caretFuncs$summary <- twoClassSummary
  
  rfe_ctrl <- rfeControl(functions=caretFuncs, 
                         method = "repeatedcv", # Apply repeated CV
                         number = 5, # use 5 folds
                         repeats =5) # repeat 5 times
  
  train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                             summaryFunction = twoClassSummary) # needed to calculate ROC
  set.seed(1)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 5, 10, 20, 30, 40, 50, 100, 250), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "svmRadial",
                    tuneLength = 5 # fine tune SVM parameters
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using 250 predictors.

  #' print the predictors  
  predictors(rfe_result)
  svmrbf_rfe_preds <- predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(svmrbf_rfe_preds, 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance for the first 25 predictors
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:25],
                            importance = varImp(rfe_result)[1:25, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Vizualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  
  #' We can see from the plots that the model that uses 250 predictors yields 
  #' the best re-sampled AUC/ROC
  #' 
  #' Create the pre-processing pipeline to be applied to all models
  initial_recipe <- recipe(train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  
  # Pre-processing pipeline for new train data
  rfe_recipe <- recipe(optimal_train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  # Container for models to be trained and compared
  models <- 
    # create empty model grid with constructor function.
    model_grid() %>%
    # set shared settings, that will apply to all models by default.
    share_settings(
      data = train_data,
      trControl = trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 5,
                               summaryFunction = newSummary,
                               classProbs = TRUE),
      metric = "ROC",
      method = "svmRadial",
      tuneLength = 10 # fine tune SVM parameters
    )
  
  #' Add the different models to be trained
  
  models <- models %>%
    add_model(model_name = "rfe", x = rfe_recipe) %>%
    add_model(model_name = "baseline", x = initial_recipe)  %>%
    add_model(model_name = "corr_.6",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .6)) %>%
    add_model(model_name = "corr_.7",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .7)) %>%
    add_model(model_name = "corr_.8",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .8)) %>%
    add_model(model_name = "pca_.75",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .75)) %>%
    add_model(model_name = "pca_.8",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .8)) %>%
    add_model(model_name = "pca_.85",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .85)) %>%
    add_model(model_name = "pca_.9",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .9)) %>%
    add_model(model_name = "pca_.95",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .95))
  
  
  set.seed(1)
  models <- models %>% train(.)
  models$model_fits
  # Display re-sampled performance statistics of the fitted models using standard 
  # functionality from the 'caret' package.
  models$model_fits %>% resamples(.) %>% bwplot(.)
  # Show the number of predictors used in each model
  models$model_fits %>%
    map(pluck(c("recipe", "term_info", "role"))) %>%
    map_int(~ sum(.x == "predictor"))
  
#' ** Summary of all RBF SVM models**
  resamps <- caret::resamples(models$model_fits)
  summary(resamps)
  
#' we can see from the plots and the summary that the RBF SVM model that gives the 
#' best median AUC/ROC is the model suggested by RFE that uses the 250 best predictors
#' with sigma = 0.005432391 and C = 4
#' 
#' based on the results obtained, we will use the RFE model as a best result 
#' for RBF SVM => __SVM RBF: Median AUC/ROC = 0.9027778__
  exp.roc_scores <- c(exp.roc_scores, 0.9027778)
  exp.mcc_scores <- c(exp.mcc_scores, 0.6527778)
  exp.feat_select[6] <- "RFE"
  exp.n_predictors <- c(exp.n_predictors, 250)
  exp.model_params[6] <- "sigma = 0.005432391 and C = 4"
  
#' **__________________________________RF_______________________________________**  
  
#' We repeat the same analysis with Random Forest Classifier
#' Together with the feature selection parameters, we will fine-tune the parameter mtry of RF
#' with GridSearch algorithm using the tuneLength parameter of Caret's train() function
#'
#' ** Feature Selection **
#' 
  caretFuncs$summary <- twoClassSummary
  
  rfe_ctrl <- rfeControl(functions=caretFuncs, 
                         method = "repeatedcv", # Apply repeated CV
                         number = 5, # use 5 folds
                         repeats =5) # repeat 5 times
  
  train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                             summaryFunction = twoClassSummary) # needed to calculate ROC
  set.seed(1)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 5, 10, 20, 30, 40, 50, 100, 250), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    metric = "ROC",
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "rf", # random forest
                    tuneLength = 3 # try different results values for mtry
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using 30 predictors
  # print the predictors  
  predictors(rfe_result)
  rf_rfe_preds <- ctors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append( rf_rfe_preds, 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:30],
                            importance = varImp(rfe_result)[1:30, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Vizualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  
  #' We can see from the plots that the model that uses 10 predictors yields 
  #' the best re-sampled AUC/ROC
  #' 
  #' Create the pre-processing pipeline to be applied to all models
  initial_recipe <- recipe(train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  
  # Pre-processing pipeline for new train data
  rfe_recipe <- recipe(optimal_train_data) %>%
    update_role(Label, new_role = "outcome") %>%
    update_role(-Label, new_role = "predictor") %>%
    step_center(all_predictors()) %>% # center to mean = 0
    step_scale(all_predictors()) # scale to std = 1
  # Container for models to be trained and compared
  models <- 
    # create empty model grid with constructor function.
    model_grid() %>%
    # set shared settings, that will apply to all models by default.
    share_settings(
      data = train_data,
      trControl = trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 5,
                               summaryFunction = newSummary,
                               classProbs = TRUE),
      metric = "ROC",
      method = "rf", # random forest
      tuneLength = 10
    )
  
  #' Add the different models to be trained
  
  models <- models %>%
    add_model(model_name = "rfe", x = rfe_recipe) %>%
    add_model(model_name = "baseline", x = initial_recipe)  %>%
    add_model(model_name = "corr_.6",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .6)) %>%
    add_model(model_name = "corr_.7",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .7)) %>%
    add_model(model_name = "corr_.8",
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .8)) %>%
    add_model(model_name = "pca_.75",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .75)) %>%
    add_model(model_name = "pca_.8",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .8)) %>%
    add_model(model_name = "pca_.85",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .85)) %>%
    add_model(model_name = "pca_.9",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .9)) %>%
    add_model(model_name = "pca_.95",
              x = initial_recipe %>%
                step_pca(all_predictors(), threshold = .95))
  
  
  set.seed(1)
  models <- models %>% train(.)
  
  models$model_fits
  # Display re-sampled performance statistics of the fitted models using standard 
  # functionality from the 'caret' package.
  models$model_fits %>% resamples(.) %>% bwplot(.)
  # Show the number of predictors used in each model
  models$model_fits %>%
    map(pluck(c("recipe", "term_info", "role"))) %>%
    map_int(~ sum(.x == "predictor"))
  
#' ** Summary of all Random Forest models**
  resamps <- caret::resamples(models$model_fits)
  summary(resamps)
  
#' we can see from the plots and the summary that the Random Forest model that gives the 
#' best median AUC/ROC is the one preceded by PCA with threshold = 0.85 resulting
#' in 16 predictors using mtry = 5
#' 
#' based on the results obtained, we will use the model pca_.85 as a best result 
#' for Random Forest => __RF: Median AUC/ROC = 0.8750000__
  exp.roc_scores <- c(exp.roc_scores, 0.8750000)
  exp.mcc_scores <- c(exp.mcc_scores, 0.5493503)
  exp.feat_select[7] <- "RFE"
  exp.n_predictors <- c(exp.n_predictors, 10)
  exp.model_params[7] <- "mtry = 5"
  
  
#' 
#' 
#' ** 4 - Summary **
#' 
#' Results Summary
  results = data.frame(
                      exp.models,
                      exp.feat_select,
                      exp.n_predictors,
                      exp.model_params,
                      exp.roc_scores,
                      exp.mcc_scores
           )

  colnames(results) = c("Classifier", "FS Method", "#Predictors", "Model Params", "Median AUC", "Median MCC")

  # Plot the dataframe as a table 
  knitr::kable(results, escape = FALSE, booktabs = TRUE)
      
#'  As we can see from the summary, the model that gave the best median AUC on Cross validation
#'  is SVM with RBF kernel using the 250 predictors suggested by RFE and sigma = 0.005432391 and C = 4 
#'  ** 5- Testing outlier removal **
#'  Train predictors suggested by RFE
   train_preds_optim <- train_data[, svmrbf_rfe_preds]
#'  
#' we will use validation set approach to test the effect of removing outliers
  train_samples_preds <- train_preds_optim[1:150, ]
  valid_samples_preds <- train_preds_optim[151:172, ]
  
  train_samples_labels <- train_labels[1:150]
  valid_samples_labels <- train_labels[151:172]
  
  
  new_train_preds <- train_samples_preds[-c(71, 125, 132, 87),]
  new_train_labels <- train_samples_labels[-c(71, 125, 132, 87)]
  
  #' __With outliers______ 
  # Training control
  trControl <- trainControl(savePredictions = TRUE, 
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction = newSummary)
  # Train SVM model with outliers
  set.seed(1)
  model_outs <- train(x= train_samples_preds , y= train_samples_labels, 
                       method = 'svmRadial',
                       trControl = trControl,
                       metric = 'ROC',
                       preProcess=c("center", "scale"),
                       tuneGrid = data.frame(sigma = 0.005432391, C = 4))
  
  val_pred_out <- predict(model_outs,  valid_samples_preds)
  acc1 <- sum(val_pred_out == valid_samples_labels)/length(val_pred_out)
  
  #' __Without outliers______ 
  # Training control
  trControl <- trainControl(savePredictions = TRUE, 
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction = newSummary)
  # Train SVM model without outliers
  set.seed(1)
  model_no_outs <- train(x= new_train_preds , y= new_train_labels, 
                      method = 'svmRadial',
                      trControl = trControl,
                      metric = 'ROC',
                      preProcess=c("center", "scale"),
                      tuneGrid = data.frame(sigma = 0.005432391, C = 4))
  
  val_pred_no_out <- predict(model_no_outs,  valid_samples_preds)
  acc2 <- sum(val_pred_no_out == valid_samples_labels)/length(val_pred_out)
  # We can see that acc2 > acc1 therefore removing outliers improves the performance
  # of the model. 
  # outliers will be removed when training the final model
#'  ** 6- Final Model and Test Predictions **
#'  
  optimal_train_predictors <- train_data[-c(71, 125, 132, 87) , svmrbf_rfe_preds]
  optimal_train_labels <- train_labels[-c(71, 125, 132, 87)]
  
  optimal_test_predictors <- test_data[, svmrbf_rfe_preds]
  
  # Training control
  trControl <- trainControl(savePredictions = TRUE, 
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction = newSummary)
  # Train SVM model
  set.seed(1)
  final_model <- train(x= optimal_train_predictors , y= optimal_train_labels, 
                  method = 'svmRadial',
                  trControl = trControl,
                  metric = 'ROC',
                  preProcess=c("center", "scale"),
                  tuneGrid = data.frame(sigma = 0.005432391, C = 4))
  
  # Predict on the the test set
  test_preds <- predict(final_model, optimal_test_predictors)
  test_preds <- data.frame(test_ids, test_preds)
  
  # Feature indices
  features <- numeric(0)
  
  train_data_original <- read.csv("MCICTLtrain.csv")

  for (p in svmrbf_rfe_preds) {
    idx <- which(colnames(train_data_original) == p) 
    features <- c(features, idx)
    
  }

  # Save predictions 
  save(test_preds,  file = "0068096_Mouheb_MCICTLres.RData")
  # Save feature indices 
  save(features,  file = "0068096_Mouheb_MCICTLfeat.RData")

  ############################################ End of Task 3 ########################################  

  
  
  
  