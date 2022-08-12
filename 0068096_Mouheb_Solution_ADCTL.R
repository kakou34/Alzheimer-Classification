library(recipes)
library(modelgrid)
library(caret)
library(purrr)
library(leaps)
library(cellWise)
library(rrcovHD)
library(mltools)


# Set seed for reproducibility 
set.seed(42)

# Set working directory to where the data set files are located
setwd("D:\\Master\\Cassino\\Statistical Learning\\final-assignment")

## ------------------------------------------------------------------------------------------------------------
#' ** TASK 1 : Control Vs Alzheimer Disease **
#' In this task we will classify patients to two classes: 
#' --> CTL : No deficit 
#' --> AD : Alzheimer Disease
#' 
#' ** 1 - Data loading ** 
# Read files as Data Frames
train_data<- read.csv("ADCTLtrain.csv") # Training set
test_data<- read.csv("ADCTLtest.csv") # Test set

# Remove patient ID columns:
test_ids = test_data$ID
train_data <- subset(train_data, select = -ID )
test_data <- subset(test_data, select = -ID )

# Separating features from labels in the training data
train_labels <- train_data$Label
train_predictors <- subset(train_data, select = 1:429)

# Encoding labels using a 2 level factor
train_data$Label <- factor(train_data$Label)
train_labels <- factor(train_labels)


#'  ** 2 - Data Analysis ** 

#' Check the dimension of the problem
dim(train_data)
#' we can see that we have a very high dimensional problem where the number of 
#' predictors (p=429) is higher than the number of samples (n=164) => p >>> n

#' Check data balance 
summary(train_labels)
#' we can see that we have almost equal number of samples in each class (81/83)
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
#' We can see from the map that there exist ~ 6 samples that have multiple cell outliers 
#' especially in the gene expression predictors. 
#' 
  pca <- PcaHubert(train_predictors, k=10)
  plot(pca)
  # We have 5 row outliers 
  
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
                               summaryFunction =  twoClassSummary) # needed to calculate AUC/ROC 
    set.seed(42)
    rfe_result <- rfe(train_predictors, 
                      train_labels,
                      sizes=c(1, 5, 10, 25, 50, 100, 250), # number of features in the subsets to be tested
                      rfeControl= rfe_ctrl,
                      trControl = train_ctrl,
                      preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                      method = "glm", # Logistic Regression 
                      family = binomial(link = "logit")
                    )
    # Check the results 
    rfe_result
#' from the result we see that the RFE suggests using 25 predictors.
#' print the predictors  
   predictors(rfe_result)
   
   # Create new train data to be used later with only the predictors 
   # suggested by the RFE method
   optimal_preds <- append(predictors(rfe_result), 'Label')
   optimal_train_data <- train_data[ ,optimal_preds]
   
   # Visualize variable importance
   varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:25],
                             importance = varImp(rfe_result)[1:25, 1])
   
   ggplot(data = varimp_data, 
          aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
     geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
     geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
     theme_bw() + theme(legend.position = "none")
   
   # Visualize the RFE results
   ggplot(data = rfe_result, metric = "ROC") + theme_bw()
   
#' We can see from the plots that the model that uses 25 predictors yields 
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
   models <- models %>%
     add_model(model_name = "rfe", x = rfe_recipe) %>%
     add_model(model_name = "baseline", x = initial_recipe)  %>%
   
   
   # train the models and compare the results 
   set.seed(42)
   models <- models %>% train(.)
   # Display re-sampled performance statistics of the fitted models using standard 
   # functionality from the 'caret' package.
   models$model_fits %>% resamples(.) %>% bwplot(.)
   # Show the number of predictors used in each model
   models$model_fits %>%
     map(pluck(c("recipe", "term_info", "role"))) %>%
     map_int(~ sum(.x == "predictor"))
#' We can see from the plots that the model that uses the predictors suggested
#' by RFE achieves a better performance in terms of median AUC/ROC.
#' 
#' __Removing Correlated Predictors___________________
  
#' Next, we try Removing Correlated predictors
  models <- models %>%
     add_model(model_name = "corr_.6", 
              x = initial_recipe %>%
                step_corr(all_predictors(), threshold = .6)) %>%
     add_model(model_name = "corr_.7", 
               x = initial_recipe %>%
               step_corr(all_predictors(), threshold = .7)) %>%
     add_model(model_name = "corr_.8", 
              x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .8))
#' train the models and compare the results 
  set.seed(42)
  models <- models %>% train(.)
  # Display re-sampled performance statistics of the fitted models using standard 
  # functionality from the 'caret' package.
  models$model_fits %>% resamples(.) %>% bwplot(.)
  # Show the number of predictors used in each model
  models$model_fits %>%
  map(pluck(c("recipe", "term_info", "role"))) %>%
  map_int(~ sum(.x == "predictor"))
#' We can see from the plots that removing correlated features did not improve 
#' the performance and the highest AUC/ROC is still achieved using the RFE result.
#' 
#' __Principal Component Analysis_______________
#' 
  # Add model specifications with PCA for dimensionality reduction.
  models <- models %>%
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
#'  Retrain and plot resampling results 
  set.seed(42)
  models <- train(models)
  models$model_fits %>% caret::resamples(.) %>% bwplot(.)

  models$model_fits[c("pca_.75","pca_.8", "pca_.85", "pca_.9", "pca_.95")] %>%
  map(pluck(c("recipe", "term_info", "role"))) %>%
  map_int(~ sum(.x == "predictor"))
#' We can see from the plots that the model trained using a dimensionality reduction
#' based on PCA with a threshold of 0.8 and that yields 15 components  
#' achieved the best performance in terms of median resampled AUC.
#' 
#' ** Summary of ROC of all LR models**s
  resamps <- caret::resamples(models$model_fits)
  summary(resamps)
  
#' based on the results obtained, we will use the model pca_.8 as a best result 
#' for Logistic Regression => __LR: Median AUC/ROC = 0.9531250__
   exp.roc_scores <- c(exp.roc_scores, 0.9531250)
   exp.mcc_scores <- c(exp.mcc_scores, 0.7638889)
   exp.feat_select[1] <- "PCA 0.8"
   exp.n_predictors <- c(exp.n_predictors, 15)
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
  set.seed(42)
  rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1, 5, 10, 25, 50, 100, 250), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "lda", # Linear Discriminant Analysis
  )
  rfe_result
#' from the result we see that the RFE suggests using 10 predictors.
#' print the predictors  
  predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(predictors(rfe_result), 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:10],
                            importance = varImp(rfe_result)[1:10, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Visualize the RFE results
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


   set.seed(42)
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
#' best median AUC/ROC is the one that uses PCA with threshold = 0.9 that results
#' in 41 components
#' 
#' based on the results obtained, we will use the model pca_.9 as a best result 
#' for Linear Discriminant Analysis => __LDA: Median AUC/ROC = 0.9557292__ 
#' 
   exp.roc_scores <- c(exp.roc_scores, 0.9557292)
   exp.mcc_scores <- c(exp.mcc_scores, 0.7692428)
   exp.feat_select[2] <- "PCA 0.9"
   exp.n_predictors <- c(exp.n_predictors, 41)
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
                              summaryFunction = twoClassSummary) # needed to calculate ROC and MCC
   set.seed(42)
   rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1, 5, 10, 25, 50, 100, 250), # number of features in the subsets to be tested
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
   
   set.seed(42)
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
#' best median AUC/ROC is the one that uses PCA with threshold = 0.75. resulting
#' in 9 components
#' 
#' based on the results obtained, we will use the model pca_.75 as a best result 
#' for Quadratic Discriminant Analysis => __QDA: Median AUC/ROC =  0.9218750__ 
   exp.roc_scores <- c(exp.roc_scores, 0.9218750)
   exp.mcc_scores <- c(exp.mcc_scores, 0.7692428)
   exp.feat_select[3] <- "PCA 0.75"
   exp.n_predictors <- c(exp.n_predictors, 9)
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
   set.seed(42)
   rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1, 5, 10, 25, 50, 100, 250), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "knn", # k-NN
                     tuneLength = 3
   )
   # Check the results 
   rfe_result
   rfe_result$fit
   #' from the result we see that the RFE suggests using 10 predictors
   #' print the predictors  
   predictors(rfe_result)
   
   # Create new train data to be used later 
   optimal_preds <- append(predictors(rfe_result), 'Label')
   optimal_train_data <- train_data[ ,optimal_preds]
   
   # Visualize variable importance
   varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:10],
                             importance = varImp(rfe_result)[1:10, 1])
   
   ggplot(data = varimp_data, 
          aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
     geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
     geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
     theme_bw() + theme(legend.position = "none", axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
   
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
                                summaryFunction =newSummary,
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
   
   
   set.seed(42)
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
#' best median AUC/ROC is the result of RFE that uses the best 10 predictors and k = 21
#' 
#' based on the results obtained, we will use the model rfe as a best result 
#' for k-Nearest Neighbors => __k-NN: Median AUC/ROC =0.9531250__
  exp.roc_scores <- c(exp.roc_scores, 0.9531250)
  exp.mcc_scores <- c(exp.mcc_scores, 0.7745967)
  exp.feat_select[4] <- "RFE"
  exp.n_predictors <- c(exp.n_predictors, 10)
  exp.model_params[4] <- "k=21"
  knn_predictors <- predictors(rfe_result)
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
  
  set.seed(42)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 5, 10, 25, 50, 100, 250), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "svmLinear",
                    tuneLength = 5 # fine tune SVM parameter
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using all predictors with C = 1
  #' therefore the RFE result will be the same as the baseline.

  
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
                               summaryFunction =newSummary,
                               classProbs = TRUE),
      metric = "ROC",
      method = "svmLinear",
      tuneLength = 10
    )
  
  #' Add the different models to be trained
  
  models <- models %>%
    add_model(model_name = "baseline=rfe", x = initial_recipe)  %>%
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
  
  
  set.seed(42)
  models <- models %>% train(.)
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
  
#' we can see from the plots and the summary that the Linear SVM model that gives the 
#' best median AUC/ROC is the baseline that uses all predictors with C=1
#' 
#' based on the results obtained, we will use the baseline model as a best result 
#' for Linear SVM => __SVM Linear: Median AUC/ROC = 0.9722222__
  exp.roc_scores <- c(exp.roc_scores, 0.9722222)
  exp.mcc_scores <- c(exp.mcc_scores, 0.7745967)
  exp.feat_select[5] <- "Baseline"
  exp.n_predictors <- c(exp.n_predictors, 429)
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
  set.seed(42)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 5, 10, 25, 50, 100, 250), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "svmRadial",
                    tuneLength = 5 # fine tune SVM parameters
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using 10 predictors.
  #' with sigma = 0.1555022 and C = 0.25.
  #' print the predictors  
  predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(predictors(rfe_result), 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:10],
                            importance = varImp(rfe_result)[1:10, 1])
  
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
                               summaryFunction =newSummary,
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
  
  
  set.seed(42)
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
#' best median AUC/ROC is the baseline that uses all predictors with C=2
#' and sigma = 0.001794
#' 
#' based on the results obtained, we will use the baseline model as a best result 
#' for RBF SVM => __SVM RBF: Median AUC/ROC = 0.9583333__
  exp.roc_scores <- c(exp.roc_scores, 0.9583333)
  exp.mcc_scores <- c(exp.mcc_scores, 0.7638889)
  exp.feat_select[6] <- "Baseline"
  exp.n_predictors <- c(exp.n_predictors, 429)
  exp.model_params[6] <- "sigma = 0.001794168, C = 2"
  
#' **__________________________________RF_______________________________________**  
  
#' We repeat the same analysis with Random Forest Classifier
#' Together with the feature selection parameters, we will fine-tune the parameter mtry of RF
#' with GridSearch algorithm using the tuneLength parameter of Caret's train() function
#'
#' ** Feature Selection **
#' 
  caretFuncs$summary <-twoClassSummary
  
  rfe_ctrl <- rfeControl(functions=caretFuncs, 
                         method = "repeatedcv", # Apply repeated CV
                         number = 5, # use 5 folds
                         repeats =5) # repeat 5 times
  
  train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                             summaryFunction =twoClassSummary) # needed to calculate ROC
  set.seed(42)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 5, 10, 25, 50, 100), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "rf", # random forest
                    tuneLength = 5
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using 10 predictors and mtry = 2
  # print the predictors  
  predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(predictors(rfe_result), 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:10],
                            importance = varImp(rfe_result)[1:10, 1])
  
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
                               summaryFunction =newSummary,
                               classProbs = TRUE),
      metric = "ROC",
      method = "rf", # random forest
      tuneLength = 10
    )
  
  #' Add the different models to be trained
  
  models <- models %>%
    add_model(model_name = "rfe", x = rfe_recipe) %>%
    add_model(model_name = "baseline", x = initial_recipe)  #%>%
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
    # 
  
  set.seed(42)
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
#' best median AUC/ROC is the result of RFE that uses 10 predictors with mtry = 2
#' 
#' based on the results obtained, we will use the baseline model as a best result 
#' for Random Forest => __RF: Median AUC/ROC = 0.9414062__
  exp.roc_scores <- c(exp.roc_scores, 0.9414062)
  exp.mcc_scores <- c(exp.mcc_scores, 0.7789731)
  exp.feat_select[7] <- "RFE"
  exp.n_predictors <- c(exp.n_predictors, 10)
  exp.model_params[7] <- "mtry = 2"
  
  
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
#'  is Linear SVM using all predictors provided and C=1
#'  
#'  ** 5- Testing outlier removal **
#'  
#' we will use validation set approach to test the effect of removing outliers
   train_samples <- train_data[1:150, ]
   valid_samples <- train_data[151:164, ]
   
   preds <- train_samples[, -430]
   labels <- train_samples[, 430]
    
   new_preds <- train_samples[-c(77, 143, 142, 145, 56), -430]
   new_labels <- train_samples[-c(77, 143, 142, 145, 56), 430]
   
#' __With outliers______ 
   # Training control
   trControl = trainControl(summaryFunction =newSummary,
                            classProbs = TRUE)
   # Train SVM model
   set.seed(42)
   model_outliers <- train(x= preds , y= labels, 
                        method = 'svmLinear',
                        trControl = trControl,
                        metric = 'ROC',
                        preProcess=c("center", "scale"),
                        tuneGrid = data.frame(C = 1))
   val_pred_out <- predict(model_outliers, valid_samples[, -430])
   acc1 <- sum(val_pred_out == valid_samples[, 430])/length(val_pred_out)
   
   #' __Without outliers______ 
   # Training control
   trControl = trainControl(summaryFunction =newSummary,
                            classProbs = TRUE)
   # Train SVM model
   set.seed(42)
   model_no_outliers <- train(x= new_preds , y= new_labels, 
                           method = 'svmLinear',
                           trControl = trControl,
                           metric = 'ROC',
                           preProcess=c("center", "scale"),
                           tuneGrid = data.frame(C = 1))
   val_pred_no_out <- predict(model_no_outliers, valid_samples[, -430])
   acc2 <- sum(val_pred_no_out == valid_samples[, 430])/length(val_pred_no_out)
   # We can see that acc1 = acc2 therefore removing outliers does not affect the
   # model's performance
   
#'  
#'  
#'  ** 6- Final Model and Test Predictions **
  
  # Training control
  trControl <- trainControl(savePredictions = TRUE, 
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction =newSummary)
  # Train SVM model
  final_model <- train(x= train_predictors , y= train_labels, 
                  method = 'svmLinear',
                  trControl = trControl,
                  metric = 'ROC',
                  preProcess=c("center", "scale"),
                  tuneGrid = data.frame(C = 1))
  
  # Predict on the the test set
  test_preds <- predict(final_model, test_data)
  test_preds <- data.frame(test_ids, test_preds)
  
  # Feature indices (all predictors are used)
  features = 2:430
  

  # Save predictions 
  save(test_preds,  file = "0068096_Mouheb_ADCTLres.RData")
  # Save feature indices 
  save(features,  file = "0068096_Mouheb_ADCTLfeat.RData")

  ############################################ End of Task 1 ########################################  

  
  
  
  