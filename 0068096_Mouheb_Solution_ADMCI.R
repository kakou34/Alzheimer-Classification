library(recipes)
library(modelgrid)
library(caret)
library(purrr)
library(leaps)
library(cellWise)
library(rrcovHD)
library(mltools)


# Set seed for reproducibility 
set.seed(777)

# Set working directory to where the data set files are located
setwd("D:\\Master\\Cassino\\Statistical Learning\\final-assignment")

## ------------------------------------------------------------------------------------------------------------
#' ** TASK 1 : Control Vs Mild Cognitive Impairment **
#' In this task we will classify patients to two classes: 
#' --> MCI : Mild Cognitive Impairment
#' --> AD : Alzheimer Disease
#' 
#' ** 1 - Data loading ** 
# Read files as Data Frames
train_data<- read.csv("ADMCItrain.csv") # Training set
test_data<- read.csv("ADMCItest.csv") # Test set

# Remove patient ID columns:
test_ids = test_data$ID
train_data <- subset(train_data, select = -ID )
test_data <- subset(test_data, select = -ID )

# Separating features from labels in the training data
train_labels <- train_data$Label
train_predictors <- subset(train_data, select = 1:63)

# Encoding labels using a 2 level factor
train_data$Label <- factor(train_data$Label)
train_labels <- factor(train_labels)


#'  ** 2 - Data Analysis ** 

#' Check the dimension of the problem
dim(train_data)
#' we can see that we have a number of predictors p = 63 lower than the number of samples n=172
#' therefore we expect the curse of dimensionality to be less severe in this task

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
#' We can see from the map that the number of outlier cells is small and there are no rows with a 
#' majority of outlier cells. 
  pca <- PcaHubert(train_predictors, k=63)
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
    rfe_result <- rfe(train_predictors, 
                      train_labels,
                      sizes=c(1:20, 30, 40, 50, 60), #number of features in the subsets to be tested
                      rfeControl= rfe_ctrl,
                      trControl = train_ctrl,
                      preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                      method = "glm", # Logistic Regression 
                      family = binomial(link = "logit")
                    )
    # Check the results 
    rfe_result
#' from the result we see that the RFE suggests using 19 predictors.
#' print the predictors  
   predictors(rfe_result)
   
   # Create new train data to be used later with only the predictors 
   # suggested by the RFE method
   optimal_preds <- append(predictors(rfe_result), 'Label')
   optimal_train_data <- train_data[ ,optimal_preds]
   
   # Visualize variable importance
   varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:19],
                             importance = varImp(rfe_result)[1:19, 1])
   
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
   #' We also add models with correlation filters and PCA applied with different
   #' thresholds
   models <- models %>%
     add_model(model_name = "baseline",
               x = initial_recipe)%>%
     add_model(model_name = "rfe",
               x = rfe_recipe) %>%
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
#'  Retrain and plot resampling results 
  set.seed(777)
  models <- train(models)
  models$model_fits %>% caret::resamples(.) %>% bwplot(.)

  models$model_fits %>%
  map(pluck(c("recipe", "term_info", "role"))) %>%
  map_int(~ sum(.x == "predictor"))
#' We can see from the results that models pca_.85, pca_.9 and pca_.95 achieve the same 
#' Median AUC/ROC therefore we will use the model that has a higher sensitivity 
#' which is pca_.85 
#' 
#' ** Summary of ROC of all LR models**
  resamps <- caret::resamples(models$model_fits)
  summary(resamps)
  
#' based on the results obtained, we will use the model pca_.85 as a best result 
#' for Logistic Regression => __LR: Median AUC/ROC = 0.8055556__
   exp.roc_scores <- c(exp.roc_scores, 0.8055556)
   exp.mcc_scores <- c(exp.mcc_scores, 0.4166667)
   exp.feat_select[1] <- "PCA 0.85"
   exp.n_predictors <- c(exp.n_predictors, 12)
   exp.model_params[1] <- "None"
#' 
#' **____________________________Linear Discriminant Analysis________________________________**
#' 
#' We repeat the same analysis with LDA 
#' ** Feature Selection **
#' 
  caretFuncs$summary <-twoClassSummary
   
  rfe_ctrl <- rfeControl(functions=caretFuncs, 
                          method = "repeatedcv", # Apply repeated CV
                          number = 5, # use 5 folds
                          repeats =5) # repeat 5 times
   
  train_ctrl <- trainControl(classProbs= TRUE, # needed to calculate the AUC/ROC 
                              summaryFunction = twoClassSummary) # needed to calculate ROC
  set.seed(777)
  rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1:10, 20, 25, 50), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "lda", # Linear Discriminant Analysis
  )
  rfe_result
#' from the result we see that the RFE suggests using 6 predictors.
#' print the predictors  
  predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(predictors(rfe_result), 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:6],
                            importance = varImp(rfe_result)[1:6, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Visualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  
  #' We can see from the plots that the model that uses 6 predictors yields 
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


   set.seed(777)
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
#' in 18 components
#' 
#' based on the results obtained, we will use the model pca_.9 as a best result 
#' for Linear Discriminant Analysis => __LDA: Median AUC/ROC = 0.8040123__ 
#' 
   exp.roc_scores <- c(exp.roc_scores, 0.8040123)
   exp.mcc_scores <- c(exp.mcc_scores, 0.4084912)
   exp.feat_select[2] <- "PCA 0.9"
   exp.n_predictors <- c(exp.n_predictors, 18)
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
   set.seed(777)
   rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1:10, 25, 50), # number of features in the subsets to be tested
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
   
   models <- models %>%
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
   
   set.seed(777)
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
#' in 6 components
#' 
#' based on the results obtained, we will use the model pca_.75 as a best result 
#' for Quadratic Discriminant Analysis => __QDA: Median AUC/ROC =  0.7638889__ 
   exp.roc_scores <- c(exp.roc_scores, 0.7638889)
   exp.mcc_scores <- c(exp.mcc_scores, 0.4260064)
   exp.feat_select[3] <- "PCA 0.75"
   exp.n_predictors <- c(exp.n_predictors, 6)
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
   set.seed(777)
   rfe_result <- rfe(train_predictors, 
                     train_labels,
                     sizes=c(1:10, 25, 50, 55:63), # number of features in the subsets to be tested
                     rfeControl= rfe_ctrl,
                     trControl = train_ctrl,
                     preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                     method = "knn", # k-NN
                     tuneLength = 3
   )
   # Check the results 
   rfe_result
   rfe_result$fit
   #' from the result we see that the RFE suggests using all predictors. 
   #' In this case RFE will be equal to the baseline
   #' print the predictors  
   predictors(rfe_result)
   
   
   # Visualize the RFE results
   ggplot(data = rfe_result, metric = "ROC") + theme_bw()
   
#' We can see from the plots that the model that uses all predictors yields 
#' the best re-sampled AUC/ROC
#' 
   # Create the pre-processing pipeline to be applied to all models
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
       method = "knn",
       tuneLength = 10
     )
   
   #' Add the different models to be trained
   models <- models %>%
     add_model(model_name = "baseline = RFE", x = initial_recipe)  %>%
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
   
   
   set.seed(777)
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
#' best median AUC/ROC is the baseline that uses all predictors and k = 21
#' 
#' based on the results obtained, we will use the model rfe as a best result 
#' for k-Nearest Neighbors => __k-NN: Median AUC/ROC =0.7916667__
  exp.roc_scores <- c(exp.roc_scores, 0.7916667)
  exp.mcc_scores <- c(exp.mcc_scores, 0.5093840)
  exp.feat_select[4] <- "Baseline"
  exp.n_predictors <- c(exp.n_predictors, 63)
  exp.model_params[4] <- "k=15"
#'
#' 
#' **__________________________________SVM, Linear_______________________________________**  
  
#' We repeat the same analysis with Support Vector Machine using a Linear kernel
#' Together with the feature selection parameters, we will fine-tune the parameter C of SVM
#' with GridSearch algorithm using the tuneGrid parameter of Caret's train() function
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
  
  set.seed(777)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1:12, 15, 25, 50), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "svmLinear",
                    tuneLength = 3 # fine tune SVM parameter
  )
  # Check the results 
  rfe_result
  rfe_result$fit
#' from the result we see that the RFE suggests using 8 predictors
#' 
  # print the predictors  
  predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(predictors(rfe_result), 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:8],
                            importance = varImp(rfe_result)[1:8, 1])
  
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
  # Create the pre-processing pipeline to be applied to all models
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
    add_model(model_name = "rfe", x = rfe_recipe) %>%
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
  
  
  set.seed(777)
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
#' best median AUC/ROC is the model preceded by PCA with threshold = 0.85 resulting
#' in 12 components and C = 1
#' 
#' based on the results obtained, we will use the model pca_.85 model as a best result 
#' for Linear SVM => __SVM Linear: Median AUC/ROC = 0.8055556__
  exp.roc_scores <- c(exp.roc_scores, 0.8055556)
  exp.mcc_scores <- c(exp.mcc_scores, 0.4366100)
  exp.feat_select[5] <- "PCA 0.85"
  exp.n_predictors <- c(exp.n_predictors, 12)
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
  set.seed(777)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 10, 20, 30, 40, 50, 60), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "svmRadial",
                    tuneLength = 3 # fine tune SVM parameters
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using all 63 predictors.
  #' with  sigma = 0.012255 and C = 0.5. In this case RFE = Baseline
  
  # Visualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  
  #' We can see from the plots that the model that uses 63 predictors yields 
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
                               summaryFunction = newSummary,
                               classProbs = TRUE),
      metric = "ROC",
      method = "svmRadial",
      tuneLength = 10 # fine tune SVM parameters
    )
  
  #' Add the different models to be trained
  
  models <- models %>%
    add_model(model_name = "baseline = rbf", x = initial_recipe)  %>%
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
  
  
  set.seed(777)
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
#' best median AUC/ROC is the one preceded by PCA with threshold = 0.85 resulting
#' in 12 principal components with  sigma = 0.05678335 and C = 0.25
#' 
#' based on the results obtained, we will use the baseline model as a best result 
#' for RBF SVM => __SVM RBF: Median AUC/ROC = 0.8194444__
  exp.roc_scores <- c(exp.roc_scores, 0.8194444)
  exp.mcc_scores <- c(exp.mcc_scores, 0.4166667)
  exp.feat_select[6] <- "PCA 0.85"
  exp.n_predictors <- c(exp.n_predictors, 12)
  exp.model_params[6] <- "sigma = 0.05678335, C = 0.25"
  
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
  set.seed(777)
  rfe_result <- rfe(train_predictors, 
                    train_labels,
                    sizes=c(1, 10, 20, 30, 40, 50, 60), # number of features in the subsets to be tested
                    rfeControl= rfe_ctrl,
                    trControl = train_ctrl,
                    preProcess=c("center", "scale"), # Normalization to zero-mean 1-std 
                    method = "rf", # random forest
                    tuneLength = 3
  )
  # Check the results 
  rfe_result
  rfe_result$fit
  #' from the result we see that the RFE suggests using 40 predictors
  # print the predictors  
  predictors(rfe_result)
  
  # Create new train data to be used later 
  optimal_preds <- append(predictors(rfe_result), 'Label')
  optimal_train_data <- train_data[ ,optimal_preds]
  
  # Visualize variable importance
  varimp_data <- data.frame(feature = row.names(varImp(rfe_result))[1:40],
                            importance = varImp(rfe_result)[1:40, 1])
  
  ggplot(data = varimp_data, 
         aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none")
  
  # Visualize the RFE results
  ggplot(data = rfe_result, metric = "ROC") + theme_bw()
  
  rf_predictors <- predictors(rfe_result)
  
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
  
  
  set.seed(777)
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
#' best median AUC/ROC is the one preceded by a correlation filter of 0.8 resulting in 28 
#' predictors 
#' 
#' based on the results obtained, we will use the baseline model as a best result 
#' for Random Forest => __RF: Median AUC/ROC = 0.7847222__
  exp.roc_scores <- c(exp.roc_scores, 0.7847222)
  exp.mcc_scores <- c(exp.mcc_scores, 0.4084912)
  exp.feat_select[7] <- "Corr 0.8"
  exp.n_predictors <- c(exp.n_predictors, 28)
  exp.model_params[7] <- "mtry = 19"
  
  
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
#'  is SVM  with RBF kernel using metrics C = 0.25 and sigma = 0.05678335 preceded by PCA with
#'  threshold = 0.85
#'  
#'  ** 5- Testing outlier removal **
#'  
#' we will use validation set approach to test the effect of removing outliers
  train_samples <- train_data[1:150, ]
  valid_samples <- train_data[151:172, ]
  
  preds <- train_samples[, -64]
  labels <- train_samples[, 64]
  
  new_preds <- train_samples[-c(110, 10, 55, 141, 62, 133), -64]
  new_labels <- train_samples[-c(110, 10, 55, 141, 62, 133), 64]
  
  #' __With outliers______ 
  trControl <- trainControl(savePredictions = TRUE, 
                            preProcOptions  = list(thresh = 0.85),
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction = newSummary)
  # Train SVM model
  set.seed(777)
  model_outs <- train(x= preds , y= labels, 
                     method = 'svmRadial',
                     trControl = trControl,
                     metric = 'ROC',
                     preProcess=c("center", "scale", "pca"),
                     tuneGrid = data.frame(C = 0.25, sigma =  0.05678335))

  val_pred_out <- predict(model_outs, valid_samples[, -64])
  acc1 <- sum(val_pred_out == valid_samples[, 64])/length(val_pred_out)
  
  #' __Without outliers______ 
  # Training control
  trControl <- trainControl(savePredictions = TRUE, 
                            preProcOptions  = list(thresh = 0.85),
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction = newSummary)
  # Train SVM model
  set.seed(777)
  model_no_outs <- train(x= new_preds , y= new_labels, 
                      method = 'svmRadial',
                      trControl = trControl,
                      metric = 'ROC',
                      preProcess=c("center", "scale", "pca"),
                      tuneGrid = data.frame(C = 0.25, sigma =  0.05678335))

  val_pred_no_out <- predict(model_no_outs, valid_samples[, -64])
  acc2 <- sum(val_pred_no_out == valid_samples[, 64])/length(val_pred_no_out)
  # We can see that acc2 < acc1 therefore removing outliers does not improve
  # the performance of the model. 
  # outliers will not be removed when training the final model
#'  
#'  ** 6- Final Model and Test Predictions **
#'
  
  # Training control
  trControl <- trainControl(savePredictions = TRUE, 
                            preProcOptions  = list(thresh = 0.85),
                            classProbs = TRUE, 
                            verboseIter = FALSE,
                            summaryFunction = newSummary)
  # Train SVM model
  final_model <- train(x= train_predictors , y= train_labels, 
                  method = 'svmRadial',
                  trControl = trControl,
                  metric = 'ROC',
                  preProcess=c("center", "scale", "pca"),
                  tuneGrid = data.frame(C = 0.25, sigma =  0.05678335))
  
  # Predict on the the test set
  test_preds <- predict(final_model, test_data)
  test_preds <- data.frame(test_ids, test_preds)
  
  # Feature indices
  features = 2:64
  

  # Save predictions 
  save(test_preds,  file = "0068096_Mouheb_ADMCIres.RData")
  # Save feature indices 
  save(features,  file = "0068096_Mouheb_ADMCIfeat.RData")

  ############################################ End of Task 2 ########################################  

  
  
  
  