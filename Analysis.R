#loading libraries 
library(dplyr)
library(zellkonverter)
library(matrixStats)
library(future.apply)
library(caret)
library(Metrics)
library(doParallel)
library(Matrix)
library(tibble)
library(xgboost)
library(reticulate)

#Processing pseudo-bulk and corresponding metadata 

# Set the directory containing .h5ad files
input_dir <- "/path/"
output_dir <- "/path/"

# Get all .h5ad files in the directory
h5ad_files <- list.files(input_dir, pattern = "\\.h5ad$", full.names = TRUE)

# Loop through all .h5ad files
for (i in seq_along(h5ad_files)) {
  file_path <- h5ad_files[i]
  filename <- tools::file_path_sans_ext(basename(file_path))
  
  cat("Processing file", i, "of", length(h5ad_files), ":", filename, "\n")
  
  tryCatch({
    # Read the .h5ad file
    sce1 <- readH5AD(file_path)
    
    # Expression matrix
    mat <- assay(sce1, "X")
    mat_dense <- as.matrix(mat)
    
    # Metadata matrix for each cell
    matdata <- colData(sce1)
    matdata <- as.data.frame(matdata)
    
    # Get cell type information together with donor ID
    cell_cat <- paste(matdata$Supertype, matdata$Donor.ID, sep = "_")
    
    # Create pseudobulk by median aggregation
    pseudobulk <- sapply(unique(cell_cat), function(grp) {
      cols <- which(cell_cat == grp)   
      apply(mat_dense[, cols, drop = FALSE], 1, median)
    })
    
    pseudobulk <- as.data.frame(pseudobulk)
    colnames(pseudobulk) <- unique(cell_cat)
    
    # Save as .rds file
    output_file <- file.path(output_dir, paste0(filename, ".rds"))
    saveRDS(pseudobulk, output_file)
    
  }, error = function(e) {
    cat("Error processing", filename, ":", e$message, "\n")
  })
}

#merge further in chunks to create one dataframe
# List all .rds files
files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE)

# Read all into a list
obj_list <- lapply(files, readRDS)

# Function to merge two dataframes/matrices by rownames and take row medians for overlaps
merge_with_median <- function(C1, C2) {
  merged <- merge(C1, C2, by = "row.names", all = TRUE)
  rownames(merged) <- merged$Row.names
  merged$Row.names <- NULL
  
  common_cols <- intersect(colnames(C1), colnames(C2))
  
  for (col in common_cols) {
    colx <- paste0(col, ".x")
    coly <- paste0(col, ".y")
    if (colx %in% names(merged) && coly %in% names(merged)){
      merged[[col]] <- rowMedians(as.matrix(merged[, c(colx, coly)]), na.rm = TRUE)
      merged[, c(colx, coly)] <- NULL
    }
  }
  return(merged)
}

# Iteratively merge all objects in the list
merged_all <- Reduce(merge_with_median, obj_list)

# Save final merged object

#for A9
saveRDS(merged_all, file.path(output_dir, "SEAAD_A9_merged.rds"))
#for MTG
saveRDS(merged_all, file.path(output_dir, "SEAAD_MTG_merged.rds"))

#-----------------------------------------------------------------------------------------------------------------

#combining and saving metadata
# List all .h5ad files in the folder
files <- list.files(input_dir, pattern = "\\.h5ad$", full.names = TRUE)

# Read, extract metadata, and add cat column for each file
all_metadata <- lapply(files, function(f) {
  obj <- readH5AD(f)
  df <- as.data.frame(colData(obj))
  df$cat <- paste(df$Supertype, df$Donor.ID, sep = "_")
  df
})
# Combine everything
metadat <- do.call(rbind, all_metadata)
# Keep only first occurrence of each cat
metadat <- metadat[!duplicated(metadat$cat), ]

#for A9
saveRDS(metadat, file.path(output_dir, "metadata_A9.rds"))
#for MTG
saveRDS(metadat, file.path(output_dir, "metadata_MTG.rds"))

#-----------------------------------------------------------------------------------------------------------------

#Ordering genes further deviant_genes.py in one list

#loading orders gene list (h5ad_rank_deviance.csv.gz) from deviant_genes.py method
genes_a9_rank <- read.csv("/path/a9.csv.gz")

#removing mitochondrial genes
mito_genes <- grep("^MT-", genes_a9_rank$X)
genes_a9_rank <- genes_a9_rank[-mito_genes, ]

# geometric mean function that ignores zeroes
geo_mean <- function(x) exp(mean(log(x[x > 0])))
# compute geometric mean per row for A9 ranked genes
row_geo_means <- apply(genes_a9_rank[, -1], 1, geo_mean)
# set the names of the vector to the gene IDs (assumes gene IDs are in genes_a9_rank$X)
names(row_geo_means) <- genes_a9_rank$X
# sort the geometric means (ascending) and store
row_geo_means_a9 <- sort(row_geo_means)

genes_mtg_rank <- read.csv("/path/mtg.csv.gz")

mito_genes <- grep("^MT-", genes_mtg_rank$X)
genes_mtg_rank <- genes_mtg_rank[-mito_genes, ]

geo_mean <- function(x) exp(mean(log(x[x > 0])))
row_geo_means <- apply(genes_mtg_rank[, -1], 1, geo_mean)
names(row_geo_means) <- genes_a9_rank$X
row_geo_means_mtg <- sort(row_geo_means)

# union of gene names present in either A9 or MTG geometric-mean vectors
genes <- unique(c(names(row_geo_means_a9), names(row_geo_means_mtg)))

# get the ranks (positions) of each gene in the sorted A9 and MTG lists
rank1 <- match(genes, names(row_geo_means_a9))
rank2 <- match(genes, names(row_geo_means_mtg))

# compute average rank across the two lists; order genes by average rank
avg_rank <- (rank1 + rank2) / 2
sorted_genes <- genes[order(avg_rank)]

#taking the top genes (optimized as a function of performance)
sorted_top_genes <- sorted_genes[1:20000]

# save the resulting ordered gene list to disk for later use
saveRDS(sorted_top_genes, "/path/gene_list_for_filtering.rds")

#-----------------------------------------------------------------------------------------------------------------

#Final submission - Task1

# load A9 merged expression matrix and its metadata
merge_dat_A9 <- readRDS("/path/SEAAD_A9_merged.rds")
metadat_A9 <- readRDS("/path/metadata_A9.rds")

# reorder A9 columns to match metadata ordering (assumes metadat_A9$cat has column IDs)
merge_dat_A9 <- merge_dat_A9[, metadat_A9$cat]

# load MTG merged expression matrix and its metadata
merge_dat_MTG <- readRDS("/path/SEAAD_MTG_merged.rds")
metadat_MTG   <- readRDS("/path/metadata_MTG.rds")

# reorder MTG columns to match metadata ordering
merge_dat_MTG <- merge_dat_MTG[, metadat_MTG$cat]

# append brain-region suffixes to column names to keep samples unique after combining
colnames(merge_dat_A9) <- paste0(colnames(merge_dat_A9), "_A9")
colnames(merge_dat_MTG) <- paste0(colnames(merge_dat_MTG), "_MTG")

# combine A9 and MTG sample matrices by columns
comb_mat <- cbind(merge_dat_A9, merge_dat_MTG)

# load filtered/ordered gene list and subset combined matrix to those genes (rows)
genes <- readRDS("/path/gene_list_for_filtering.rds")
comb_mat <- comb_mat[genes, , drop = FALSE]

# transpose so samples are rows and genes are columns, convert to matrix
X <- t(comb_mat)
X <- as.matrix(X)

# save gene order (column names of X) for later reference
gene_order <- colnames(X)
saveRDS(gene_order, "/path/gene_order.rds")

# -----------------------------
# Build XGBoost multi-class models for several labels
# -----------------------------

# ADNC: create response vector by concatenating A9 and MTG metadata
y <- c(as.numeric(metadat_A9$ADNC), as.numeric(metadat_MTG$ADNC))

# factorize labels and convert to 0-based integers required by xgboost
y_factor <- as.factor(y)
train_y <- as.integer(y_factor) - 1
levels_used_ADNC <- levels(y_factor)   # store original factor levels

# create xgboost DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X), label = train_y)

# set up multiclass parameters
num_class <- length(unique(train_y))
params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = num_class
)

# cross-validated training to find optimal number of rounds (10-fold, early stop)
set.seed(123)
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 100,
  nfold = 10,
  stratified = TRUE,
  early_stopping_rounds = 20,
  maximize = FALSE,
  verbose = 1
)

# best number of boosting rounds from CV
best_nrounds <- cv_model$best_iteration

# train final model with best number of rounds
final_model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

# save model and factor level mapping
saveRDS(final_model, "/path/Submission_model1_ADNC.rds")
saveRDS(levels_used_ADNC, "/path/Submission_model1_ADNC_level.rds")

# -----------------------------
# Repeat same pipeline for Braak
# -----------------------------

y <- c(as.numeric(metadat_A9$Braak), as.numeric(metadat_MTG$Braak))

y_factor <- as.factor(y)
train_y <- as.integer(y_factor) - 1  
levels_used_Braak <- levels(y_factor)      

dtrain <- xgb.DMatrix(data = as.matrix(X), label = train_y)

num_class <- length(unique(train_y))
params <- list(booster = "gbtree",
    objective = "multi:softprob", 
    eval_metric = "mlogloss",
    num_class = num_class)
  
set.seed(123)
cv_model <- xgb.cv(params = params,
    data = dtrain,
    nrounds = 100,         
    nfold = 10,           
    stratified = TRUE,     
    early_stopping_rounds = 20,
    maximize = FALSE,
    verbose = 1)
  
best_nrounds <- cv_model$best_iteration
  
final_model <- xgboost(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 1)

saveRDS(final_model, "/path/Submission_model1_Braak.rds")
saveRDS(levels_used_Braak, "/path/Submission_model1_Braak_level.rds")

# -----------------------------
# Repeat same pipeline for Thal
# -----------------------------

y <- c(as.numeric(metadat_A9$Thal), as.numeric(metadat_MTG$Thal))

y_factor <- as.factor(y)
train_y <- as.integer(y_factor) - 1   
levels_used_Thal <- levels(y_factor)     

dtrain <- xgb.DMatrix(data = as.matrix(X), label = train_y)

num_class <- length(unique(train_y))
params <- list(booster = "gbtree",
    objective = "multi:softprob", 
    eval_metric = "mlogloss",
    num_class = num_class)
  
set.seed(123)
cv_model <- xgb.cv(params = params,
    data = dtrain,
    nrounds = 100,         
    nfold = 10,            
    stratified = TRUE,     
    early_stopping_rounds = 20,
    maximize = FALSE,
    verbose = 1)
  
best_nrounds <- cv_model$best_iteration
  
final_model <- xgboost(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 1)

saveRDS(final_model, "/path/Submission_model1_Thal.rds")
saveRDS(levels_used_Thal, "/path/Submission_model1_Thal_level.rds")

# -----------------------------
# Repeat same pipeline for CERAD
# -----------------------------

y <- c(as.numeric(metadat_A9$CERAD), as.numeric(metadat_MTG$CERAD))

y_factor <- as.factor(y)
train_y <- as.integer(y_factor) - 1  
levels_used_CERAD <- levels(y_factor)   

dtrain <- xgb.DMatrix(data = as.matrix(X), label = train_y)

num_class <- length(unique(train_y))
params <- list(booster = "gbtree",
    objective = "multi:softprob", 
    eval_metric = "mlogloss",
    num_class = num_class)
  
set.seed(123)
cv_model <- xgb.cv(params = params,
    data = dtrain,
    nrounds = 100,         
    nfold = 10,        
    stratified = TRUE,   
    early_stopping_rounds = 20,
    maximize = FALSE,
    verbose = 1)
  
best_nrounds <- cv_model$best_iteration
  
final_model <- xgboost(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 1)

saveRDS(final_model, "/path/Submission_model1_CERAD.rds")
saveRDS(levels_used_CERAD, "/path/Submission_model1_CERAD_level.rds")

#-----------------------------------------------------------------------------------------------------------------

#converting the model objects to python (for running inside docker python based image)

# Convert XGBoost models from RDS to JSON
model <- readRDS("/path/Submission_model1_ADNC.rds")
#Save as JSON (XGBoost native format)
xgb.save(model, "/path/Submission_model1_ADNC.json")

model <- readRDS("/path/Submission_model1_Braak.rds")
xgb.save(model, "/path/Submission_model1_Braak.json")

model <- readRDS("/path/Submission_model1_CERAD.rds")
xgb.save(model, "/path/Submission_model1_CERAD.json")

model <- readRDS("/path/Submission_model1_Thal.rds")
xgb.save(model, "/path/Submission_model1_Thal.json")

# Convert levels to Python pickle format using reticulate
pickle <- reticulate::import("pickle")
builtins <- import_builtins()

# Convert gene_order to pkl 
gene_order <- readRDS("/path/gene_order.rds")
py_gene_order <- r_to_py(gene_order)
pickle_path <- "/path/gene_order.pkl"

file_handle <- builtins$open(pickle_path, "wb")
pickle$dump(py_gene_order, file_handle)
file_handle$close()

# Convert level to pkl 
level <- readRDS("/path/Submission_model1_ADNC_level.rds")
py_level <- r_to_py(level)
pickle_path <- "/path/Submission_model1_ADNC_level.pkl"

file_handle <- builtins$open(pickle_path, "wb")
pickle$dump(py_level, file_handle)
file_handle$close()

level <- readRDS("/path/Submission_model1_Braak_level.rds")
py_level <- r_to_py(level)
pickle_path <- "/path/Submission_model1_Braak_level.pkl"

file_handle <- builtins$open(pickle_path, "wb")
pickle$dump(py_level, file_handle)
file_handle$close()

py_level <- readRDS("/path/Submission_model1_CERAD_level.rds")
py_level <- r_to_py(py_level)
pickle_path <- "/path/Submission_model1_CERAD_level.pkl"

file_handle <- builtins$open(pickle_path, "wb")
pickle$dump(py_level, file_handle)
file_handle$close()

py_level <- readRDS("/path/Submission_model1_Thal_level.rds")
py_level <- r_to_py(py_level)
pickle_path <- "/path/Submission_model1_Thal_level.pkl"

file_handle <- builtins$open(pickle_path, "wb")
pickle$dump(py_level, file_handle)
file_handle$close()

#-----------------------------------------------------------------------------------------------------------------

#Final submission - Task2

# --- load expression matrices and metadata -----------------------------------
merge_dat_A9 <- readRDS("/path/SEAAD_A9_merged.rds")
metadat_A9  <- readRDS("/path/metadata_A9.rds")

# reorder A9 columns to the order in metadata (metadat_A9$cat holds column/sample IDs)
merge_dat_A9 <- merge_dat_A9[, metadat_A9$cat]

merge_dat_MTG <- readRDS("/path/SEAAD_MTG_merged.rds")
metadat_MTG   <- readRDS("/path/metadata_MTG.rds")

# reorder MTG columns to the order in metadata
merge_dat_MTG <- merge_dat_MTG[, metadat_MTG$cat]

# append region suffix to column names so sample IDs remain unique after combining
colnames(merge_dat_A9) <- paste0(colnames(merge_dat_A9), "_A9")
colnames(merge_dat_MTG) <- paste0(colnames(merge_dat_MTG), "_MTG")

# combine A9 and MTG by columns -> combined genes x samples matrix
comb_mat <- cbind(merge_dat_A9, merge_dat_MTG)

# continuous target: concatenated percent.6e10.positive.area from both metadata objects
y <- c(as.numeric(metadat_A9$percent.6e10.positive.area),
       as.numeric(metadat_MTG$percent.6e10.positive.area))

# --- (optional) keep original gene order from raw input (commented out) ------
# If needed, match comb_mat rows to an original H5AD gene order.

# --- filter to precomputed gene list (top genes) -----------------------------

genes <- readRDS("/path/gene_list_for_filtering.rds")
comb_mat <- comb_mat[genes, , drop = FALSE]  # subset rows (genes)

# transpose to samples x genes and coerce to numeric matrix for xgboost
X <- t(comb_mat)
X <- as.matrix(X)

# save the gene order used (column names of X)
genes <- rownames(comb_mat)
saveRDS(genes, "/path/gene_order_sub.rds")

# --- two-stage training function -------------------------------------------

train_two_stage_model <- function(X, y){
  # X: samples x features matrix
  # y: numeric continuous target (e.g., percent positive area)
  
  # create binary label for presence/absence using threshold 0.5
  # (choice of 0.5 is dataset-specific; >0.5 => "non-zero/high" class)
  y_binary <- as.numeric(y > 0.5)
  
  # classification stage: predict probability of being non-zero
  dtrain_class <- xgb.DMatrix(data = X, label = y_binary)
  params_class <- list(
    booster = "gbtree",
    objective = "binary:logistic",  # binary classification
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  set.seed(123)
  cv_class <- xgb.cv(
    params = params_class,
    data = dtrain_class,
    nrounds = 100,
    nfold = 5,
    early_stopping_rounds = 10,
    maximize = FALSE,
    verbose = 1
  )
  best_nrounds_class <- cv_class$best_iteration
  
  # final classification model trained on all data
  class_model <- xgboost(
    params = params_class,
    data = dtrain_class,
    nrounds = best_nrounds_class,
    verbose = 1
  )
  
  # regression stage: only use samples with y > 0.5 (non-zero group)
  non_zero_idx <- y > 0.5
  X_nonzero <- X[non_zero_idx, , drop = FALSE]
  y_nonzero <- log(y[non_zero_idx])           # log-transform target for regression
  
  # note: log() requires y > 0; using only non-zero subset avoids log(0)
  dtrain_reg <- xgb.DMatrix(data = X_nonzero, label = y_nonzero)
  params_reg <- list(
    booster = "gbtree",
    objective = "reg:squarederror",  # regression
    eval_metric = "rmse",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  set.seed(123)
  # use min(5, number_of_nonzero_samples) folds to avoid nfold > n_samples error
  cv_reg <- xgb.cv(
    params = params_reg,
    data = dtrain_reg,
    nrounds = 100,
    nfold = min(5, sum(non_zero_idx)),
    early_stopping_rounds = 10,
    maximize = FALSE,
    verbose = 1
  )
  best_nrounds_reg <- cv_reg$best_iteration
  
  # final regression model trained on non-zero samples
  reg_model <- xgboost(
    params = params_reg,
    data = dtrain_reg,
    nrounds = best_nrounds_reg,
    verbose = 1
  )
  
  # return both models as a list
  return(list(classification_model = class_model, regression_model = reg_model))
}

# run two-stage training for percent.6e10.positive.area and save
models <- train_two_stage_model(X, y)
saveRDS(models, "/path/model_6e10.rds")

# repeat for AT8 percent positive area target
y <- c(as.numeric(metadat_A9$percent.AT8.positive.area),as.numeric(metadat_MTG$percent.AT8.positive.area))

models <- train_two_stage_model(X, y)
saveRDS(models, "/path/model_AT8.rds")

#-----------------------------------------------------------------------------------------------------------------

# export trained XGBoost models to JSON for use in Python --------------------

# load model (example path; assumes the RDS contains the list with two xgboost objects)
model <- readRDS("/path/model_6e10_sub.rds")
# save native xgboost JSON model files (classification + regression)
xgb.save(model$classification_model, "/path/class_model_6e10_sub.json")
xgb.save(model$regression_model,     "/path/reg_model_6e10_sub.json")

model <- readRDS("/path/model_AT8_sub.rds")
xgb.save(model$classification_model, "/path/class_modelAT8_sub.json")
xgb.save(model$regression_model,     "/path/reg_model_AT8_sub.json")

# --- convert R objects to Python pickle via reticulate ----------------------

pickle <- reticulate::import("pickle")    # Python pickle module
builtins <- import_builtins()             # convenience for open() - requires reticulate::import_builtins()

# read RDS gene order and convert to Python list
gene_order <- readRDS("/path/gene_order_sub.rds")
py_gene_order <- r_to_py(gene_order)

pickle_path <- "/path/gene_order_sub.pkl"
file_handle <- builtins$open(pickle_path, "wb")  # open binary file for writing
pickle$dump(py_gene_order, file_handle)          # write python object to pickle
file_handle$close()

#-----------------------------------------------------------------------------------------------------------------

