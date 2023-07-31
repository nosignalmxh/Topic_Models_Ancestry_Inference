# Install necessary packages
install.packages(c("genio", "data.table", "writexl"))

# Load necessary libraries
library(genio)
library(data.table)
library(writexl)

# Read the plink data
gen_data <- read_plink("task2/1kg_phase1_all_1m")
data_table <- as.data.table(t(gen_data$X))

# Initialization of variables
clusters <- 3           
samples <- nrow(data_table) 
features <- ncol(data_table) 
tau <- 1
a <- 1
b <- 1
c <- 1/clusters
theta_values <- matrix(rgamma(samples*clusters, 100, 0.01), nrow = samples, ncol = clusters)
beta_values <- array(rbeta(clusters*features*2, a, b), dim = c(clusters, features, 2))
continue_condition <- TRUE
iteration <- 0 
old_theta <- theta_values

# Function definitions
CheckConvergence <- function(new, old){
  d <- abs(new - old)
  sum(d)/sum(abs(new))
}
ExpectationLogBeta <- function(beta, index){
  digamma(beta[index]) - digamma(sum(beta))
}
ExpectationLogDirichlet <- function(dir){
  digamma(dir) - digamma(sum(dir))
}
LogSumExpCalculation <- function(x){
  x <- x - max(x)
  exp(x - log(sum(exp(x))))
}
NormalizeRows <- function(df) {
  row_sums <- rowSums(df)
  df_norm <- sweep(df, 1, row_sums, "/")
  return(df_norm)
}

# Main computation
while(continue_condition){
  iteration <- iteration + 1
  if (iteration %% 100 == 1) {
    print(paste("Iteration count:", iteration))
  }
  phi_values <- matrix(0, nrow = samples, ncol = clusters)
  xi_values <- matrix(0, nrow = samples, ncol = clusters)
  l <- sample(1:features, 1)
  data_table_l <- data_table[[l]] 
  new_beta_values <- matrix(nrow = clusters, ncol = 2)
  new_phi_values <- matrix(nrow = samples, ncol = clusters)
  new_xi_values <- matrix(nrow = samples, ncol = clusters)  
  continue_inner_condition <- TRUE
  
  while(continue_inner_condition){   
    e_log_dirichlet <- apply(theta_values, MARGIN = 1, FUN = ExpectationLogDirichlet) 
    e_log_beta_1 <-  apply(beta_values[ , l, ], MARGIN = 1, FUN = function(x) ExpectationLogBeta(x, 1))
    e_log_beta_2 <-  apply(beta_values[ , l, ], MARGIN = 1, FUN = function(x) ExpectationLogBeta(x, 2))
    x_values <- e_log_dirichlet + e_log_beta_1 
    y_values <- e_log_dirichlet + e_log_beta_2
    new_phi_values <- t(apply(x_values, MARGIN = 2, FUN = LogSumExpCalculation))
    new_xi_values <- t(apply(y_values, MARGIN = 2, FUN = LogSumExpCalculation))
    new_beta_values[ , 1] <- a + as.vector(t(data_table_l) %*% new_phi_values)
    new_beta_values[ , 2] <- b + as.vector(t(2 - data_table_l) %*% new_xi_values)

    if(CheckConvergence(new_beta_values, beta_values[ , l, ]) + CheckConvergence(new_phi_values, phi_values) + CheckConvergence(new_xi_values, xi_values) < 1e-3){
      continue_inner_condition <- FALSE
    }
    
    phi_values <- new_phi_values
    xi_values <- new_xi_values
    beta_values[ , l, ] <- new_beta_values
    
  }
    rho <- (tau + iteration)^(-0.5)
    theta_values <- (1 - rho) * theta_values + rho * (c + features * (data_table_l * phi_values + (2 - data_table_l) * xi_values))
  if (iteration >= 10000 && CheckConvergence(old_theta, theta_values) < 1e-3) {
    continue_condition <- FALSE
  }
  old_theta <- theta_values
  
}

# Writing the results
theta_data_frame <- as.data.frame(theta_values)
theta_normalized <- NormalizeRows(theta_data_frame)
write_xlsx(theta_normalized, "output.xlsx")
