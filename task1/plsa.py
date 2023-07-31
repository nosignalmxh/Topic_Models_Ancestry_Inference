import torch
import pandas as pd
import time

data = pd.read_csv('task1/topic_data.txt', sep=' ', header=None)
X = torch.tensor(data.values, dtype=torch.float32)

n, m = X.shape 
K = 6  
max_iter = 300 

torch.manual_seed(0) 
L = torch.rand(n, K)
F = torch.rand(m, K)
t = torch.sum(X, axis=1).reshape(n, -1) 
L = L / t
F = F / torch.sum(F, axis=0)

log_likelihood_df = pd.DataFrame(columns=['Iteration', 'Log-Likelihood'])
start_time = time.time()
for iteration in range(max_iter):
    # E-step
    pi = L @ F.T + 1e-10  
    
    # M-step
    L = L * (X / pi @ F) / t
    for j in range(m):
        for k in range(K):
            F[j, k] = F[j, k] * torch.sum(X[:, j] * L[:, k] / pi[:, j])
    F = F / torch.sum(F, axis=0) 
    
    log_likelihood = torch.sum(X * torch.log(pi))
    print(f"Iteration {iteration+1}, log-likelihood: {log_likelihood.item()}")

    log_likelihood_df = log_likelihood_df.append({'Iteration': iteration+1, 'Log-Likelihood': log_likelihood.item()}, ignore_index=True)
end_time = time.time()
log_likelihood_df.to_excel('log_likelihood.xlsx', index=False)
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
