import numpy as np
import pandas as pd
import glob
# Compute the Euclidean distance matrix between points
def euclidean_distance_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return D
def point_sigam(D):
    n = D.shape[0]
    m = np.mean(D, axis=1)
    diffs = D - m[:, np.newaxis]
    sigam = np.mean(diffs ** 2, axis=1)
    return sigam
# Calculate the affinity matrix between points
def affinity_matrix(D, sigma):
    A = np.exp(-D ** 2 / (2 * sigma ** 2))

    np.fill_diagonal(A, 0)
    return A
def normalize_rows(A):
    """
    Each row in matrix A is normalized
    """
    row_sums = A.sum(axis=1)
    return A / row_sums[:, np.newaxis]

def normalize_affinity_matrix(A):
    B = normalize_rows(A)
    return B

def random_neighbor_graph(A):
    n = A.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = np.random.binomial(1, A[i, j])
                S[j, i] = np.random.binomial(1, A[j, i])
    return S

def combine_matrices(B, S):
    B_expanded = np.broadcast_to(B, S.shape)
    C = np.where(S == 0, 0, B_expanded)
    return C

# Calculate the outlier probability of a data point
def outlier_probability(C):
    n = S.shape[0]
    p = np.zeros(n)
    for i in range(n):
        if np.sum(C[i]) == 0:
            p[i] = 1.0
        else:
            p[i] = np.prod(1 - C[:, i])
    return p

def top_outliers(p,q):
    n = p.shape[0]
    top_pct = int(np.ceil(n * q))  # Take the first 10% of the number
    idx = np.argsort(p)  # Sorts p and returns its index position
    return idx[-top_pct:]  # Returns the top 10% index position

# Determine whether the data point is an outlier according to the outlier probability and threshold
def detect_outliers(p, threshold):
    
    idx = np.where(p > threshold)[0]
    return idx


if __name__ == '__main__':
   
    XX= excel_files = glob.glob('e:/pythonpuod/pythonpuod/1(10).xlsx')
    # XX = excel_files = glob.glob('new4.xlsx')
    #ts=[0.95,0.86,0.75,0.74,0.56,0.41,0.40,0.25,0.15,0.11]  # What percentage of output points #
    ts=[0.30,0.30,0.12,0.08,0.06,0.04,0.11,0.24]
    # Loop reads the first worksheet of each Excel file
    for l, file in enumerate(excel_files):
        t=ts[0]
        df = pd.read_excel(file, sheet_name="Sheet6")
        # Extract coordinates of data points
        X = df.iloc[:, 0:2].values
        # Compute Euclidean distance matrix and affinity matrix
        D = euclidean_distance_matrix(X)
        sigam=np.std(D)
        # sigam=point_sigam(D)
        #sigam = 0.7
        A = affinity_matrix(D, sigam)
        B = normalize_affinity_matrix(A)

        n=100 # Number of iterations

        for i in range(n):
        # Generate a random neighborhood map
            S = random_neighbor_graph(A)
            C=combine_matrices(B, S)
        # Calculate the probability of outliers and determine outliers

            p = outlier_probability(C)
            f=0
            k=top_outliers(p,f) 

        w = []
        for i in range(n):
            result = p
            w.append(result)
        result_matrix = np.array(w)
        # print(result_matrix) 

        votes = result_matrix.tolist()      
        h=1                 ##############0-1
        for i in range(len(votes)):
            vote = votes[i]
            sorted_vote = sorted(vote, reverse=True)
            q=len(sorted_vote)
            for j in range(q):        
                if j < h*q:
                    score = (q * h - (j - 1)) / (q * h)
                else:
                    score=0
                vote[vote.index(sorted_vote[j])] = score
            
        ############Output outliers in percentage terms###########
        candidates_scores = [0] * len(votes[0])
        for i in range(len(votes)):
            vote = votes[i]
            for j in range(len(vote)):
                score = vote[j]
                candidates_scores[j] += score

        # binary_scores = [int(score >= 0.5) for score in candidates_scores]
        # print(binary_scores)
        sorted_candidates = sorted(enumerate(candidates_scores), key=lambda x: x[1], reverse=True)
       
        top_10_percent_index = int(len(sorted_candidates) * t)
        threshold = sorted_candidates[top_10_percent_index][1]

        binary_scores = [int(score >= threshold) for score in candidates_scores]
        print(binary_scores)
        ################## Calculate metrics###############
        AC= df.iloc[:,2].values
        TP=0
        FP=0
        FN=0
        TN=0
        for i in range(len(binary_scores)):
            if(AC[i]==0 and binary_scores[i]==0):
                TP=TP+1
            if(AC[i]==1 and binary_scores[i]==0):
                FP=FP+1
            if (AC[i] == 0 and binary_scores[i] == 1):
                FN=FN+1
            if (AC[i] == 1 and binary_scores[i] == 1):
                TN=TN+1
        print(TP,FP,FN,TN)
        ZQL=(TP+TN)/(TP+TN+FP+FN)
        JQL=(TP)/(TP+FP)
        ZHL=(TP)/(TP+FN)
        JZL=(FP)/(TN+FP)
        print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)
