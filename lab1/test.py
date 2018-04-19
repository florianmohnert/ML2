### 1.7 Independent Component Analysis

def ICA(X, activation_function, learning_rate=0.1):
    '''
    Independent Component Analysis. Returns the demixing matrix, W.
    '''

    iteration = 0
    number_of_signals = X.shape[0]
    W = np.random.rand(number_of_signals, number_of_signals) * 0.01  # blows up without the *0.01
    grad = np.random.rand(number_of_signals, number_of_signals)

    while np.linalg.norm(grad) > 1e-5:
        a = np.matmul(W, X)
        x_prime = np.matmul(np.transpose(W), a)
        grad = W + np.matmul(activation_function(a), np.transpose(x_prime))
        W = W + learning_rate * grad
        iteration += 1

    print("Number of iterations until convergence:", iteration)

    return W


W_est = ICA(Xw, phi_3)  # Compare with ICA(X)
