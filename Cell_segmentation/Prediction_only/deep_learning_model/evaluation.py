import torch
import numpy as np

# Prediction : Prediction
# GT : Ground Truth

def get_MSE(Prediction_vector, GT_vector):
	MSE = sum((Prediction_vector - GT_vector)**2)/len(GT_vector)
	return MSE
def CrossEntropy(yHat, y):
    entropy_mat = np.zeros(yHat.shape)
    entropy_mat[y == 1] = -np.log(yHat)[y == 1]
    entropy_mat[y == 0] = -np.log(1 - yHat)[y == 0]
    return entropy_mat