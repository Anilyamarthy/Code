import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error

class Optimize_loss:
    def __init__(self,model,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        
    def loss_function(self,weights):
        self.model.set_weights(weights)
        y_pred = self.model.predict(self.X_train)
        return mean_squared_error(self.y_train, y_pred[:,0])
    
    def sooty_tern_Opt(self):
        # Define STO parameters
        population_size = 50
        max_generations = 1
        alpha = 0.1
        
        # Initialize population with random weights
        initial_weights = self.model.get_weights()
        population = [initial_weights.copy() for _ in range(population_size)]  
        
        # Perform STO optimization
        for generation in range(max_generations):
            for i in range(population_size):
                # Calculate the loss for the current individual
                loss = self.loss_function(population[i])
                
                neighbor_index = np.random.randint(0, population_size)
                neighbor = population[neighbor_index]
                
                # Update the current individual's weights using STO formula
                for j in range(len(neighbor)):
                    population[i][j] += alpha * (neighbor[j] - population[i][j])
                
                # Recalculate the loss for the updated weights
                new_loss = self.loss_function(population[i])
                
                # Update the weights if it reduces the loss
                if new_loss < loss:
                    loss = new_loss
                    self.model.set_weights(population[i])
                    
                    return self.model
        
    
