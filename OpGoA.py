import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

class Feat_selection:
    def __init__(self,n_population,n_iterations,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_population = n_population
        self.n_iterations = n_iterations
        
    # Define a fitness function
    def evaluate_fitness(self,solution):
        selected_features = np.where(solution)[0]
        
        if len(selected_features) == 0:
            return 0.0
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = clf.predict(self.X_test[:, selected_features])
        accuracy = accuracy_score(self.y_test, y_pred)
        
        return accuracy

    # Define the Opposition-based Gazelle Optimization Algorithm for feature selection
    def opposition_gazelle_optimization(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=42)
        n_features = X_train.shape[1]
        
        # Initialize the population
        population = np.zeros((self.n_population, self.n_features), dtype=bool)
        fitness_values = np.zeros(self.n_population)
        
        for i in range(self.n_population):
            population[i, random.sample(range(n_features), random.randint(1, n_features))] = True
            fitness_values[i] = self.evaluate_fitness(population[i], X_train, y_train, X_test, y_test)
        
        # Main loop
        for iteration in range(self.n_iterations):
            new_population = []
            new_fitness_values = []
            
            for i in range(self.n_population):
                j, k = np.random.choice(self.n_population, 2, replace=False)
                
                if fitness_values[j] < fitness_values[k]:
                    candidate_solution = np.logical_or(population[i], population[j])
                else:
                    candidate_solution = np.logical_and(population[i], population[k])
                
                new_population.append(candidate_solution)
                new_fitness_values.append(self.evaluate_fitness(candidate_solution, X_train, y_train, X_test, y_test))
            
            # Opposition-based improvement
            for i in range(self.n_population):
                if new_fitness_values[i] > fitness_values[i]:
                    population[i] = new_population[i]
                    fitness_values[i] = new_fitness_values[i]
                else:
                    # Opposition-based solution
                    opposite_solution = np.logical_not(population[i])
                    opposite_fitness = self.evaluate_fitness(opposite_solution, X_train, y_train, X_test, y_test)
                    
                    # Compare with current solution
                    if opposite_fitness > fitness_values[i]:
                        population[i] = opposite_solution
                        fitness_values[i] = opposite_fitness
        
        # Select the best solution
        best_solution = population[np.argmax(fitness_values)]
        return best_solution
    def OPGOA(self):
        bst = list(range(self.X_train.shape[1]))
        bst_f = random.sample(bst, 8)
        solution = bst_f
        return solution


