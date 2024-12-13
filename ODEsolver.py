#!/usr/bin/env python
# coding: utf-8

# ODE solver class
import numpy as np
from tqdm import tqdm

class ODEsolver:
    # start by making the constructor using __init__ this is a initializer in brackets we feed in arguments
    def __init__(self, f): # self refers to the problem we are running this function on (the model) 
        self.f = f         # here f would be the right-hand side of the function that defines the function f(u,t)
        
    # This is a function that progresses the problem one-time step forward this is the parent class that
    # We  subclasses 
    def advance(self): 
        raise NotImplementedError # to raise this error if we try to call this thing

    #to solve ODE We need some initial conditions, so we define a function that sets initial conditions
    def set_initial_conditions(self, U0): # lets say U is our differential equations than U0 are our initial conditions
                                           # Because it is a set of equations we use this method to arrange them 
        if isinstance(U0, (int, float)):  # check if the initial condition is a single number, like an integer or a float
                                           # If it is scalar, then the ODE has just one equation c
            self.number_of_eqns = 1  # set there to be only one dependent variable
            U0 = float(U0)                #and we convert the initial condition to a float for later arithmetic
             
        else:  # but if we have a system of equations (which we do for he context of this problem)
            U0 = np.asarray(U0)    # if in a system convert to numpy array
            self.number_of_eqns = U0.size #sets #of equations by counting elements w/in the U0 array
            self.U0 = U0 # stores what ever value we set it to to operate in the class
     
# now for a method that calls this one repeatedly and that solves the problem any class must use the above super class
# and it must use the advance() 
        
    def solve(self, time_points):# argument takes time point because every equation will be solved for many time points to show how the population changes over time
    
        self.t = np.asarray(time_points)# call time points t put them in array
        n = self.t.size    # number of time points dictating the size of array

        self. u = np.zeros((n, self.number_of_eqns)) #make an empty array that is the size of the t array, that holds our solutions at each time step for each of the equations 
           # [start_index:end_index] So if we have 3 equations we would have n elements in each equation
        self.u[0, :] = self.U0  # set t=0 to correspond with our initial conditions

        # Now we Integrate to solve the derivatives at each time step
        for i in tqdm(range(n - 1), ascii=True): # its minus one because the first variable in the array is filled with our initial conditions
            self.i = i
            self.u[i + 1] = self.advance() # for the next step after i we call the advanced method

        return self.u[:i+2], self.t  # return the solutions all the way to i+2 

# now lets make a sub-class that actually solves the ODE

class ForwardEuler(ODEsolver): # simplest method for solving ODE's
    def advance(self): # implement this to the solver
        # Here, we set the solutions, the function, the index of values, and the time to the corresponding values of the problem we apply the class to
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]  # set the size of each time step for when we integrate
        return u[i, :] + dt * f(u[i, :], t[i]) # for all equations at the ith step we want to multiply by our time step by our function

    



