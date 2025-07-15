
import numpy as np
import matlab.engine
 

class C4DynamicModel:
    
    def __init__(self, config):
               
        # setup matlab engine
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(config.paths.matlab_code_dir)
        
        # CO2 and light steps in the order of measurement/simulation
        self.light_steps_simulation = [1800, 1100, 500, 300, 150, 50]
        self.co2_steps_simulation = [400, 600, 800, 1000, 1250, 300, 250, 200, 100, 75, 25]
        self.order_co2_steps = np.argsort(self.co2_steps_simulation)
        self.order_light_steps = np.argsort(self.light_steps_simulation)

    def simulate(self, params):
        """
        Calls matlab functions to simulate A/CO2 and A/light curves using the given
        parameter set)
    
        Args:
            params: C4 model parameter
    
        Returns:
            A tuple (a_co2, a_light), which are the simulated A/CO2 and A/light curves
        """        
        
        # Convert parameters to MATLAB format
        params_matlab = matlab.double(params)  # MATLAB expects double precision arrays   
        
        # Solve the ODE in MATLAB
        a_co2_matlab = self.eng.simulate_ACI(params_matlab, 'equilibrator')
        a_light_matlab = self.eng.simulate_AQ(params_matlab, 'equilibrator')
        
        # Convert the MATLAB result back to NumPy arrays
        a_co2 = np.array(a_co2_matlab)
        a_light = np.array(a_light_matlab)
    
        a_co2 = a_co2[self.order_co2_steps].T
        a_light = a_light[self.order_light_steps].T
    
        return a_co2.ravel(), a_light.ravel()
    
