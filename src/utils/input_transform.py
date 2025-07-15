import torch


def prepare_model_inputs(a_co2, co2_steps, light_a_co2, a_light,
                       light_steps, co2_a_light, b_size):
    '''
    Configures inputs for LSTM module by adding a third dimension

    Parameters
    ----------
    a_co2 : Tensor
        Anet values at the different CO2 steps.
    co2_steps : Tensor
        CO2 steps.
    light_a_co2 : Tensor
        constant light for A/CO2 curves.
    a_light : Tensor
        Anet values at the different light steps.
    light_steps : Tensor
        light steps.
    co2_a_light : Tensor
        constant CO2 for A/light curves.
    b_size : Tensor
        batch size.

    Returns
    -------
    a_co2 : Tensor
        Anet values at the different CO2 steps.
    co2_steps : Tensor
        CO2 steps.
    light_a_co2 : Tensor
        constant light for A/CO2 curves.
    a_light : Tensor
        Anet values at the different light steps.
    light_steps : Tensor
        light steps.
    co2_a_light : Tensor
        constant CO2 for A/light curves.

    '''
    
    n_co2_steps = len(co2_steps)
    n_light_steps = len(light_steps)
    
    # add a third dimension and repeat the values across the second dimension
    if a_co2 is not None:
        a_co2 = a_co2.unsqueeze(1).repeat(1, n_co2_steps, 1)
    
    # add a third dimension to CO2 steps and repeat CO2 steps across the
    # third dimension
    co2_steps = co2_steps.unsqueeze(1).repeat(b_size, 1, n_co2_steps)
        
    # repeat CO2 curve constant light value across first dimension to 
    # adjust for batch size; add third dimension and repeat the values
    # over the second dimension
    light_a_co2 = light_a_co2.repeat(b_size, 1).unsqueeze(1).repeat(1, n_co2_steps, 1)    
    
    
    # add a third dimension and repeat the values across the second dimension
    if a_light is not None:
        a_light = a_light.unsqueeze(1).repeat(1, n_light_steps, 1)
    
    # add a third dimension to light steps and repeat light steps across
    # the third dimension
    light_steps = light_steps.unsqueeze(1).repeat(b_size, 1, n_light_steps)
    
    # repeat light curve constant CO2 value across first dimension to 
    # adjust for batch size; add third dimension and repeat the values
    # over the second dimension
    co2_a_light = co2_a_light.repeat(b_size, 1).unsqueeze(1).repeat(1, n_light_steps, 1)
    
    return a_co2, co2_steps, light_a_co2, a_light, light_steps, co2_a_light