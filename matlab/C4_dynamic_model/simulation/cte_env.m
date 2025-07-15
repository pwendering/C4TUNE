classdef cte_env
   properties(Constant)
        WeatherWind=3.5;%m/s %Set as 3.5 to match the bundary layer conductance from licor data
        PhiLeaf=0;%Mpa
        Radiation_NIR=0;
        Radiation_LW=0;
        ainter=[]; % not used when setting.kdcon==0
        VmaxC4=160; %For steady-state photosynthesis mdoel
   end
end