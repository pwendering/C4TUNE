classdef var_env
   properties
       Q_t
       Ca_t
       MeasuredTemp
       WeatherTemperature
       WeatherRH
       CI
       Radiation_PAR
       WeatherWind
   end

   methods 
       function self = var_env(Q_t,Ca_t,air_temp,RH,WeatherWind) %,initCO2,PAR
            self.Ca_t=Ca_t;
            self.Q_t=Q_t;
            
            if isempty(Ca_t)
               self.Ca_t=400;
            end

            if isempty(Q_t)
                self.Q_t=1800;
            end
            
            self.WeatherWind = WeatherWind;
            if isempty(WeatherWind)
                self.WeatherWind = cte_env.WeatherWind;
            end
            
            self.CI=self.Ca_t*0.4/(3 * 10^4);%The initial intercelluar CO2 Concnetration mmol/L
            
            self.MeasuredTemp = air_temp;
            self.WeatherTemperature = air_temp;
            self.WeatherRH = RH;

            Convert=1E6/(2.35E5); 
            self.Radiation_PAR=self.Q_t/Convert;
       end

   end
end