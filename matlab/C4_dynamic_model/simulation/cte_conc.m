classdef cte_conc
    properties(Constant)

        Bchl_CA= 1.5;
        Bchl_CN= 0.5;
        Bchl_CP= 25.0; %Pi content in BS chloroplast

        MC_CU=1.5;
        MC_CA=1.0;
        MC_CP=15.0; %Pi content in MC
        MC_UTP=1.26;%0.75

        Mchl_CA= 1.5;
        Mchl_CN= 0.5;
        Mchl_CP=15.0; %Pi content in MC chloroplast
        
        Bper_GLU=24;
        Bper_KG=0.4;
        Bper_NADH=0.47;
        Bper_NAD=0.4;

        U=0;% light partition coefficient
        V=0;% light partition coefficient

        PPDKRP=0.0001; %5.8000e-05;

        % Air_O2=210;
        % O2=Air_O2*1.26/1000;%O2 concentration 
        O2=210*1.26/1000; 

    end
end