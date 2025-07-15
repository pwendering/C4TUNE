classdef setting
    properties(Constant)
        pathway_option=0;
        %%% 0=Normol NADP-ME type 
        %%% 1=Asp+Mal transport and MDH type 
        %%% 2=Asp+Mal and PCK type 
        %%% 3 Asp+Mal and PCK+MDH type 
        %%% 4 Asp and PCK only type
        %%% 6 DiT2 mutant
        %%% 7 NAD-ME type
        %%% 8 NAD-ME+PCK type

        EAPPDK=1;%%if EAPPDK=0 PPDK is fully actived; %%if EAPPDK=1 include PPDK activation by PDRP
        PRac=1;%%if PRac=0 Rubisco is fully actived; %%if PRac=1 include Rubisco activation by Rca
        RedoxEnyAct=1; %%if RedoxEnyAct=0 activities of photosynthetic enzymes are not regulated by light intensity; %%if RedoxEnyAct=1 include light induced enzyme activation 
        GsResponse=1; %%if GsResponse=0 Ball berry model no time dependent Gs response ; %%if GsResponse=1 include Gs response, using ki and kd

        kdcon=1;%=1: constant kd; =0: kd change with light
        Para_mata=1;%%if Para_mata=1, C4 Metabolic model and Gs model integrated  if Para_mata=0 Steady-state mdoel and gs model

        Ratio=4; % Enezyme activity variation factor

        RatioPPDK=0; %PPDK in BSC
        Pvpr8M=0;%if 0 Glycerate kinase in BSchl; if 1 Glycerate kinase in Mchl;
        
        option2=1; %O2 diffusion

    end
end