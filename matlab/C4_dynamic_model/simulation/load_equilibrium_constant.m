function [ke_value]=load_equilibrium_constant(ke_type)

% Keq values were obtained from eQuilibrator 3.0 (accessed on 22.08.2024)
% Settings: pH=7.2, pMg=3.0, I=0.25 M

% Rules
% * Keq of the reverse direction is Keq(reverse) = 1/Keq(forward)
% * Keq of the sum of multiple reactions is the product of their Keq values
%%
Ke_value=zeros(35,2);

% Ke[1]
% 4.2.1.1 Carbonic anhydrase (CA)
% CO2->HCO3
Ke_value(1,1) =11.2; % 5.6e-7 (SI)
Ke_value(1,2)= NaN;  % Cannot estimate ΔrG'° because the uncertainty is too high

% Ke[3]
% 1.1.1.82 Malate dehydrogenase (MDH)
% OAA+NADPH->MAL+NADP
Ke_value(2,1) =4450.0; 
Ke_value(2,2)=8.9e4;

% Ke[4]
% 1.1.1.40 NADP-Malic enzyme (ME)
% MAL+NADP->CO2+PYR+NADPH
Ke_value(3,1) =0.051*55.35*1000; %(SI: 0.051)
Ke_value(3,2)=7.4e-3;

% Ke[9]
% 5.3.1.1 Triose-phosphate isomerase
% GAP<->DHAP
Ke_value(4,1)=0.05;    
Ke_value(4,2)=10;  % updated, previous value: Keq=0.1    

% Ke[10]
% 4.1.2.13 Fructose-bisphosphate aldolase 
% DHAP+GAP<->FBP
Ke_value(5,1) =7.1; % 1/millimolarity
Ke_value(5,2)=9.4e3;  % updated, previous value: Keq=7e3;

% Ke[11]
% 3.1.3.11 Fructose-bisphosphatase 
% FBP<->F6P+Pi
Ke_value(6,1) =666000.0;
Ke_value(6,2)=1.5e2;

% Ke[12]
% 4.1.2.13 Fructose-bisphosphate aldolase
% E4P+DHAP<->SBP
Ke_value(7,1) =1.017;  % 1/millimolarity
Ke_value(7,2)=4.6e3;  % updated, previous value: Keq=2.2e-4

% Ke[13]
% 3.1.3.37 Sedoheptulose-bisphosphatase
% SBP<->S7P+Pi
Ke_value(8,1) =666000.0;
Ke_value(8,2)=2.1e2;

% Ke[14]
% 2.2.1.1 Transketolase
% F6P+GAP<->E4P+Xu5P
Ke_value(9,1) =0.084;  
Ke_value(9,2)=0.02;

% Ke[15]
% 2.2.1.1 Transketolase
% S7P+GAP<->Ri5P+Xu5P
Ke_value(10,1) =0.9;
Ke_value(10,2)=0.2;

% Ke[16]
% 5.3.1.6 Ribose-5-phosphate isomerase
% Ri5P<->Ru5P
Ke_value(11,1)=0.4;
Ke_value(11,2)=0.4;

% Ke[17]
% 5.1.3.1 Ribulose-phosphate 3-epimerase
% Xu5P<->Ru5P
Ke_value(12,1)=0.67;
Ke_value(12,2)=0.3;  % updated, previous value: Keq=4

% Ke[18]
% 2.7.1.19 Phosphoribulokinase  
% Ru5P+ATP<->RuBP+ADP
Ke_value(13,1)=6846.0; 
Ke_value(13,2)=5.3e4;

% Ke[21]-1
% 5.3.1.9 Phosphohexomutase
% F6P<->G6P [BC]
Ke_value(14,1)=2.3;   
Ke_value(14,2)=3;   

% Ke[21]-2
% 5.4.2.2 Glucose phosphomutase
% G6P<->G1P [BC]
Ke_value(15,1)=0.058;
Ke_value(15,2)=0.05;

% Ke[23]
% 4.1.2.13 Fructose-bisphosphate aldolase
% DHAP+GAP<->FBP
Ke_value(16,1) =12;
Ke_value(16,2)=9.4e3;  % updated, previous value: Keq=7e3;

% Ke[24]
% 3.1.3.11 Fructose-bisphosphatase
% FBP<->F6P+Pi
Ke_value(17,1) =174.0;
Ke_value(17,2)=1.5e2;

% Ke[25]-1
% 5.3.1.9 Phosphohexomutase
% F6P<->G6P
Ke_value(18,1)=2.3;
Ke_value(18,2)=3;   % updated, previous value: Keq=0.3

% Ke[25]-2
% 5.4.2.2 Glucose phosphomutase
% G6P<->G1P
Ke_value(19,1)=0.0584;
Ke_value(19,2)=0.05;   % updated, previous value: Keq=0.7

% Ke[26]
% 2.7.7.9 UTP---glucose-1-phosphate uridylyltransferase
% G1P+UTP<->UDPG+PPi
Ke_value(20,1) =0.31;
Ke_value(20,2)=2;

% Ke[27]
% 2.4.1.14 Sucrose-phosphate synthase
% UDPG+F6P<->SUCP+UDP
Ke_value(21,1) =10.0; 
Ke_value(21,2)=0.2;

% Ke[28]
% 3.1.3.24 Sucrose-phosphate phosphatase
% SUCP<->Pi+SUC
Ke_value(22,1) =780.0;
Ke_value(22,2)=1.3e3;

% Ke[30]
% 2.7.1.105 6-phosphofructo-2-kinase
% F6P+ATP->F26BP+ADP
Ke_value(23,1) =590.0;
Ke_value(23,2)=0.3;

% Ke[32]
% 3.6.3.14
% ADP+Pi<->ATP (Mesophyll)
% Warning: The ΔG' of ATP hydrolysis is highly affected by Mg2+ ions.
Ke_value(24,1) =5.734;        %1/millimolarity
Ke_value(24,2)=1.1e-5;  % updated, previous value: Keq=8.8e4

% Ke[33]
% 1.18.1.2 Ferredoxin---NADP+ reductase 
% NADP<->NADPH (M)
Ke_value(25,1)=502;
Ke_value(25,2)=4.3e2;  % updated, previous value: Keq=2.2e2

% Ke[34]
% 3.6.3.14
% ADP+Pi<->ATP (Bundle-sheath)
% Warning: The ΔG' of ATP hydrolysis is highly affected by Mg2+ ions.
Ke_value(26,1) =5.734;  
Ke_value(26,2)=Ke_value(24,2); 

% Ke[36]
% 3.6.1.1 (inorganic diphosphatase)
% H2O + Diphosphate <-> 2 Orthophosphate
Ke_value(27,1) =128.4; % 1/millimolarity
Ke_value(27,2)=8.5e2;

% Ke[37]
% 1.18.1.2 Ferredoxin---NADP+ reductase 
% NADP<->NADPH (Bundle-sheath chlorophyll)
Ke_value(28,1) =502;  % No unit
Ke_value(28,2)=Ke_value(25,2);

% Ke[41]
% 2.6.1.4 Glycine transaminase
% GOA+GLU<->GLY+KG
Ke_value(29,1)= 607.0; 
Ke_value(29,2)=30;  % updated, previous value: Keq=0.03

% Ke[43]
% 2.6.1.45 Serine---glyoxylate transaminase
% SER+GOA<->HPR+GLY
Ke_value(30,1)= 0.24; 
Ke_value(30,2)=6;

% Ke[44]
% 1.1.1.29 Hydroxypyruvate reductase
% HPR+NAD<->GCEA+NADH
Ke_value(31,1)= 250000.0; % 2.5e-5 (SI)
Ke_value(31,2)=1.8e5;  % updated, previous value: Keq=5.5e-6

% Ke[45]
% 2.7.1.31 Glycerate kinase
% GCEA+ATP<->PGA+ADP
Ke_value(32,1)= 300.0;  
Ke_value(32,2)=2.6e2;

% Ke[48]
% 5.4.2.1 Phosphoglycerate mutase, 4.2.1.11 2-phosphoglycerate enolase
% PGA<->PEP : 3PGA <-> 2PGA, 2PGA <-> PEP
Ke_value(33,1)=0.4302;
Ke_value(33,2)=0.2*5;

% Ke[51]
% 2.7.7.27 ADPG pyrophosphorylase
% G1P + ATP <-> PPi + ADPG
Ke_value(34,1)=1.1;
Ke_value(34,2)=30;

% Ke[52]
% 3.6.1.1 inorganic diphosphatase
% PPi + H2O <-> 2 Pi
Ke_value(35,1)=15700.0;
Ke_value(35,2)=8.5e2;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch ke_type
    case 'equilibrator'
        ke_value=Ke_value(:,1);
        final_ind=[2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 25 28 29 30 31 32 34 35];
        ke_value(final_ind)=Ke_value(final_ind,2);
    case 'original'
        ke_value=Ke_value(:,1);
end
