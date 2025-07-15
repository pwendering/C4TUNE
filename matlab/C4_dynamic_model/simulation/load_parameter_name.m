function [param_names,enzymes, reactions]=load_parameter_name()
reactions=cell(57,1);
enzymes=cell(57,1);
vmax=cell(53,1);

reactions{1}='1. CO2->HCO3 [MC]';
enzymes{1}='4.2.1.1 Carbonic anhydrase (CA)';

reactions{2}='2. HCO3->PEP+OAA [MC]';
enzymes{2}='4.1.1.31 Phosphoenolpyruvate carboxylase (PEPC)';

reactions{3}='3. OAA+NADPH->MAL+NADP [MCchl]';
enzymes{3}='1.1.1.82 Malate dehydrogenase (MDH)';

reactions{4}='4. MAL+NADP->CO2+PYR+NADPH [BCchl]';
enzymes{4}='1.1.1.40 NADP-Malic enzyme (ME)';

reactions{5}='5. PYR+ATP->PEP [MCchl]';
enzymes{5}='2.7.9.1 Pyruvate, phosphate dikinase (PPDK)';

reactions{6}='6. CO2+RuBP->2PGA [BCchl]';
enzymes{6}='4.1.1.39 Rubisco';

reactions{7}='7. PGA+ATP->ADP+DPGA [BCchl]';
enzymes{7}='2.7.2.3 (PGAK)';

reactions{8}='8. DPGA+NADPH->GAP+Pi+NADP [BCchl]';
enzymes{8}='1.2.1.13 ()';

reactions{9}='9. GAP<->DHAP [BCchl]';
enzymes{9}='5.3.1.1 Triose-phosphate isomerase';

reactions{10}='10. DHAP+GAP<->FBP [BCchl]';
enzymes{10}='4.1.2.13 Fructose-bisphosphate aldolase';

reactions{11}='11. FBP<->F6P+Pi [BCchl]';
enzymes{11}='3.1.3.11 Fructose-bisphosphatase';

reactions{12}='12. E4P+DHAP<->SBP [BCchl]';
enzymes{12}='4.1.2.13 Fructose-bisphosphate aldolase';

reactions{13}='13. SBP<->S7P+Pi [BCchl]';
enzymes{13}='3.1.3.37 Sedoheptulose-bisphosphatase';

reactions{14}='14. F6P+GAP<->E4P+Xu5P [BCchl]';
enzymes{14}='2.2.1.1 Transketolase';

reactions{15}='15. S7P+GAP<->Ri5P+Xu5P [BCchl]';
enzymes{15}='2.2.1.1 Transketolase';

reactions{16}='16. Ri5P<->Ru5P [BCchl]';
enzymes{16}='5.3.1.6 Ribose-5-phosphate isomerase';

reactions{17}='17. Xu5P<->Ru5P [BCchl]';
enzymes{17}='5.1.3.1 Ribulose-phosphate 3-epimerase';

reactions{18}='18. Ru5P+ATP<->RuBP+ADP [BCchl]';
enzymes{18}='2.7.1.19 Phosphoribulokinase';

reactions{19}='19. PGA+ATP->ADP+DPGA [MCchl]';
enzymes{19}='2.7.2.3 ()';

reactions{20}='20. DPGA+NADPH->GAP+Pi+NADP [MCchl]';
enzymes{20}='1.2.1.13 ()';

reactions{21}='21. F6P<->G6P, G6P<->G1P [BC]'; % Sta1 and Sta2
enzymes{21}='5.3.1.9 Phosphohexomutase, 5.4.2.2 Glucose phosphomutase';

reactions{22}='22. PGA->sink ';
enzymes{22}='PGA sink ()';

reactions{23}='23. DHAP+GAP<->FBP [MC]'; % Suc1
enzymes{23}='4.1.2.13 Fructose-bisphosphate aldolase';

reactions{24}='24. FBP<->F6P+Pi [MC]'; % Suc2
enzymes{24}='3.1.3.11 Fructose-bisphosphatase';

reactions{25}='25. F6P<->G6P, G6P<->G1P [MC]'; % Suc 5 and Suc6
enzymes{25}='5.3.1.9 Phosphohexomutase, 5.4.2.2 Glucose phosphomutase';

reactions{26}='26. G1P+UTP<->UDPG+PPi [MC]'; %Suc7
enzymes{26}='2.7.7.9 UTP---glucose-1-phosphate uridylyltransferase';

reactions{27}='27. UDPG+F6P<->SUCP+UDP [MC]'; %Suc8
enzymes{27}='2.4.1.14 Sucrose-phosphate synthase';

reactions{28}='28. SUCP<->Pi+SUC [MC]'; % Suc9
enzymes{28}='3.1.3.24 Sucrose-phosphate phosphatase';

reactions{29}='29. SUC->sink [MC]';
enzymes{29}='SUC sink ()';

reactions{30}='30. F6P+ATP->F26BP+ADP [MC]'; % Suc3
enzymes{30}='2.7.1.105 6-phosphofructo-2-kinase';

reactions{31}='31. F26BP[c]->F6P[c]+Pi[c] [MC]'; % Suc4
enzymes{31}='3.1.3.46 Fructose-2,6-bisphosphate 2-phosphatase';

reactions{32}='32. ADP+Pi<->ATP [MCthy]'; % light reactions
enzymes{32}='3.6.3.14 ()';

reactions{33}='33. NADP<->NADPH [MCthy]'; % light reactions
enzymes{33}='1.18.1.2 Ferredoxin---NADP+ reductase';

reactions{34}='34. ADP+Pi<->ATP [BCthy]'; % light reactions
enzymes{34}='3.6.3.14 ATP synthase';

reactions{35}='35. Metabolite transport through plasmodesmata';
enzymes{35}='Metabolite transport through plasmodesmata';

reactions{36}='36. Pi equilibrium';% 
enzymes{36}='KEPi ()';

reactions{37}='37. NADP<->NADPH [BCthy]'; % light reactions
enzymes{37}='1.18.1.2 ()';


% Photorespiration
reactions{38}='38. O2+RuBP<->PGCA+PGA [BCchl]';
enzymes{38}='4.1.1.39';

reactions{39}='39. PGCA->Pi+GCA [BCchl]';
enzymes{39}='3.1.3.18';

reactions{40}='40. GCA+O2<->H2O2+GOA [BCper]'; % GLX==GOA
enzymes{40}='1.1.3.15';

reactions{41}='41. GOA+GLU<->GLY+KG [BCmit]';
enzymes{41}='2.6.1.4 Glycine transaminase';

reactions{42}='42. GLY+NAD<->SER+NADH+NH3 [BCmit]';
enzymes{42}='Gly_ser';

reactions{43}='43. SER+GOA<->HPR+GLY [BCper]';
enzymes{43}='2.6.1.45 Serine---glyoxylate transaminase';

reactions{44}='44. HPR+NAD<->GCEA+NADH [BCmit]';
enzymes{44}='1.1.1.29 Hydroxypyruvate reductase';

reactions{45}='45. GCEA+ATP<->PGA+ADP [BCchl]';
enzymes{45}='2.7.1.31';

reactions{46}='46. GCA[BCmit]<->GCA [BCchl]';
enzymes{46}='GCA transport';

reactions{47}='47. GCEA[BCmit]<->GCEA [BCchl]';
enzymes{47}='GCEA transport';

reactions{48}='48. PGA<->PEP [MC]';
enzymes{48}='5.4.2.12 Phosphoglycerate mutase, 4.2.1.11 2-phosphoglycerate enolase';

reactions{49}='49. PPDK inactivation';
enzymes{49}='';

reactions{50}='50. PPDK activation';
enzymes{50}='';

reactions{51}='51. G1P + ATP <-> PPi + ADPG';
enzymes{51}='2.7.7.27 (ADPG pyrophosphorylase)';

reactions{52}='52. PPi + H2O <-> 2 Pi [Bchl]';
enzymes{52}='3.6.1.1 (inorganic diphosphatase)';

reactions{53}='53. ADPG <-> Starch + ADP';
enzymes{53}='2.4.1.21 (starch synthase)';

reactions{54}='54. Hexose phosphate rate';
enzymes{54}='';

reactions{55}='55. PGA, GAP and DHAP transport [BC]';
enzymes{55}='TPT';

reactions{56}='56. PGA, GAP and DHAP transport [MC]';
enzymes{56}='TPT';

reactions{57}='57. Gs dynamics [MC]';
enzymes{57}='';

%% VMAX
vmax{1}='Vm1';
vmax{2}='Vm2';
vmax{3}='Vm3';
vmax{4}='Vm4';
vmax{5}='Vm5';
vmax{6}='Vm6';
vmax{7}='Vm7_8';
% vmax{8}='Vm8';
vmax{8}='Vm10';
vmax{9}='Vm11';
vmax{10}='Vm12';
vmax{11}='Vm13';
vmax{12}='Vm14';
vmax{13}='Vm15';
vmax{14}='Vm18';
vmax{15}='Vm19_20'; 
% vmax{17}='Vm20';
% vmax{18}='Vm21'; starch synthesis in Mchl
vmax{16}='Vm22';
vmax{17}='Vm23';
vmax{18}='Vm24';
vmax{19}='Vm26';
vmax{20}='Vm27';
vmax{21}='Vm28';
vmax{22}='Vm29';
vmax{23}='Vm30';
vmax{24}='Vm31';
vmax{25}='Jmax';
% ATP and NADPH synthesis in BC and MC
vmax{26}='Vm32';
vmax{27}='Vm33';
vmax{28}='Vm34';
vmax{29}='Vm37';
% photorespiration
vmax{30}='Vm38';
vmax{31}='Vm39';
vmax{32}='Vm40';
vmax{33}='Vm41';
vmax{34}='Vm42';
vmax{35}='Vm43';
vmax{36}='Vm44';
vmax{37}='Vm45';
vmax{38}='Vm46';
vmax{39}='Vm47';
vmax{40}='Vm48';

vmax{41}='Vm55';  % 'Vtp_Bchl'
vmax{42}='Vm56';  % 'Vtp_Mchl'
vmax{43}='Vm51';  % 'Vm_Sta1'
vmax{44}='Vm52';  % 'Vm_Sta2'
vmax{45}='Vm53';  % 'Vm_Sta3'
vmax{46}='Vm35Mc_OAA';
vmax{47}='Vm35B_pyr';
vmax{48}='Vm35M_pyr';
vmax{49}='Vm35M_PEP';
vmax{50}='Vm35B_PEP';
vmax{51}='Vm35B_mal';
vmax{52}='Vm35C_mal';
vmax{53}='Vm35_Hep';

%% KM

km=cell(57,8);
km{1,1}='Km1_CO2';  % km{1,2}='Ke1';
km{2,1}='Km2_HCO3';  km{2,2}='Km2_PEP';   km{2,3}='Ki2_MAL'; km{2,4}='Ki2_MALn';
km{3,1}='Km3_NADPH';  km{3,2}='Km3_OAA';  km{3,3}='Km3_NADP';  km{3,4}='Km3_MAL';  % km{3,5}='Ke3';
km{4,1}='Km4_CO2';  km{4,2}='Km4_NADP';  km{4,3}='Km4_NADPH';  km{4,4}='Km4_Pyr';  km{4,5}='Km4_MAL';  % km{4,6}='Ke4';
km{5,1}='Ki5_PEP';  km{5,2}='Km5_ATP';  km{5,3}='Km5_Pyr';
km{6,1}='Km6_CO2';  km{6,2}='Km6_O2';  km{6,3}='Km6_RuBP';  km{6,4}='Ki6_PGA';  km{6,5}='Ki6_FBP';  km{6,6}='Ki6_SBP';  km{6,7}='Ki6_Pi';  km{6,8}='Ki6_NADPH';

% km{7,1}='KmATP';  km{7,2}='KmPGA'; 
% km{8,1}='KmNADPH';
km{7,1}='Km7_ATP';  km{7,2}='Km7_PGA'; 
km{8,1}='Km8_NADPH';

% km{9,1}='Ke9'; 
km{10,1}='Km10_DHAP';  km{10,2}='Km10_FBP';  km{10,3}='Km10_GAP';  % km{10,4}='Ke10';
km{11,1}='Ki11_F6P';  km{11,2}='Ki11_Pi';  km{11,3}='Km11_FBP';  % km{11,4}='Ke11';
km{12,1}='Km12_DHAP';  km{12,2}='Km12_E4P';  % km{12,3}='Ke12';
km{13,1}='Ki13_Pi';  km{13,2}='Km13_SBP';  % km{13,3}='Ke13';
km{14,1}='Km14_E4P';  km{14,2}='Km14_F6P';  km{14,3}='Km14_GAP';  km{14,4}='Km14_Xu5P';  % km{14,5}='Ke14';
km{15,1}='Km15_GAP';  km{15,2}='Km15_Ri5P';  km{15,3}='Km15_S7P';  km{15,4}='Km15_Xu5P';  % km{15,5}='Ke15';
% km{16,1}='Ke16';
% km{17,1}='Ke17';
km{18,1}='Ki18_ADP';  km{18,2}='Ki18_ADP2';  km{18,3}='Ki18_PGA';  km{18,4}='Ki18_Pi';  km{18,5}='Ki18_RuBP';  km{18,6}='Km18_ATP';  km{18,7}='Km18_Ru5P';  % km{18,8}='Ke18';

% km{19,1}='KmADP';  km{19,2}='KmATP';  km{19,3}='KmPGA';
% km{20,1}='KmDPGA';  km{20,1}='KmNADPH';

km{19,1}='Km19_ATP';  km{19,2}='Km19_PGA';
km{20,1}='Km20_NADPH';

% km{21,1}='KiADP';  km{21,2}='KmATP';  km{21,3}='KmG1P';  km{21,4}='KaF6P';  km{21,5}='KaFBP';  km{21,6}='KaPGA';  km{21,7}='Ke';   km{21,8}='Ke';
% km{21,1}='Ke21_Starch1';   km{21,2}='Ke21_Starch2';

km{22,1}='Km22_PGA';
km{23,1}='Km23_DHAP';  km{23,2}='Km23_GAP';  km{23,3}='Km23_FBP';  % km{23,4}='Ke23';
km{24,1}='Ki24_F26BP';  km{24,2}='Ki24_F6P';  km{24,3}='Ki24_Pi';  km{24,4}='Km24_FBP';  % km{24,5}='Ke24';
% km{25,1}='Ke25_F6P';  km{25,2}='Ke25_G1P';
km{26,1}='Km26_G1P';  km{26,2}='Km26_PPi';  km{26,3}='Km26_UDPG';  km{26,4}='Km26_UTP';  % km{26,5}='Ke26';
km{27,1}='Ki27_FBP';  km{27,2}='Ki27_Pi';  km{27,3}='Ki27_Suc';  km{27,4}='Ki27_SucP';  km{27,5}='Ki27_UDP';  km{27,6}='Km27_F6P';  km{27,7}='Km27_UDPG';  % km{27,8}='Ke27'; 
km{28,1}='Km28_Suc';  km{28,2}='Km28_SucP';  % km{28,3}='Ke28';
km{29,1}='Km29_Suc';
km{30,1}='Ki30_ADP';  km{30,2}='Ki30_DHAP';  km{30,3}='Km30_ATP';  km{30,4}='Km30_F26BP';  km{30,5}='Km30_F6P';  % km{30,6}='Ke30';
km{31,1}='Ki31_F6P';  km{31,2}='Ki31_Pi';  km{31,3}='Km31_F26BP';  
% km{36,1}='Ke36_Pi';

km{32,1}='Km32_ADP';  km{32,2}='Km32_ATP';  km{32,3}='Km32_Pi';  km{32,4}='X32';  km{32,5}='Y32';  km{32,6}='F32';  km{32,7}='Q32';  km{32,8}='D32'; % km{32,9}='Ke32';
km{33,1}='Km33_NADP';  km{33,2}='Km33_NADPH'; km{33,3}='E33'; % km{33,3}='Ke33';  
km{34,1}='Km34_ADP';  km{34,2}='Km34_Pi';   km{34,3}='Km34_ATP';  km{34,4}='G34'; % km{34,5}='Ke34'; 
km{37,1}='Km37_NADP';  km{37,2}='Km37_NADPH'; % km{37,3}='Ke37'; 

% km{35,1}='Voaa';  km{35,2}='VMAL';  km{35,3}='Vpyr';  km{35,4}='Vpep';  km{35,5}='Vt';  km{35,6}='Vleak'; km{35,7}='Vpga';
km{35,1}='Km35Mc_OAA';  km{35,2}='Ki35Mc_mal_OAA';  km{35,3}='Km35M_mal';  km{35,4}='Ki35M_OAA_mal'; km{35,5}='Km35B_mal'; km{35,6}='Km35B_pyr';  km{35,7}='Km35M_pyr'; km{35,8}='Km35M_PEP';

% km{38,1}='Km38_CO2'; km{38,2}='Km38_O2';  km{38,3}='Km38_RuBP';  km{38,4}='Ki38_PGA';  km{38,5}='Ki38_FBP';  km{38,6}='Ki38_SBP';  km{38,7}='Ki38_Pi';  km{38,8}='Ki38_NADPH';
km{39,1}='Km39_PGCA';  km{39,2}='Ki39_PI';  km{39,3}='Ki39_GCA';
km{40,1}='KmGCA40';
km{41,1}='Km41_GOA';  km{41,2}='Km41_GLU';  km{41,3}='Ki41_GLY'; % km{41,4}='Ke41';  
km{42,1}='Km42_GLY';  km{42,2}='Ki42_SER';
km{43,1}='Km43_GOA';  km{43,2}='Km43_SER';  km{43,3}='Km43_GLY'; % km{43,4}='Ke43';  
km{44,1}='Ki44_HPR';  km{44,2}='Km44_HPR'; % km{44,3}='Ke44';  
km{45,1}='Km45_ATP';  km{45,2}='Km45_GCEA';  km{45,3}='Ki45_PGA'; % km{45,4}='Ke45';  
km{46,1}='Km46_GCA';  km{46,2}='Ki46_GCEA';
km{47,1}='Km47_GCEA';  km{47,2}='Ki47_GCA';
km{48,1}='Km48_PGA'; km{48,2}='Km48_PEP'; % km{48,3}='Ke48';

% new:
km{49,1}='Kcat49_EA_PPDKRP_I'; km{49,2}='Km49_EA_PPDKRP_I_ADP'; km{49,3}='Ki49_EA_PPDKRP_I_Pyr'; km{49,4}='Km49_EA_PPDKRP_I_E';
km{50,1}='Kcat50_EA_PPDKRP_A'; km{50,2}='Km50_EA_PPDKRP_A_Pi'; km{50,3}='Ki50_EA_PPDKRP_A_AMP'; km{50,4}='Km50_EA_PPDKRP_A_EP'; km{50,5}='Ki50_EA_PPDKRP_A_ADP'; km{50,6}='Ki50_EA_PPDKRP_A_PPI';

km{51,1}='Ka51_PGA'; km{51,2}='Km51_G1P'; km{51,3}='Km51_ATP'; km{51,4}='Ki51_APi_ATP'; km{51,5}='Km51_PPi'; km{51,6}='Ki51_CPP1_ATP'; km{51,7}='Km51_ADPG'; km{51,8}='Ki51_AADP_ATP'; % km{51,9}='Ke51'; 
km{52,1}='Km52_PPi'; % km{52,2}='Ke52';
km{53,1}='Km53_ADPG'; 
km{54,1}='Km54_pi'; km{54,2}='Km54_hexp';

km{55,1}='Km55_PGA'; km{55,2}='Km55_GAP'; km{55,3}='Km55_DHAP'; 
km{56,1}='Km56_PGA'; km{56,2}='Km56_GAP'; km{56,3}='Km56_DHAP';
km{57,1}='Ki57'; km{57,2}='Kd57';

%%
[~, KVnz] = load_initial_solution;
km_t = km';
final_km = km_t(KVnz');

%%
act_rate=cell(10,1);
act_rate{1}='tao_ActPEPC';
act_rate{2}='tao_ActFBPase';
act_rate{3}='tao_ActSBPase';
act_rate{4}='tao_ActATPsynthase';
act_rate{5}='tao_ActGAPDH';
act_rate{6}='tao_ActPRK';
act_rate{7}='tao_ActNADPMDH';
act_rate{8}='KaRac';
act_rate{9}='tao_ActRubisco';
act_rate{10}='tao_ActRca';


%%
perm=cell(6,1);
perm{1}='Perm_MAL';
perm{2}='Perm_PYR';
perm{3}='Perm_CO2';
perm{4}='Perm_PGA';
perm{5}='PermBC_CO2';
perm{6}='gm';


% params=["BBslope","BBintercept","factorvp","factorvc","[PDRP]","MRd"];
params=["BBslope","BBintercept","MRd"];

% tempvmax=vmax(setdiff(1:length(vmax),[2,6]));
param_names=[final_km;vmax;act_rate;perm;params'];