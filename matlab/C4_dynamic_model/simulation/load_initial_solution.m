function [initial_solution, KVnz]=load_initial_solution()

gm=3;
vpmax=124.1280;
vcmax=49.919;
factorvp=0.7200;
factorvc=0.6700;
ki=0.127;
kd=0.123;
%%
%Km Ki(mM)
%4.2.1.1        1
KmCO2_1 = 2.8;%2.8;  
% Ke_1 =11.2;%20;%;% No unit  k=4.3*10^(-7)  PH=7.5 13.6 11.82
%4.1.1.31       2 %KmHCO3_2 =0.05 KmPEP_2 =0.047
% 
KmPEP_2 =0.1; KmHCO3_2 =0.02;  Kimal_2 =15;  Kimal_2n=1.5;
%%%%%WY202003 Maize KmHCO3_2 =0.04216; KmPEP_2 =1;   Kimal_2 =15; 
%KmPEP_2 =0.038; 
%1.1.1.82       3
KmNADPH_3 =0.024;  KmOAA_3 =0.056;  KmNADP_3 =0.073;  Kmmal_3 =32.0;  
% Ke_3 =4450.0; % No unit
%1.1.1.40       4 Kmmal_4 =0.04;
KmCO2_4 =1.1;  KmNADP_4 =0.0080;  KmNADPH_4 =0.045;  KmPyr_4 =3;  Kmmal_4 =0.23;  % Ke_4 =0.051*55.35*1000;%0.0344 KmNADP_4 =0.008
%2.7.9.1        5 KiPEP_5 =0.5;
KiPEP_5 =0.15;  KmATP_5 =0.082;  KmPyr_5 =0.082;% KmPyr_5 =0.25
%4.1.1.39       6  KiPGA_6 =0.84; KiFBP_6 =0.4;
KmCO2_6 =0.0162;  KmO2_6 =0.183;  KmRuBP_6 =0.02;  KiPGA_6 =2.52;  KiFBP_6 =0.04;  KiSBP_6 =0.075;  KiPi_6 =0.9*3;  KiNADPH_6 =0.07*3;%KiNADPH_6 =0.07;KiPi_6 =0.9

%2.7.2.3        7
% KmADP_7 =0.5;  KmATP_7 =0.3;  KmPGA_7 =2.4;%1.2;%KmPGA_7 =0.24
%1.2.1.13       8
% KmDPGA_8 =0.4;  KmNADPH_8=0.1;

% Changed to 
KmPGA_78=5; KmATP_78=0.3; KmNADPH_78=0.1;

%5.3.1.1        9
% Ke_9=0.05;    
%4.1.2.13FBP    10
KmDHAP_10 =0.45;  KmFBP_10 =0.0923;  KmGAP_10 =0.04;  %KmGAP_10 =0.3; 
% Ke_10 =7.1; % 1/millimolarity
%3.1.3.11       11 KmFBP_11 =0.033; Chloroplast fructose-1,6-bisphosphatase
%from Oryza differs in salt tolerance property from the Porteresia enzyme
%and is protected by osmolytes  Ghosh, S.; Bagchi, S.; Lahiri Majumder, A.;Plant Sci. 160, 1171-1181 (2001)
KiF6P_11 =0.7;  KiPi_11 =12.0;  KmFBP_11 =0.066;  % Ke_11 =666000.0;
%4.1.2.13SBP    12
KmDHAP_12 =0.4;  KmE4P_12 =0.2;  
% Ke_12 =1.017;  % 1/millimolarity
%3.1.3.37       13
KiPi_13 =12.0;  KmSBP_13 =0.05;  % Ke_13 =666000.0;
%2.2.1.1X       14 Datta, A.G.; Racker, E.; J. Biol. Chem.; 236, 617 (1961). 
KmE4P_14 =0.1;  KmF6P_14 =0.1;  KmGAP_14 =0.1;  KmXu5P =0.1;  
% Ke_14 =0.084;  % No unit Ke_14 =10.0£»Ke_14 =0.076
%2.2.1.1R       15
KmGAP_15 =0.072;  KmRi5P_15 =1.5;  KmS7P_15 =0.015;  KmXu5P_15 =0.1;  % KmS7P_15 =0.015  KmS7P_15 =0.46
% Ke_15 =0.9;%1.176470588235294;  % No unit Datta, A.G.; Racker, E.; J. Biol. Chem.; 236, 617 (1961). 
%5.3.1.6:Chl    16
% Ke_16=0.4;
%5.1.3.1:Chl    17
% Ke_17=0.67;
%2.7.1.19       18 % KiPGA_18 =1.0;  KiPi_18 =1.0; KmATP_18 =0.059;KmATP_18=0.625
KiADP_18 =2.5;  Ki_ADP_18 =0.4;  KiPGA_18 =2.0;  KiPi_18 =4.0;  KiRuBP_18 =0.7;  KmATP_18 =0.625;  KmRu5P_18 =0.05; % KiPi_18 =4.0; 
% Ke_18 =6846.0; % No unit

% %2.7.2.3:MChl  7Mchl
% KmADP_7Mchl =0.5;  KmATP_7Mchl =0.3;  KmPGA_7Mchl =2.4;
% %1.2.1.13:MChl   8Mchl
% KmDPGA_8Mchl =0.4;  KmNADPH_8Mchl =0.1;  
KmPGA_78Mchl=5; KmATP_78Mchl=0.3; KmNADPH_78Mchl=0.1;
%StarchSynthesis:Chl   Starchsyn  KmG1P_Starch =0.24;  KaF6P_Starch =0.06;
%KaFBP_Starch =0.06;
% KiADP_Starch =10.0;  KmATP_Starch =0.08;  KmG1P_Starch =0.48;  KaF6P_Starch =0.12;  KaFBP_Starch =0.12; KaPGA_Starch =0.3;% Ka (No unit)KmG1P_Starch =0.08;  KaF6P_Starch =0.02;  KaFBP_Starch =0.02; KaPGA_Starch =0.1
% Ke_Starch1=2.3;   Ke_Starch2=0.058;
%PGASink
KmPGA_PGASink=1;%0.5;

%4.1.2.13FBP:Cel    Suc1
KmDHAP_Suc1 =0.45;  KmGAP_Suc1 =0.04/2;  KmFBP_Suc1 =0.0023;   %KmDHAP_Suc1 =0.4;  KmGAP_Suc1 =0.1;  KmFBP_Suc1 =0.2;
% Ke_Suc1 =12;%12.0; 10000 % 1/millimolarity
%3.1.3.11:Cel    Suc2 KiF26BP_Suc2 =7.0E-5;KmFBP_Suc2 =0.165;Ke_Suc2=6663.0
%Roles of the residues Lys115 and Tyr116 in the binding of an allosteric inhibitor AMP to pea cytosolic D-fructose-1,6-bisphosphatase
%Jang, H.; Cho, M.; Kwon, Y.; Bhoo, S.H.; Jeon, J.; Hahn, T.; J. Appl. Biol. Chem. 51, 45-49 (2008)
KiF26BP_Suc2 =0.00007;  KiF6P_Suc2 =0.7;  KiPi_Suc2 =12.0;  KmFBP_Suc2 =0.00108;  % Ke_Suc2 =174.0;
%5.3.1.9:Cel    Suc5
% Ke_Suc5=2.3;
%5.4.2.2:Cel    Suc6
% Ke_Suc6=0.0584;
%2.7.7.9:Cel    Suc7
KmG1P_Suc7 =0.14;  KmPPi_Suc7 =0.11;  KmUDPG_Suc7 =0.12;  KmUTP_Suc7 =0.1;  
% Ke_Suc7 =0.31;%0.31; % No unit
%2.4.1.14:Cel   Suc8
KiFBP_Suc8 =0.8;  KiPi_Suc8 =5.0;  KiSuc_Suc8 =50.0;  KiSucP_Suc8 =0.4;  KiUDP_Suc8 =0.7;  KmF6P_Suc8 =0.8;  KmUDPG_Suc8 =1.3;  
% Ke_Suc8 =10.0;  % No unit
%3.1.3.24:Cel     Suc9
KmSuc_Suc9 =80.0;  KmSucP_Suc9 =0.35;  % Ke_Suc9 =780.0;
%SUCSink:Cel   Suc10
KmSuc_Suc10 =1.5;
%2.7.1.105:Cel     Suc3
KiADP_Suc3 =0.16;  KIDHAP_Suc3 =0.7;  KmATP_Suc3 =1.32;  KmF26BP_Suc3 =0.021;  KmF6P_Suc3 =1.4;  
% Ke_Suc3 =590.0;% No unit
%3.1.3.46:Cel    Suc4
KiF6P_Suc4 =0.1;  KiPi_Suc4 =0.5*10;  KmF26BP_Suc4= 0.032; 
%3.6.1.1:Cel 
% KePi=128.4;
%3.6.3.14:MChl   ATPM
KmADP_ATPM =0.014;  KmATP_ATPM =0.11;  KmPi_ATPM =0.3;
% Ke_ATPM =5.734;        %1/millimolarity
% global Xd;%0.667
X =0.667;  Y =0.6;  F =0.7225;  Q =0.9;  D =1;% No unit
%1.18.1.2:MChl  NADPHM
KmNADP_NADPHM =0.05;  KmNADPH_NADPHM =0.058; % KmNADP_NADPHM =0.0072;  KmNADPH_NADPHM =0.0058;  
E =0.5; % No unit  % Ke_NADPHM =502; 
%V3.6.3.14:Chl       ATPB
KmADP_ATPB =0.014;  KmPi_ATPB =0.11;   KmATP_ATPB =0.3;
% Ke_ATPB =5.734; % 1/millimolarity
G =0.667; % No unit
%1.18.1.2:BChl  NADPHB
KmNADP_NADPHB =0.05;  KmNADPH_NADPHB =0.058;  
% Ke_NADPHB =502;  % No unit

%4.1.1.39 O2 1
%KmCO2_6 =0.01935;  KmO2_6 =0.222;  KmRuBP_6 =0.02;  KiPGA_6 =2.52; KiFBP_6 =0.8;  KiSBP_6 =0.75;  KiPi_6 =0.9;  KiNADPH_6 =0.07;
KmCO2_PR1=0.0162;  KmO2_PR1=0.183;  KmRuBP_PR1=0.02;  KiPGA_PR1=2.52;  KiFBP_PR1=0.04;  KiSBP_PR1=0.75;  KiPi_PR1=0.9*3;  KiNADPH_PR1=0.21;
%3.1.3.18 2
KmPGCA_PR2=0.026;  KiPI_PR2=2.55;  KiGCA_PR2=94.0;%KmPGCA_PR2=0.026; 0.57
%1.1.3.15 3
KmGCA_PR3= 0.1;%KmGCA_PR3= 0.1;0.02
%2.6.1.4 4
KmGOA_PS4=0.15;  KmGLU_PS4= 1.7;  KiGLY_PS4=2.0; % Ke_PS4= 607.0;
%GLY_SER:Mit 5
KmGLY_PS5= 6.0;  KiSER_PS5=4.0;
%2.6.1.45 6
KmGOA_PR6=0.15;  KmSER_PR6=2.7;  KmGLY_PR6=33.0; % Ke_PR6= 0.24;
%1.1.1.29 7
KiHPR_PR7= 12.0;  KmHPR_PR7=0.09; % Ke_PR7= 250000.0;
%2.7.1.31 8
KmATP_PR8= 0.21;  KmGCEA_PR8=0.25;  KiPGA_PR8=0.72;% Ke_PR8= 300.0; KmATP_PR8= 0.21;  KmGCEA_PR8=0.25;  KiPGA_PR8=0.36;
%Tgca 9
KmGCA_PR9= 0.2;  KiGCEA_PR9= 0.22;
%Tgcea 10
KmGCEA_PR10= 0.39;  KiGCA_PR10= 0.28;

% Transport coeffcient (1/second)
% Voaa =1.5;
% Vmal =1.5;
% Vpyr =1.5;
% Vpep =1.5;
% Vt =1.5;
% Vleak=1;
% Vpga=2;

KmPGA_62 = 0.08; 
KmPEP_62 = 0.3;
% Ke_62=0.4302;%G66 = +0.5; 



%%%%%%%%%%%%%%%%%%% new parameters %%%%%%%%%%%%%%%%%%%%

Km_OAA_M=0.053;  
Kimal_OAA_M=7.5;  
Km_MAL_M=0.5;
KiOAA_MAL_M=0.3;%0.3;
Km_MAL_B=1;
Km_PYR_B=0.1;
Km_PEP_M=0.5;
Km_PYR_M=0.1;


Kcat_EA_PPDKRP_I= 1.125; % s-1
Kcat_EA_PPDKRP_A= 0.578; % s-1
% inactivation
Km_EA_PPDKRP_I_ADP=0.052; %mM
Ki_EA_PPDKRP_I_Pyr=0.08;
Km_EA_PPDKRP_I_E=0.0012;
% activation
Km_EA_PPDKRP_A_Pi= 0.65;
Ki_EA_PPDKRP_A_AMP=0.4;
Km_EA_PPDKRP_A_EP=0.0007;
Ki_EA_PPDKRP_A_ADP=0.085;
Ki_EA_PPDKRP_A_PPI=0.16;


KaPGA_Sta1=0.2;%0.2252;
KmG1P_Sta1=0.06;%0.06;
KmATP_Sta1=0.12;%0.18
KIAPi_ATP_Sta1=2.96;
KmPPi_Sta1=0.033;
KICPP1_ATP_Sta1=13.8E-4;
KmADPG_Sta1=0.24;
KIAADP_ATP_Sta1=2.0;
% Ke_Sta1=1.1;

%  3.6.1.1
KmPPi_Sta2=0.154;
% Ke_Sta2=15700.0;

%  2.4.1.21
KmADPG_Sta3=0.077;

%
% Hexose phosphate, includes F6P, G6P, and G1P
Kmpi_hexp=1.5;
Kmhexp_hexp=1;%1/5;%WY20/02
%KmATP_hexp=0.8;
Vm_Hep=0.0005;

% TPT transport
KmPGA_B = 2; 
KmGAP_B =2; 
KmDHAP_B =2; 

KmPGA_M =2; 
KmGAP_M =2; 
KmDHAP_M = 2; 

KValue=zeros(48,8);
KValue(1,1)=KmCO2_1;  % KValue(1,2)=Ke_1;
KValue(2,1)=KmHCO3_2;  KValue(2,2)= KmPEP_2;   KValue(2,3)=Kimal_2; KValue(2,4)=Kimal_2n;
KValue(3,1)=KmNADPH_3;  KValue(3,2)=KmOAA_3;  KValue(3,3)=KmNADP_3;  KValue(3,4)=Kmmal_3;  % KValue(3,5)=Ke_3;
KValue(4,1)=KmCO2_4;  KValue(4,2)=KmNADP_4;  KValue(4,3)=KmNADPH_4;  KValue(4,4)=KmPyr_4;  KValue(4,5)=Kmmal_4;  % KValue(4,6)=Ke_4;
KValue(5,1)=KiPEP_5;  KValue(5,2)=KmATP_5;  KValue(5,3)=KmPyr_5;
KValue(6,1)=KmCO2_6;  KValue(6,2)=KmO2_6;  KValue(6,3)=KmRuBP_6;  KValue(6,4)=KiPGA_6;  KValue(6,5)=KiFBP_6;  KValue(6,6)=KiSBP_6;  KValue(6,7)=KiPi_6;  KValue(6,8)=KiNADPH_6;
% KValue(7,1)=KmADP_7;  KValue(7,2)=KmATP_7;  KValue(7,3)=KmPGA_7;
KValue(7,1)=KmATP_78;  KValue(7,2)=KmPGA_78;
% KValue(8,1)=KmDPGA_8;  KValue(8,2)=KmNADPH_8;
KValue(8,1)=KmNADPH_78;

% KValue(9,1)=Ke_9; 
KValue(10,1)=KmDHAP_10;  KValue(10,2)=KmFBP_10;  KValue(10,3)=KmGAP_10;  % KValue(10,4)=Ke_10;
KValue(11,1)=KiF6P_11;  KValue(11,2)=KiPi_11;  KValue(11,3)=KmFBP_11;  % KValue(11,4)=Ke_11;
KValue(12,1)=KmDHAP_12;  KValue(12,2)=KmE4P_12;  % KValue(12,3)=Ke_12;
KValue(13,1)=KiPi_13;  KValue(13,2)=KmSBP_13;  % KValue(13,3)=Ke_13;
KValue(14,1)=KmE4P_14;  KValue(14,2)=KmF6P_14;  KValue(14,3)=KmGAP_14;  KValue(14,4)=KmXu5P;  % KValue(14,5)=Ke_14;
KValue(15,1)=KmGAP_15;  KValue(15,2)=KmRi5P_15;  KValue(15,3)=KmS7P_15;  KValue(15,4)=KmXu5P_15;  % KValue(15,5)=Ke_15;
% KValue(16,1)=Ke_16;
% KValue(17,1)=Ke_17;
KValue(18,1)=KiADP_18;  KValue(18,2)=Ki_ADP_18;  KValue(18,3)=KiPGA_18;  KValue(18,4)=KiPi_18;  KValue(18,5)=KiRuBP_18;  KValue(18,6)=KmATP_18;  KValue(18,7)=KmRu5P_18;  % KValue(18,8)=Ke_18;

% KValue(19,1)=KmADP_7Mchl;  KValue(19,2)=KmATP_7Mchl;  KValue(19,3)=KmPGA_7Mchl;
KValue(19,1)=KmATP_78Mchl;  KValue(19,2)=KmPGA_78Mchl;
% KValue(20,1)=KmDPGA_8Mchl;  KValue(20,1)=KmNADPH_8Mchl;
KValue(20,1)=KmNADPH_78Mchl;

% KValue(21,1)=KiADP_Starch;  KValue(21,2)=KmATP_Starch;  KValue(21,3)=KmG1P_Starch;  KValue(21,4)=KaF6P_Starch;  KValue(21,5)=KaFBP_Starch;  KValue(21,6)=KaPGA_Starch;  KValue(21,7)=Ke_Starch1;   KValue(21,8)=Ke_Starch2;
% KValue(21,1)=Ke_Starch1;   KValue(21,2)=Ke_Starch2;

KValue(22,1)=KmPGA_PGASink;
KValue(23,1)=KmDHAP_Suc1;  KValue(23,2)=KmGAP_Suc1;  KValue(23,3)=KmFBP_Suc1;  % KValue(23,4)=Ke_Suc1;
KValue(24,1)=KiF26BP_Suc2;  KValue(24,2)=KiF6P_Suc2;  KValue(24,3)=KiPi_Suc2;  KValue(24,4)=KmFBP_Suc2;  % KValue(24,5)=Ke_Suc2;
% KValue(25,1)=Ke_Suc5;  KValue(25,2)=Ke_Suc6;
KValue(26,1)=KmG1P_Suc7;  KValue(26,2)=KmPPi_Suc7;  KValue(26,3)=KmUDPG_Suc7;  KValue(26,4)=KmUTP_Suc7;  % KValue(26,5)=Ke_Suc7;
KValue(27,1)=KiFBP_Suc8;  KValue(27,2)=KiPi_Suc8;  KValue(27,3)=KiSuc_Suc8;  KValue(27,4)=KiSucP_Suc8;  KValue(27,5)=KiUDP_Suc8;  KValue(27,6)=KmF6P_Suc8;  KValue(27,7)=KmUDPG_Suc8;  % KValue(27,8)=Ke_Suc8; 
KValue(28,1)=KmSuc_Suc9;  KValue(28,2)=KmSucP_Suc9;  % KValue(28,3)=Ke_Suc9;
KValue(29,1)=KmSuc_Suc10;
KValue(30,1)=KiADP_Suc3;  KValue(30,2)=KIDHAP_Suc3;  KValue(30,3)=KmATP_Suc3;  KValue(30,4)=KmF26BP_Suc3;  KValue(30,5)=KmF6P_Suc3;  % KValue(30,6)=Ke_Suc3;
KValue(31,1)=KiF6P_Suc4;  KValue(31,2)=KiPi_Suc4;  KValue(31,3)=KmF26BP_Suc4;  
% KValue(36,1)=KePi;

KValue(32,1)=KmADP_ATPM;  KValue(32,2)=KmATP_ATPM;  KValue(32,3)=KmPi_ATPM;  KValue(32,4)=X;  KValue(32,5)=Y;  KValue(32,6)=F;  KValue(32,7)=Q;  KValue(32,8)=D; % KValue(32,9)=Ke_ATPM;
KValue(33,1)=KmNADP_NADPHM;  KValue(33,2)=KmNADPH_NADPHM;  KValue(33,3)=E;  % KValue(33,3)=Ke_NADPHM; 
KValue(34,1)=KmADP_ATPB;  KValue(34,2)=KmPi_ATPB;   KValue(34,3)=KmATP_ATPB;    KValue(34,4)=G; % KValue(34,4)=Ke_ATPB; 
KValue(37,1)=KmNADP_NADPHB;  KValue(37,2)=KmNADPH_NADPHB; % KValue(37,3)=Ke_NADPHB; 

% KValue(35,1)=Voaa;  KValue(35,2)=Vmal;  KValue(35,3)=Vpyr;  KValue(35,4)=Vpep;  KValue(35,5)=Vt;  KValue(35,6)=Vleak; KValue(35,7)=Vpga;
KValue(35,1)=Km_OAA_M;  KValue(35,2)=Kimal_OAA_M;  KValue(35,3)=Km_MAL_M;  KValue(35,4)=KiOAA_MAL_M; KValue(35,5)=Km_MAL_B; KValue(35,6)=Km_PYR_B;  KValue(35,7)=Km_PYR_M; KValue(35,8)=Km_PEP_M;


% KValue(38,1)= KmCO2_PR1; KValue(38,2)= KmO2_PR1;  KValue(38,3)=KmRuBP_PR1;  KValue(38,4)=KiPGA_PR1;  KValue(38,5)=KiFBP_PR1;  KValue(38,6)=KiSBP_PR1;  KValue(38,7)=KiPi_PR1;  KValue(38,8)=KiNADPH_PR1;
KValue(39,1)=KmPGCA_PR2;  KValue(39,2)=KiPI_PR2;  KValue(39,3)=KiGCA_PR2;
KValue(40,1)=KmGCA_PR3;
KValue(41,1)=KmGOA_PS4;  KValue(41,2)=KmGLU_PS4;  KValue(41,3)=KiGLY_PS4; % KValue(41,1)=Ke_PS4; 
KValue(42,1)=KmGLY_PS5;  KValue(42,2)=KiSER_PS5;
KValue(43,1)=KmGOA_PR6;  KValue(43,2)=KmSER_PR6;  KValue(43,3)=KmGLY_PR6; % KValue(43,1)=Ke_PR6; 
KValue(44,1)=KiHPR_PR7;  KValue(44,2)=KmHPR_PR7; % KValue(44,1)=Ke_PR7;  
KValue(45,1)=KmATP_PR8;  KValue(45,2)=KmGCEA_PR8;  KValue(45,3)=KiPGA_PR8; % KValue(45,1)=Ke_PR8;  
KValue(46,1)=KmGCA_PR9;  KValue(46,2)=KiGCEA_PR9;
KValue(47,1)=KmGCEA_PR10;  KValue(47,2)=KiGCA_PR10;
KValue(48,1)=KmPGA_62; KValue(48,2)=KmPEP_62; % KValue(48,3)=Ke_62;

KValue(49,1)=Kcat_EA_PPDKRP_I; KValue(49,2)=Km_EA_PPDKRP_I_ADP; KValue(49,3)=Ki_EA_PPDKRP_I_Pyr; KValue(49,4)=Km_EA_PPDKRP_I_E;
KValue(50,1)=Kcat_EA_PPDKRP_A; KValue(50,2)=Km_EA_PPDKRP_A_Pi; KValue(50,3)=Ki_EA_PPDKRP_A_AMP; KValue(50,4)=Km_EA_PPDKRP_A_EP; KValue(50,5)=Ki_EA_PPDKRP_A_ADP; KValue(50,6)=Ki_EA_PPDKRP_A_PPI;
KValue(51,1)=KaPGA_Sta1; KValue(51,2)=KmG1P_Sta1; KValue(51,3)=KmATP_Sta1; KValue(51,4)=KIAPi_ATP_Sta1; KValue(51,5)=KmPPi_Sta1; KValue(51,6)=KICPP1_ATP_Sta1; KValue(51,7)=KmADPG_Sta1; KValue(51,8)=KIAADP_ATP_Sta1; % KValue(51,9)=Ke_Sta1; 
KValue(52,1)=KmPPi_Sta2; % KValue(52,2)=Ke_Sta2;
KValue(53,1)=KmADPG_Sta3; 
KValue(54,1)=Kmpi_hexp; KValue(54,2)= Kmhexp_hexp;

KValue(55,1)=KmPGA_B; KValue(55,2)=KmGAP_B; KValue(55,3)=KmDHAP_B; 
KValue(56,1)=KmPGA_M; KValue(56,2)=KmGAP_M; KValue(56,3)=KmDHAP_M;
KValue(57,1)=ki; KValue(57,2)=kd;

KVnz = KValue~=0;
kvalues = nonzeros(KValue')';

%mM/(L*s) 

% global Vpmax;
%  global Vcmax;
% global MeasuredTemp;
MeasuredTemp=28;
% global FactorVP;
% global FactorVC;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Temp correction
Ea_Vpmax=94.8*1000;
dS_Vpmax=0.25*1000;
Hd_Vpmax=73.3*1000;
Ea_Vcmax=78*1000;
Ea_PPDK=58.1*1000;
R=8.3144598;% m2 kg s-2 K-1 mol-1
MTempCorr_V2=exp(Ea_Vpmax *((MeasuredTemp+273.15)-298.15)/(298.15*R*(MeasuredTemp+273.15)))*(1+exp((298.15*dS_Vpmax-Hd_Vpmax)/(298.15*R)))/(1+exp(((MeasuredTemp+273.15)*dS_Vpmax-Hd_Vpmax)/((MeasuredTemp+273.15)*R)));
MTempCorr_V6=exp(Ea_Vcmax*((MeasuredTemp+273.15)-298.15)/(298.15*R*(MeasuredTemp+273.15)));
MTempCorr_V5=exp(Ea_PPDK*((MeasuredTemp+273.15)-298.15)/(298.15*R*(MeasuredTemp+273.15)));
Vpmax_25=vpmax/MTempCorr_V2;
Vcmax_25=vcmax/MTempCorr_V6;

Vppdkmax_25=vcmax/MTempCorr_V5;
Vm_2 = Vpmax_25/1000/factorvp;%3.6;%4; 
Vm_1 = 200;
Vm_6o = 0.065;%2.4;%1.8;%2.9;%3.4967170968;2.913930914 2.09803025808
Vppdkmax_E=Vppdkmax_25/1000;
Vm_6 = Vcmax_25/1000/factorvc;

VPPDKsen=1;
VMDHsen=1;
VMEsen=1;
VGAPDHsen=1;
VSBPsen=1;
VFBPsen=1;
VPRKsen=1;
Jmaxsen=1;
% Jsen=1;
Vexsen=1;
% gsxsen=1;
% V6sen=1;

Vm_3 = 1.8*Vppdkmax_E*VMDHsen;%/2.4*Vm_2;%/1.2;        
Vm_4 = 1.33*Vppdkmax_E*VMEsen;%/2.4*Vm_2;%/1.5;%2.4;%3.2
Vm_5 = 1.33*Vppdkmax_E*VPPDKsen;%/2.4*Vm_2;%*1.5;%2.4; WY 201902
Vm_78= 0.4*Vm_6/Vm_6o*VGAPDHsen;%Vm_7 = 30.15;%16.07510272;% 30.15;%16.07510272
% Vm_8 = 0;%4.308781695;
Vm_10 = 0.0731*Vm_6/Vm_6o;%2.9253370968*1.5;%1.4626685484;
Vm_11 =0.0436*Vm_6/Vm_6o*VFBPsen;
Vm_12 = 0.1097*Vm_6/Vm_6o;
Vm_13 = 0.0292*Vm_6/Vm_6o*VSBPsen;
Vm_14 = 0.2810*Vm_6/Vm_6o;
Vm_15 = 0.2810*Vm_6/Vm_6o;
Vm_18 = 1.7552*Vm_6/Vm_6o*VPRKsen;


Vm_78Mchl=0.3*Vm_6/Vm_6o*VGAPDHsen;%Vm_7Mchl = 15.1;
% Vm_8Mchl =0;%2.6929885593*Vm_6/Vm_6o;
% Vm_Starch =0.0133*Vm_6/Vm_6o;
Vm_Sta1=0.03*Vm_6/Vm_6o;
Vm_Sta2=1*Vm_6/Vm_6o;
Vm_Sta3=0.025*Vm_6/Vm_6o;

Vm_PGASink = 0.002*Vm_6/Vm_6o;%0.5/5;
Vm_Suc1 = 0.0081*Vm_6/Vm_6o;
Vm_Suc2 = 0.0064*Vm_6/Vm_6o;
Vm_Suc7 = 0.0058*Vm_6/Vm_6o;
Vm_Suc8 = 0.0278*Vm_6/Vm_6o;
Vm_Suc9 = 0.0278*Vm_6/Vm_6o;
Vm_Suc10 =0.0035*Vm_6/Vm_6o; %2.0
Vm_Suc3 =0.0010*Vm_6/Vm_6o;
Vm_Suc4 =8.4096e-04*Vm_6/Vm_6o;

Jmax =0.5*Vm_6/Vm_6o*Jmaxsen;% 20;
Vm_ATPM = 0.3*Vm_6/Vm_6o;
Vm_NADPHM = 0.2*Vm_6/Vm_6o; 
Vm_ATPB = 0.3*Vm_6/Vm_6o;
Vm_NADPHB= 0.2*Vm_6/Vm_6o;

Vm_PR1= Vm_6*0.11;%0.69934341936;(Cousins 2010 0.11)
Vm_PR2= 2.6210*Vm_6/Vm_6o;
Vm_PR3= 0.0728*Vm_6/Vm_6o;
Vm_PR4= 0.1373*Vm_6/Vm_6o;
Vm_PR5= 0.1247*Vm_6/Vm_6o;
Vm_PR6= 0.1653*Vm_6/Vm_6o;
Vm_PR7= 0.5005*Vm_6/Vm_6o;
Vm_PR8= 0.2858*Vm_6/Vm_6o;
VTgca_PR9=0.3*Vm_6/Vm_6o;
VTgcea_PR10=0.25*Vm_6/Vm_6o;

Vm_62 =0.001*Vexsen;%mM/s %1.45 E-5; 

Vm_OAA_M=0.08*Vm_6/Vm_6o;
Vm_PYR_B=0.15*Vm_6/Vm_6o;
Vm_PYR_M=0.15*Vm_6/Vm_6o;
Vm_PEP_M=0.15*Vm_6/Vm_6o;
Vtp_Bchl=0.75;
Vtp_Mchl=0.75;



Vm_PEP_B=Vm_PEP_M;

Vm_MAL_B=Vm_PEP_M;
Vm_MAL_M=Vm_PEP_M;


% transport between two cell types
% global phi;
% global Lpd;
% phi=0.03;
% Lpd=400;

% Pmal=0.0421*(phi/0.03)/(Lpd/400);
% Ppyr=0.0436*(phi/0.03)/(Lpd/400);
% Pco2=0.1139*(phi/0.03)/(Lpd/400);
% PC3P=0.0327;
% Pc3p=PC3P*(phi/0.03)/(Lpd/400);
% Pco2_B=0.4;%%PCO2_B=0.002 cm s-1 SChl/Sl =10


Velocity_s=zeros(53,1); 
Velocity_s(1)=Vm_1;
Velocity_s(2)=Vm_2;     %vpmax; %          
Velocity_s(3)=Vm_3;
Velocity_s(4)=Vm_4;        
Velocity_s(5)=Vm_5;
Velocity_s(6)=Vm_6;
Velocity_s(7)=Vm_78;
% Velocity_s(8)=Vm_8;
Velocity_s(8)=Vm_10;
Velocity_s(9)=Vm_11;
Velocity_s(10)=Vm_12;
Velocity_s(11)=Vm_13;
Velocity_s(12)=Vm_14;
Velocity_s(13)=Vm_15;
Velocity_s(14)=Vm_18;
Velocity_s(15)=Vm_78Mchl;
% Velocity_s(17)=Vm_8Mchl;
% Velocity_s(18)=Vm_Starch;
Velocity_s(16)=Vm_PGASink;
Velocity_s(17)=Vm_Suc1;
Velocity_s(18)=Vm_Suc2;
Velocity_s(19)=Vm_Suc7;
Velocity_s(20)=Vm_Suc8;
Velocity_s(21)=Vm_Suc9;
Velocity_s(22)=Vm_Suc10;
Velocity_s(23)=Vm_Suc3;
Velocity_s(24)=Vm_Suc4;
% Velocity_s(28)=0;%Radiation_PAR*Convert/1000;
Velocity_s(25)=Jmax;
Velocity_s(26)=Vm_ATPM;
Velocity_s(27)=Vm_NADPHM; 
Velocity_s(28)=Vm_ATPB;
Velocity_s(29)=Vm_NADPHB;
Velocity_s(30)=Vm_PR1;
Velocity_s(31)=Vm_PR2;
Velocity_s(32)=Vm_PR3;
Velocity_s(33)=Vm_PR4;
Velocity_s(34)=Vm_PR5;
Velocity_s(35)=Vm_PR6;
Velocity_s(36)=Vm_PR7;
Velocity_s(37)=Vm_PR8;
Velocity_s(38)=VTgca_PR9;
Velocity_s(39)=VTgcea_PR10;
Velocity_s(40)=Vm_62;
Velocity_s(41)=Vtp_Bchl;
Velocity_s(42)=Vtp_Mchl;
Velocity_s(43)=Vm_Sta1;
Velocity_s(44)=Vm_Sta2;
Velocity_s(45)=Vm_Sta3;
Velocity_s(46)=Vm_OAA_M;
Velocity_s(47)=Vm_PYR_B;
Velocity_s(48)=Vm_PYR_M;
Velocity_s(49)=Vm_PEP_M;
Velocity_s(50)=Vm_PEP_B;
Velocity_s(51)=Vm_MAL_B;
Velocity_s(52)=Vm_MAL_M;
Velocity_s(53)=Vm_Hep; 

%%
% transport between two cell types
% global phi;
% global Lpd;
phi=0.03;
Lpd=400;

Pmal=0.0421*(phi/0.03)/(Lpd/400);
Ppyr=0.0436*(phi/0.03)/(Lpd/400);
Pco2=0.1139*(phi/0.03)/(Lpd/400);
PC3P=0.0327;
Pc3p=PC3P*(phi/0.03)/(Lpd/400);
Pco2_B=0.4;%%PCO2_B=0.002 cm s-1 SChl/Sl =10


perm=zeros(6,1);
perm(1)=Pmal;
perm(2)=Ppyr;
perm(3)=Pco2;
perm(4)=Pc3p;
perm(5)=Pco2_B;
perm(6)=gm;


%%
Tao_MDH=1;
Tao_PEPC=2;
taoRub=3.8800; % Initially from params (x11)

tao_ActPEPC =60*Tao_PEPC;
tao_ActFBPase =1.878*60;%1.878*60;
tao_ActSBPase =3.963*60;%3.963*60;
tao_ActATPsynthase=0.5*60;
tao_ActGAPDH=1*60/10;
tao_ActPRK=1*60/10;
tao_ActNADPMDH=0.965*60*Tao_MDH;%0.5*60*4*Tao_MDH;
KaRac=12.4;%mg m-2
tao_ActRubisco=taoRub*60;%2.5;%KTaoRac/Rac; % s
tao_ActRca =0.7594*60/10;

act_rate=zeros(10,1);
act_rate(1)=tao_ActPEPC;
act_rate(2)=tao_ActFBPase;
act_rate(3)=tao_ActSBPase;
act_rate(4)=tao_ActATPsynthase;
act_rate(5)=tao_ActGAPDH;
act_rate(6)=tao_ActPRK;
act_rate(7)=tao_ActNADPMDH;
act_rate(8)=KaRac;
act_rate(9)=tao_ActRubisco;
act_rate(10)=tao_ActRca;

%%
BBslope=5.2000;
BBintercept=0.0576;
% PDRP=0.0001;
MRd=2.2820;

other_params=[BBslope;BBintercept;MRd];

initial_solution=[kvalues';Velocity_s;act_rate;perm;other_params];