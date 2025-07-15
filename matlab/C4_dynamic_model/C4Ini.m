%the code for the initalizer EnzymeIni.m
%simulation of our enzyme system
 
function [con] = C4Ini(envFactor)%(Begin, GenNum)
CI=envFactor.CI;
O2=cte_conc.O2;

%MC

MC_HCO3= 0.3;%WY1911 0.005
MC_OAA=0.01;
MC_PEP=0.1;
MC_Malate=1.0;
MC_Pyruvate=2;%%%%%%0.02,0427%%%%%%
MC_PGA=0.3;
MC_FBP=0.04;
MC_UDPG=0.035;
MC_SUCP=0.0;
MC_SUC=0.0;
MC_F26BP=7.8E-5;
MC_ATP=0.35;%0.35
MC_T3P=0.55;
MC_HexP=2.4;
MC_Sucrose=0.0;
%Mchl_
Mchl_OAA= 0.005;
Mchl_Malate =1.8;
Mchl_PEP =0.1;
Mchl_Pyruvate= 0.01;
Mchl_NADPH= 0.21;
Mchl_ATP= 1.4;
Mchl_PGA= 0.04;
Mchl_DPGA= 0.0;
Mchl_T3P= 0.6;
%BSC
BSC_T3P= 0.45;
BSC_PGA= 0.2;
BSC_Malate=0.8;
BSC_Pyruvate= 2;%%%0427,0.15%%%%
BSC_CO2= 0.001;
%Bchl
Bchl_CO2= 0.01;
Bchl_RuBP= 2.0;
Bchl_PGA= 0.3;
Bchl_DPGA= 0;
Bchl_ATP= 1.4;
Bchl_NADPH= 0.1;
Bchl_SBP= 0.015;
Bchl_S7P= 0.045;
Bchl_FBP= 0.06;
Bchl_E4P= 0.05;
Bchl_Starch= 0.0;
Bchl_Rubisco= 1.456965457;
Bchl_T3P= 0.5;
Bchl_HexP= 6;%2.2;WY2002
Bchl_Pent =0.05;
Bchl_Malate= 0.3;
Bchl_Pyruvate= 0.23;

Bchl_PGCA=0.0029;
Bchl_GCA=0.36;
Bchl_GCEA=0.1812;

Bper_GCA=0.36;
Bper_GOA=0.028;
Bper_GLY=1.8;
Bper_SER=7.5;
Bper_HPR=0.0035;
Bper_GCEA=0.1812;

Bchl_PPi=0;
Bchl_ADPG=0;



MC_CO2=0.8*CI;

MC_Glu=15;
MC_OxoG=3;
MC_Asp=5*2;
MC_Ala=5*2;

BSC_OxoG=3;
BSC_Glu=15;
BSC_Asp=5*2;
BSC_Ala=5*2;
BSC_OAA=0;
BSC_PEP=0.1;
BSC_ATP=0.5;
Bchl_OAA=0;


MC_O2=O2;
Mchl_O2=O2;
BSC_O2=O2;
Bchl_O2=O2;

Bchl_PEP=0.1;
Mchl_GCEA=0.1812;

Bmito_OAA=0;
Bmito_MAL=0;
Bmito_PYR=5;
Bmito_CO2=0.001;
Bmito_NADH=0.3;

Bchl_Asp=0;
Bchl_Ala=0;
Mchl_Asp=0;
Mchl_Ala=0;
E_PPDK_Mchl=0;% E_PPDK active
EP_PPDK_Mchl=0.616;%1.28;%EP_PPDK inactive %WY202010 0.616 asumming Vmax_PPDK = 80


con=zeros(1,81);
con(1)=MC_HCO3;
con(2)=MC_OAA;
con(3)=MC_PEP;
con(4)=MC_Malate;
con(5)=MC_Pyruvate;
con(6)=MC_PGA;
con(7)=MC_FBP;
con(8)=MC_UDPG;
con(9)=MC_SUCP;
con(10)=MC_SUC;
con(11)=MC_F26BP;
con(12)=MC_ATP;
con(13)=MC_T3P;
con(14)=MC_HexP;
con(15)=MC_Sucrose;
con(16)=Mchl_OAA;
con(17)=Mchl_Malate;
con(18)=Mchl_PEP;
con(19)=Mchl_Pyruvate;
con(20)=Mchl_NADPH;
con(21)=Mchl_ATP;
con(22)=Mchl_PGA;
con(23)=Mchl_DPGA;
con(24)=Mchl_T3P;
con(25)=BSC_T3P;
con(26)=BSC_PGA;
con(27)=BSC_Malate;
con(28)=BSC_Pyruvate;
con(29)=BSC_CO2;
con(30)=Bchl_CO2;
con(31)=Bchl_RuBP;
con(32)=Bchl_PGA;
con(33)=Bchl_DPGA;
con(34)=Bchl_ATP;
con(35)=Bchl_NADPH;
con(36)=Bchl_SBP;
con(37)=Bchl_S7P;
con(38)=Bchl_FBP;
con(39)=Bchl_E4P;
con(40)=Bchl_Starch;
con(41)=Bchl_Rubisco;
con(42)=Bchl_T3P;
con(43)=Bchl_HexP;
con(44)=Bchl_Pent;
con(45)=Bchl_Malate;
con(46)=Bchl_Pyruvate;

con(47)=Bchl_PGCA;
con(48)=Bchl_GCA;
con(49)=Bchl_GCEA;

con(50)=Bper_GCA;
con(51)=Bper_GOA;
con(52)=Bper_GLY;
con(53)=Bper_SER;
con(54)=Bper_HPR;
con(55)=Bper_GCEA;
con(56)=MC_CO2;

con(57)=Bchl_PPi;
con(58)=Bchl_ADPG;

con(59)=MC_Glu;
con(60)=MC_OxoG;
con(61)=MC_Asp;
con(62)=MC_Ala;
con(63)=BSC_OxoG;
con(64)=BSC_Glu;
con(65)=BSC_Asp;
con(66)=BSC_Ala;
con(67)=BSC_OAA;
con(68)=BSC_PEP;
con(69)=BSC_ATP;
con(70)=Bchl_OAA;
con(71)=MC_O2;
con(72)=Mchl_O2;
con(73)=BSC_O2;
con(74)=Bchl_O2;
con(75)=Bchl_PEP;%%%%%%%WY PPDK in BSCytosol
con(76)=Mchl_GCEA;

con(77)=Bmito_OAA;
con(78)=Bmito_MAL;
con(79)=Bmito_PYR;
con(80)=Bmito_CO2;
con(81)=Bmito_NADH;

con(82)=Bchl_Asp;
con(83)=Bchl_Ala;
con(84)=Mchl_Asp;
con(85)=Mchl_Ala;
con(86)=E_PPDK_Mchl;% E_PPDK active
con(87)=EP_PPDK_Mchl;%EP_PPDK inactive
