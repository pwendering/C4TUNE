function []=initialize_reaction_rates()
% This function is needed to record the simulated fluxes at different time
% point

%Reaction rate record
global TIME_M;
global OLD_TIME_M;
global Meta_VEL;
TIME_M=0;
OLD_TIME_M=0;
Meta_VEL=zeros(1,9);

global TIME_N;
global OLD_TIME;

TIME_N=0;
OLD_TIME=0;
global Gs_VEL;
Gs_VEL=zeros(1,9);

global TIME_K;
global OLD_TIME_K;
global MY_VEL;
TIME_K=0;
OLD_TIME_K=0;
MY_VEL=zeros(1,124);
