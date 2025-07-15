function []=initialization_flux_output()
% This function is needed to restart the record

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


