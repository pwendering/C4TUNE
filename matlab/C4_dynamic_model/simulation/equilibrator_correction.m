function [var,ke_ind]=equilibrator_correction(init_sol)
[param_name,~]=load_parameter_name();
ke_ind=find(contains(param_name,"Ke"));
Ke_name=param_name(ke_ind);
% Ke_value0=init_sol(ke_ind);
% 
% ind=[1,3,4,9:18,21,23:28,30,32:34,36,37,41,43:45,48,51,52];
% Ke_EC=enzymes(ind);
% 
% 
Ke_value=zeros(length(Ke_name),1);
Ke_value(1)=NaN; % Cannot estimate ΔrG'° because the uncertainty is too high
Ke_value(2)=8.9e4;
Ke_value(3)=7.4e-3;
Ke_value(4)=0.05;
Ke_value(5)=9.4e3;
Ke_value(6)=1.5e2;
Ke_value(7)=4.6e3;
Ke_value(8)=2.1e2;
Ke_value(9)=0.02;
Ke_value(10)=0.2;
Ke_value(11)=0.4;
Ke_value(12)=0.3;
Ke_value(13)=5.3e4;
Ke_value(14)=3; % no results about starch synthase
Ke_value(15)=0.05; % no results about starch synthase
Ke_value(16)=9.4e3;
Ke_value(17)=1.5e2;
Ke_value(18)=3;
Ke_value(19)=0.05;
Ke_value(20)=2;
Ke_value(21)=0.2;
Ke_value(22)=1.3e3;
Ke_value(23)=0.3;
Ke_value(24)=5.734; %Warning: The ΔG' of ATP hydrolysis is highly affected by Mg2+ ions.
Ke_value(25)=4.3e2;
Ke_value(26)=Ke_value(24);
Ke_value(27)=8.5e2;
Ke_value(28)=Ke_value(25);
Ke_value(29)=30;
Ke_value(30)=6;
Ke_value(31)=1.8e5;
Ke_value(32)=2.6e2;
Ke_value(33)=0.4302;
Ke_value(34)=30;
Ke_value(35)=8.5e2;


% check=[Ke_value0,Ke_value];
% 
% lb=0.1*Ke_value;
% ub=10*Ke_value;
% 
% ind=find(Ke_value0<=ub & Ke_value0>=lb);


%%

% [res0,zA0,zgs0]=simulate_ACI2(init_sol,"SSA_00001",2022);
% 
% zAnew=zeros(35,1);
% zgsnew=zAnew;
% for i=1:length(Ke_value)
%     var=init_sol;
%     if ~isnan(Ke_value(i))
%         var(ke_ind(i))=Ke_value(i);
%         [~,zA,zgs]=simulate_ACI2(var,"SSA_00001",2022);
%         zAnew(i)=zA;
%         zgsnew(i)=zgs;
%     end
% end

%%
final_ind=[2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 25 27 28 29 30 31 32 34 35];

% reduce A 3,19
% % not good: 5,7,12,14,15,18,19,20,27,
% notgood=setdiff(1:length(Ke_value),final_ind);
var=init_sol;
var(ke_ind(final_ind))=Ke_value(final_ind);

% [res,zA,zgs]=simulate_ACI2(var,"SSA_00001",2022);
% 
% 
% subplot(1,2,1)
% plot(res(:,end),res(:,1),'o')
% ylim([0,45])
% subplot(1,2,2)
% plot(res0(:,end),res0(:,1),'o')
% ylim([0,45])
