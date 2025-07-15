function [final_A,final_Asd,final_gs,final_gssd]=load_ACIdata23(genotype)
%%
[~,~,~,uniq_acc23,acc_plot23]=load_common_accessions22_23();

curve_data23=readtable("../data/2023_ACi_rawData_maize.csv",'VariableNamingRule','preserve');
%%

removeplots=[2191,2011,2122,1253,1017];
removereps=[3,3,1,1,2];

for k=1:length(removeplots)
    ind=intersect(find(curve_data23{:,"Plot"}==removeplots(k)),find(curve_data23{:,"repeat"}==removereps(k)));
    curve_data23=curve_data23(setdiff(1:size(curve_data23,1),ind),:); % no time data is available
end
%%


[~,arg_ind]=ismember(genotype,uniq_acc23);
% figure
Atot=[];
Ctot=[];
gstot=[];
Tairtot=[];
for p=1:2
    plot_i=acc_plot23(arg_ind,p);
    if plot_i~=0
        for r=1:3
            ind=intersect(find(curve_data23{:,"Plot"}==plot_i),find(curve_data23{:,"repeat"}==r));
            if ~isempty(ind)
                A=curve_data23{ind,"A"};
                C=curve_data23{ind,"CO2_r"};
                gs=curve_data23{ind,"gsw"}/1.6;
                Tair=curve_data23{ind,"Tair"};
                
                if length(A)==11
                    Atot=[Atot,A];
                    Ctot=[Ctot,C];
                    gstot=[gstot,gs];
                    Tairtot=[Tairtot,Tair];
                    % plot(C,A,'o')
                    % hold on
                end
            end 
        end
    end
end
%%

final_A=mean(Atot,2,"omitnan");
final_Asd=std(Atot,0,2,"omitnan");
% final_C=mean(C_t,2,"omitnan");
% final_Ci=mean(Ci_t,2,"omitnan");
final_gs=mean(gstot,2,"omitnan");
final_gssd=std(gstot,0,2,"omitnan");
final_Tair=mean(Tairtot,2,"omitnan");
final_Tairsd=std(Tairtot,0,2,"omitnan");
