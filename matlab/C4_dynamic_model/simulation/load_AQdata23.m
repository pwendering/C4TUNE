function [final_A,final_Asd]=load_AQdata23(genotype)
[~,~,~,uniq_acc23,acc_plot23]=load_common_accessions22_23();
AQ23=readtable("../data/2023_AQcurves&lightSaturatedGasExchange_maize.csv",'VariableNamingRule','preserve');
%%
removeplots=["1173_3","1005_2","2120_2","1148_2","2245_3"];
% "1005_1"?
for k=1:length(removeplots)
    ind=find(AQ23{:,"plot_id"}==removeplots(k));
    AQ23=AQ23(setdiff(1:size(AQ23,1),ind),:); % no time data is available
end

%%
plots=acc_plot23(contains(uniq_acc23,genotype),:);
Plot_Rep=strcat(string(repelem(plots,3)),'_',string(repmat(1:3,1,2)));
% Plot_Rep with 0_1, 0_2 or 0_3 means that plot is not found in the
% dataset
A=zeros(6,length(Plot_Rep));
Q=A;
for i=1:length(Plot_Rep)
    ind_i=contains(AQ23{:,"plot_id"},Plot_Rep(i));
    if sum(ind_i)==6
        A(:,i)=AQ23{ind_i,"Photo"};
    end
end
keep=sum(A,1)~=0;
A=A(:,keep);

% subplot(2,5,j)
% 
% for i=1:size(A,2)
%     plot(Q(:,i),A(:,i),'o')
%     hold on
% end
% title(common_acc(a))
% legend(strcat("Line",string(1:size(A,2))),'Location','southeast')

final_A=mean(A,2,"omitnan");
final_Asd=std(A,0,2,"omitnan");
%% 
% 12