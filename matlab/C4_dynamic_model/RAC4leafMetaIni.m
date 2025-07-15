function [RALeafMs] = RAC4leafMetaIni(envFactor)


leafMetaInis= C4leafMetaIni(envFactor);
RuACTInis = zeros(1,4);
%RuACTInis= RuACT_Ini;

%Initial value
Mchl_ActATPsynthase=0;
Mchl_ActGAPDH=0.0;
Mchl_ActNADPMDH=0;
Bchl_ActATPsynthase=3;
Mchl_ActPEPC=0; %0.05; %
Bchl_ActGAPDH=0.3;
Bchl_ActFBPase=0.3;
Bchl_ActSBPase=0.3;
Bchl_ActPRK=0.3;
Bchl_ActRubisco=0.1;
Bchl_ActRca=0.05;

AEMBInis(1)=Mchl_ActATPsynthase;
AEMBInis(2)=Mchl_ActGAPDH;
AEMBInis(3)=Mchl_ActNADPMDH;
AEMBInis(4)=Bchl_ActATPsynthase;
AEMBInis(5)=Mchl_ActPEPC;
AEMBInis(6)=Bchl_ActGAPDH;
AEMBInis(7)=Bchl_ActFBPase;
AEMBInis(8)=Bchl_ActSBPase;
AEMBInis(9)=Bchl_ActPRK;
AEMBInis(10)=Bchl_ActRubisco;
AEMBInis(11)=Bchl_ActRca;

RALeafMs=zeros(1,109);

RALeafMs(1:94) = leafMetaInis(1:94);


% for m=1:4
%     RALeafMs(94+m)= RuACTInis(m);
% end

RALeafMs(99:109)= AEMBInis(1:11);
