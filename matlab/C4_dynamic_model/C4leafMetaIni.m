function [LeafMs] = C4leafMetaIni(envFactor)

leafInis= LeafIni(envFactor);

MetaInis= C4Ini(envFactor);

LeafMs = zeros(1,109);
LeafMs(1:7) = leafInis(1:7);

LeafMs(8:94)= MetaInis(1:87);



