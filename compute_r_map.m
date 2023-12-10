function compute_r_map(dim,gnd,ranks,test_dataset,crop,PCA)
% This code is from open source: https://github.com/filipradenovic/revisitop/tree/master/matlab
ks = [1, 5, 10];
gndE=gnd;
gndM=gnd;
gndH=gnd;
% search for easy (E setup)
for i = 1:numel(gndE), gndE(i).ok = [gndE(i).easy]; gndE(i).junk = [gndE(i).junk, gndE(i).hard]; end
[mapE, apsE, mprE, prsE] = rcompute_map (ranks, gndE, ks);
% search for easy & hard (M setup)
for i = 1:numel(gndM), gndM(i).ok = [gndM(i).easy, gndM(i).hard]; gndM(i).junk = gndM(i).junk; end
[mapM, apsM, mprM, prsM] = rcompute_map (ranks, gndM, ks);
% search for hard (H setup)
for i = 1:numel(gndH), gndH(i).ok = [gndH(i).hard]; gndH(i).junk = [gndH(i).junk, gndH(i).easy]; end
[mapH, apsH, mprH, prsH] = rcompute_map (ranks, gndH, ks);

fprintf(['>> %s: %s: %s:  %d  \n'],test_dataset,crop,PCA,dim)
fprintf('>> %s: mAP E: %.2f, M: %.2f, H: %.2f\n', test_dataset, 100*mapE, 100*mapM, 100*mapH);
fprintf('>> %s: mP@k[%d %d %d] E: [%.2f %.2f %.2f], M: [%.2f %.2f %.2f], H: [%.2f %.2f %.2f]\n\n', test_dataset, ks(1), ks(2), ks(3), 100*mprE, 100*mprM, 100*mprH);
