function []=display_query_results(net,layers,test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe)
for m=1:7
    dim=8*2^(m-1);

    %%%%%%%%%%%%%%%%  PCA-whitening %%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%% PW %%%%%%
    [PW_test_features_pca,PW_query_nocrop_features_pca]=APW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim);

    %%%%%%% APW %%%%%%
    [PWP_test_features_pca,APW_query_nocrop_features_pca]=APW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim,'APW');


    PW_dist=pdist2(PW_test_features_pca,PW_query_nocrop_features_pca,'euclidean');
    [~, PW_ranks] = sort(PW_dist, 'ascend');
    [PWnocrop_map,~] = compute_map (PW_ranks, gnd);

    [ranks_QE] = rank_qe(PW_test_features_pca', PW_query_nocrop_features_pca', PW_ranks,qe);
    [PWnocrop_qe_map,~] = compute_map (ranks_QE, gnd);

    dist=pdist2(PWP_test_features_pca,APW_query_nocrop_features_pca,'euclidean');
    [~, PWPcrop_ranks] = sort(dist, 'ascend');
    [PWPnocrop_map,~] = compute_map (PWPcrop_ranks, gnd);

    [PWPranks_QE] = rank_qe(PWP_test_features_pca', APW_query_nocrop_features_pca', PWPcrop_ranks,qe);
    [PWPnocrop_qe_map,~] = compute_map (PWPranks_QE, gnd);



    fprintf(['>> %s: %s: %s: %d dim:\n   nocrop: PWnocrop_mAP:%.4f,PWnocrop_qe_mAP:%.4f,PWPnocrop_mAP:%.4f,PWPnocrop_qe_mAP:%.4f\n \n'], net,layers,test_set, dim, ...
        PWnocrop_map,PWnocrop_qe_map,PWPnocrop_map,PWPnocrop_qe_map);
end


end

