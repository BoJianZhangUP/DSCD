function []=display_query_results(net,layers,test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe)
for m=1:7
    dim=8*2^(m-1);

    %%%%%%%%%%%%%%%%  PCA-whitening %%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%% PW %%%%%%
    [PW_test_features_pca,PW_query_nocrop_features_pca]=APW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim);

    %%%%%%% APW %%%%%%
    [APW_test_features_pca,APW_query_nocrop_features_pca]=APW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim,'APW');


    PW_dist=pdist2(PW_test_features_pca,PW_query_nocrop_features_pca,'euclidean');
    [~, PW_ranks] = sort(PW_dist, 'ascend');
    [PWnocrop_map,~] = compute_map (PW_ranks, gnd);

    [ranks_QE] = rank_qe(PW_test_features_pca', PW_query_nocrop_features_pca', PW_ranks,qe);
    [PWnocrop_qe_map,~] = compute_map (ranks_QE, gnd);

    dist=pdist2(APW_test_features_pca,APW_query_nocrop_features_pca,'euclidean');
    [~, APWcrop_ranks] = sort(dist, 'ascend');
    [APWnocrop_map,~] = compute_map (APWcrop_ranks, gnd);

    [APWranks_QE] = rank_qe(APW_test_features_pca', APW_query_nocrop_features_pca', APWcrop_ranks,qe);
    [APWnocrop_qe_map,~] = compute_map (APWranks_QE, gnd);



    fprintf(['>> %s: %s: %s: %d dim:\n   nocrop: PWnocrop_mAP:%.4f,PWnocrop_qe_mAP:%.4f,APWnocrop_mAP:%.4f,APWnocrop_qe_mAP:%.4f\n \n'], net,layers,test_set, dim, ...
        PWnocrop_map,PWnocrop_qe_map,APWnocrop_map,APWnocrop_qe_map);
end


end

