function []=display_rquery_results(test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd)
for m=1:7
    ks = [1, 5, 10];
    dim=8*2^(m-1);

    %%%%%%%%%%%%%%%%  PCA-whitening %%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %%%%%%% PW %%%%%%
    [PW_test_features_pca,PW_query_nocrop_features_pca]=APW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim);

    %%%%%%% APW %%%%%%
    [APW_test_features_pca,APW_query_nocrop_features_pca]=APW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim,'APW');


    PW_dist=pdist2(PW_test_features_pca,PW_query_nocrop_features_pca,'euclidean');
    [~, PW_ranks] = sort(PW_dist, 'ascend');
    compute_r_map(dim,gnd,PW_ranks,test_set,'nocrop','PW_nocrop__map');


    dist=pdist2(APW_test_features_pca,APW_query_nocrop_features_pca,'euclidean');
    [~, APWcrop_ranks] = sort(dist, 'ascend');
    compute_r_map(dim,gnd,APWcrop_ranks,test_set,'nocrop','APW_nocrop__map');

   
end

end

