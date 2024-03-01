function [test_feature_pca,query_feature_pca] = APW_whitening(train_features_normalize,test_features_normalize,query_features,dim,APW)

pp=20; %num(CC=1)
if nargin==5
    %%% APW %%%%
    [PW_feature,coeff,mu,u,s,~]= pca_and_whitening(train_features_normalize,test_features_normalize);
    [train_feature,~,~,d_u,d_s]= pca_and_whitening(train_features_normalize,test_features_normalize,dim);

    XTrain=[PW_feature,max(1,(2*pp/dim))*train_feature];
    XTrain=normalize(XTrain,2,"norm");
    [coeff11,scoreTrain11,~,~,~,mu11]=pca(XTrain);
    test_feature_pca=scoreTrain11(:,1:dim);
    test_feature_pca=normalize(test_feature_pca,2,"norm");

    q_features=normalize(query_features,2,"norm");
    query_features_white=query_pca(q_features,coeff,mu,u,s,size(PW_feature,2));

    query_features_dim_white=query_pca(q_features,coeff,mu,d_u,d_s,dim);
    query_features_white=[query_features_white,max(1,(2*pp/dim))*query_features_dim_white];%
    query_features_white=normalize(query_features_white,2,"norm");

    q_features=(query_features_white-mu11)*coeff11;
    query_feature_pca=q_features(:,1:dim);
    query_feature_pca=normalize(query_feature_pca,2,"norm");
else
    %%%% PW %%%%%
    [PW_feature,coeff,mu,u,s]= pca_and_whitening(train_features_normalize,test_features_normalize,dim);
    test_feature_pca=normalize(PW_feature,2,"norm");
    q_features=normalize(query_features,2,"norm");
    query_features_white=query_pca(q_features,coeff,mu,u,s,dim);
    query_feature_pca=normalize(query_features_white,2,"norm");
end
end
