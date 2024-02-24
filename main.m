clear;
%%% parameters seting
netname='vgg16';      %% network
layers='pool5';
image_num='r5'; % 5-oxford5k,6-paris6k,105-oxford105k,106-paris106k,r5-roxford,r6-rparis,
datapath='..\data\';
train_set='GLD5k';

qe=10;

switch image_num
    case {'5'}
        test_set='oxford5k';
        query_set='oxford5k';
    case {'6'}
        test_set='paris6k';
        query_set='paris6k';
    case {'r5'}
        test_set='roxford';
        query_set='roxford';  
    case {'r6'}
        test_set='rparis';
        query_set='rparis'; 
    case {'106'}
        test_set='paris106k';
        query_set='paris6k';
        query_files = dir(fullfile(datapath,netname,'\datasets\',['paris6k','_nquery_pool5'],'*.mat'));
    case {'105'}
        test_set='oxford105k';
        query_set='oxford5k';
        query_files = dir(fullfile(datapath,netname,'\datasets\',['oxford5k','_nquery_pool5'],'*.mat'));
end


test_files = dir(fullfile(datapath,netname,'\datasets\'，[test_set,'_pool5'],'*.mat'));
train_files = dir(fullfile(datapath,netname,'\datasets\'，[train_set,'_pool5'],'*.mat'));
if ~exist("query_files","var")
    query_files = dir(fullfile(datapath,netname,'\datasets\'，[test_set,'_nquery_pool5'],'*.mat'));
end

[ind,cov_x]=var_v(train_files); % calculate covariance matrix

eval(['load gnd_' test_set '.mat']);
test_features=extract_features(test_files,imlist,ind,cov_x);
test_features_normalize=normalize(test_features,2,"norm");


train_features=extract_train_features(train_files,ind,cov_x);
train_features_normalize=normalize(train_features,2,"norm");

if ~exist("q_name","var")
    q_name=qimlist;
    qidx=priorindex_queries;
end

if size(query_files,1)==70
    query_nocrop_features=extract_features(query_files,q_name,ind,cov_x);
    query_nocrop_features_normalize=normalize(query_nocrop_features,2,"norm");
else
    query_nocrop_features_normalize=test_features_normalize(qidx,:);
end

warning off;


if  strcmpi(test_set,'roxford') || strcmpi(test_set,'rparis')
    display_rquery_results(test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd);
else
    display_query_results(netname,layers,test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe);
end
