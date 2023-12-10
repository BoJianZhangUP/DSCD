clear;
%%% parameters seting
netname='vgg16';      %% network
layers='pool5';
image_num='r5'; % 5-oxford5k,6-paris6k,105-oxford105k,106-paris106k,r5-roxford5k,r6-rparis6k,
datapath='..\data\';
train_set='GLD5k';

qe=10;

switch image_num
    case {'5'}
        test_set='oxford5k';
    case {'6'}
        test_set='paris6k';
    case {'r5'}
        test_set='roxford5k';
    case {'r6'}
        test_set='rparis6k';
    case {'106'}
        test_set='paris106k';
    case {'105'}
        test_set='oxford105k';
end

switch test_set
    case {'oxford5k'}
        train_set2='paris6k'; % traditional PCA-whitening dataset
        query_set='oxford5k';
    case {'paris6k'}
        train_set2='oxford5k';
        query_set='paris6k';
    case {'roxford5k'}
        train_set2='rparis6k';
        query_set='roxford5k';   
    case {'rparis6k'}
        train_set2='roxford5k';
        query_set='rparis6k';        
    case {'oxford105k'}
        query_set='oxford5k';
        train_set2='paris6k';
        query_files = dir(fullfile(datapath,netname,'\pool5\',['oxford5k','_query'],'*.mat'));
    case {'paris106k'}
        query_set='paris6k';
        train_set2='oxford5k';
        query_files = dir(fullfile(datapath,netname,'\pool5\',['paris6k','_query'],'*.mat'));
end


test_files = dir(fullfile(datapath,netname,'\pool5\',test_set,'*.mat'));
train_files = dir(fullfile(datapath,netname,'\pool5\',train_set,'*.mat'));
if ~exist("query_files","var")
    query_files = dir(fullfile(datapath,netname,'\pool5\',[test_set,'_query'],'*.mat'));
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


if  strcmpi(test_set,'roxford5k') || strcmpi(test_set,'rparis6k')
    display_rquery_results(test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd);
else
    display_query_results(netname,layers,test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe);
end
