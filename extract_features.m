function features=extract_features(files,list,ind,cov_x)
features=[];

parfor i=1:size(files,1)
    path=[files(i).folder,'/',list{i},'.mat'];
  
    pool5 = importdata(path);
    
    feature = apply_DSCD_aggregation(pool5,ind,cov_x);

    features = [features;feature];
    if mod(i,1000) == 0
        i
    end
end


end