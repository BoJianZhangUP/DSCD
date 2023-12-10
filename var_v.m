function [ind,cov_x] = var_v(files)

max_pool5=[];

parfor i=1:size(files,1)
    files_path=[files(i).folder,'/',files(i).name];
    pool5 = importdata(files_path);
    [~,~,channel] = size(pool5);    
    d = reshape(max(max(pool5)),[1,channel]);  
    max_pool5=[max_pool5;d];
end 
     cov_x=cov(max_pool5);      
     [~,ind]=sort(diag(cov_x),'descend');
end
