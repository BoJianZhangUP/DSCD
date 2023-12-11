function [ranks_QE] =rank_qe(PW_test_features_pca, qvecs, ranks,qe)
qnd_qe = qe;

% QE

qvecs_qe = qvecs;
for i=1:size(qvecs,2)
       qvecs_qe(:,i) = mean([qvecs(:,i) PW_test_features_pca(:,ranks(1:qnd_qe,i))],2);  
end

qvecs_qe = normalize(qvecs_qe,1,'norm');


PW_dist=pdist2(PW_test_features_pca',qvecs_qe','euclidean');
[~, ranks_QE] = sort(PW_dist, 'ascend');

end
