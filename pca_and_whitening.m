function [pw_feature,coeff,mu,u,s,idx] = pca_and_whitening(XTrain,XText,dim)

[coeff,scoreTrain,~,~,explained,mu]=pca(XTrain);
%%%%%%    Inspired by: https://ww2.mathworks.cn/help/stats/pca.html?searchHighlight=PCA&s_tid=srchtitle_support_results_1_PCA (idx = find(cumsum(explained)>95,1))%%%%%%%%%
if nargin==2
    sum_explained = 0;
    idx = 0;
    while sum_explained < 99
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    dim=idx;
end

if ~exist("idx","var")
   idx=dim;
end

x_train=scoreTrain(:,1:dim);
sigma=cov(x_train,'omitrows');
[u,s,~]=svd(sigma);

scoreTest=(XText-mu)*coeff;
x_test=scoreTest(:,1:dim);

xRot=x_test*u;

epsilon=1*10^(-5);
xPCAWhite=diag(1./(sqrt(diag(s)+epsilon)))*xRot';
pw_feature=xPCAWhite';

end


