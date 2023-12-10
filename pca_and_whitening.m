function [oxford_feature,coeff,mu,u,s,idx] = pca_and_whitening(XTrain,XText,dim)

[coeff,scoreTrain,~,~,explained,mu]=pca(XTrain);

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
oxford_feature=xPCAWhite';

end


