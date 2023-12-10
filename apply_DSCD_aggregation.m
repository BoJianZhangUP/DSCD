function features = apply_DSCD_aggregation(X,ind,cov_x)

%%% Channel semantic correlation learning
e=1*10^(-5);
X = imresize(X, 1.5);
X(X<0)=0;
[H,W,N] = size(X);

E_max=reshape(cov_x(:,ind(1)),[1,N]);
XX=zeros(size(X));
[E_maxs,ind2]=sort(E_max,'descend');
E_maxs(E_maxs<0)=0;
i_vec = 1:N;
t1_over_2i2 = E_maxs./(2.*i_vec.^2);
CC = min(1, t1_over_2i2).^i_vec;
vec3d=permute(CC,[1 3 2]);
X_mult = bsxfun(@times, vec3d, X(:,:,ind2(i_vec)));
XX(:,:,ind2(i_vec)) = X_mult;

pp = sum(CC == 1); % The value of pp is 20, which is the parameter in adaptive PCA-whitening
s=round(min(H,W)/4); % Size of the convolution kernel

SP=sum(XX,3);
SR=((SP+sum(X,3)./N));
SR(SR<0)=0;

%%% Hierarchical attention mechanism

% Object attention
if var(SR(:))>=mean(diag(cov_x))
    SR=SR.^2;
else
    f = fspecial('disk', s); % circular mean convolution kernel
    SR=conv2(SR,f,'same');

end
XO=X.*normalization(SR);
[C] = col_max_pooling(XO,N,e); % Channel attention coefficients
XO_c = XO.*permute(C,[1,3,2]);%

%%% Focus attention
% Part1
X1 =X.*focus_attention_module(SR);
[C] = col_max_pooling(X1,N,e);
X1_c = X1.*permute(C,[1,3,2]);%;

% Part2
k=zeros(s,s);
k(1:2:end,1:2:end)=1; % dilated convolution kernel
X_k=conv2(SR,k,'same');
[m,n]=find(X_k==max(max(X_k)));
i2=mean(m);
j2=mean(n);
X2=X.*focus_attention_module(X_k,i2,j2);
X12=X1+X2;
[C] = col_max_pooling(X12,N,e);


X12_c = X12.*permute(C,[1,3,2]);%

G=XO_c+X1_c+X12_c;
G(G<0)=0;

features=pooling(G,N);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rst= focus_attention_module(S1,i,j)

if nargin < 3
[i,j]=get_mean_point(S1);
end
rst2=map_autogaussian(S1,i,j);

rst = bsxfun(@times, S1, rst2);
rst_norm = sqrt(sum(rst(:).^2));
rst = (rst /rst_norm).^(1/2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rst= normalization(SR)

rst=SR;
rst_norm = sqrt(sum(rst(:).^2));
rst = (rst /rst_norm).^(1/2);%

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function features= pooling(X2,N)

rst1 = reshape(sum(X2,[1,2]),[1,N]);

rst2 = reshape(sum(max(X2)),[1,N]);

features=[rst1,rst2];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [channel_wt] = col_max_pooling(X3,N,e)

t=reshape((sum((max(X3)))),[1,N]);

t2=reshape(sum(X3,[1,2]),[1,N]);

channel_wt = sqrt((mean(t)./(t+e)))+sqrt((mean(t2)./(t2+e)));

end
