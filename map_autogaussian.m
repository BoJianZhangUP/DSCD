function S=map_autogaussian(X,m,n,crop)

d=4;
[x,y]=size(X);

lambda=(nnz(X)/(x*y))/d;
[i, j] = meshgrid(1:y, 1:x);
S = exp(-lambda * sqrt((i-n).^2 + (j-m).^2));

end




