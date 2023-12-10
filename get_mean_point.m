function [m1,n1]=get_mean_point(Z)
    [m,n]=find(Z~=0);

    m1=mean(m);
    n1=mean(n);    
end


