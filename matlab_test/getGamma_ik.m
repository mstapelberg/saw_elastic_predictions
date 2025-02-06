function [Gamma]=getGamma_ik(C,n)

Gamma=zeros(3,3);

if (size(size(C),2)~=4) || (min(size(n))~=1)
    fprintf('Matrices are the wrong size')
else
    for ii=1:3
        for kk=1:3
            Gamma(ii,kk)=C(ii,1,kk,1)*n(1)*n(1)+C(ii,1,kk,2)*n(1)*n(2)+C(ii,1,kk,3)*n(1)*n(3)+C(ii,2,kk,1)*n(2)*n(1)+C(ii,2,kk,2)*n(2)*n(2)+C(ii,2,kk,3)*n(2)*n(3)+C(ii,3,kk,1)*n(3)*n(1)+C(ii,3,kk,2)*n(3)*n(2)+C(ii,3,kk,3)*n(3)*n(3);
        end
    end
end

end