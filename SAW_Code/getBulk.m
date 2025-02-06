function [v]=getBulk(C,n,den)

n_norm=n./sqrt(dot(n,n));

Gamma=getGamma_ik(C,n_norm);

velocities=sqrt((eig(Gamma))/den);

v=sort(velocities,'descend');

end