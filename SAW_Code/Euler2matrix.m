function M=Euler2matrix(a,b,r)

%This program is to convert the Euler angle set to matrix multiplication

%M: transformation matrix
%a,b,r: Input the Euler angles(radian) in order with the intrinsic convention (Bunge's convention) z-x'-z'' 
%It brings the sample coordinate frame into coincidence with the crystalline coordinate frame

Rza=[cos(a) sin(a) 0
    -sin(a) cos(a) 0
    0 0 1];

Rxb=[1 0 0
    0 cos(b) sin(b)
    0 -sin(b) cos(b)];

Rzr=[cos(r) sin(r) 0
    -sin(r) cos(r) 0
    0 0 1];

M=Rzr*Rxb*Rza;

%%Some good properties of M

%First column of M:  old X direction expressed in new coordiante system
%Second column of M: old Y direction expressed in new coordiante system
%Third column of M:  old Z direction expressed in new coordiante system

%First row of M:  new X direction expressed in old coordiante system
%Second row of M: new Y direction expressed in old coordiante system
%Third row of M:  new Z direction expressed in old coordiante system





%%%---Old version---%%%

% Rza=[cos(a) -sin(a) 0
%     sin(a) cos(a) 0
%     0 0 1];
% 
% Rxb=[1 0 0
%     0 cos(b) -sin(b)
%     0 sin(b) cos(b)];
% 
% Rzr=[cos(r) -sin(r) 0
%     sin(r) cos(r) 0
%     0 0 1];
% 
% M1=Rza*Rxb*Rzr;
% %The matrix form is defined in different ways, but M finally is the Matrix with Bunge Angles, confirmed by the PPT Euler Angle To Matrix
% M2=Rzr'*Rxb'*Rza';
% M=M2;