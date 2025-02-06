% Database of common sets of Euler angles that are used often

%proscription for the matrix is that the surface direction you want {hkl} is the third column, 
%the direction on that surface <ijk> is the second column, and the first column is the cross 
%product <ijk>x{hkl}. All vectors must be normalized. 

euler_100_001=[0 0 0]*pi/180;
euler_112_111=Matrix2Euler([cross([1 1 -2]/sqrt(6),[1 1 1]/sqrt(3))' ([1 1 -2]/sqrt(6))' ([1 1 1]/sqrt(3))'])*pi/180;
euler_110_111=Matrix2Euler([cross([1 -1 0]/sqrt(2),[1 1 1]/sqrt(3))' ([1 -1 0]/sqrt(2))' ([1 1 1]/sqrt(3))'])*pi/180;
euler_100_011=Matrix2Euler([cross([1 0 0],[0 1 1]/sqrt(2))' [1 0 0]' ([0 1 1]/sqrt(2))'])*pi/180;

euler_111_431=Matrix2Euler([cross(([-1 1 1]/sqrt(3)),([4 3 1]/sqrt(26)))' ([-1 1 1]/sqrt(3))' ([4 3 1]/sqrt(26))'])*pi/180;
euler_111_211=Matrix2Euler([cross([1 -1 -1]/sqrt(3),[2 1 1]/sqrt(6))' ([1 -1 -1]/sqrt(3))' ([2 1 1]/sqrt(6))'])*pi/180;
euler_111_532=Matrix2Euler([cross(([1 -1 -1]/sqrt(3)),([5 3 2]/sqrt(38)))' ([1 -1 -1]/sqrt(3))' ([5 3 2]/sqrt(38))'])*pi/180;