function Euler=Matrix2Euler(g);
%This function is used to get Euler angles from Euler matrix 
%Euler: the Euler angles 
%g: the Euler matrix

psid=acosd(g(3,3)); %The second angle is from 0 to 180, the same range as cosd

phy1d=atan2d(g(3,1)/sind(psid),-g(3,2)/sind(psid)); %The range for atan2d is from -180 to 180, so add 360 degree to convert negative angles
if phy1d<0
    phy1d=phy1d+360;
end

phy2d=atan2d(g(1,3)/sind(psid),g(2,3)/sind(psid));
if phy2d<0
    phy2d=phy2d+360;
end
Euler=[phy1d, psid, phy2d];
