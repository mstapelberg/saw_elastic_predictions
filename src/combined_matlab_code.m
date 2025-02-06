%% Example code

% Calculate the SAW response on {001} Ni3Al at 500C
% over 90 degrees

run euler_database.m
run material_database.m

C_mat=getCijkl(Ni3Al);
den=getDensity(Ni3Al);

Ni3Al_SAW_111=zeros(2,61);
angles=0:60;
for jj=1:length(angles)
    display(1*jj)
    [Ni3Al_SAW_111(:,jj),~]=getSAW(C_mat,den,euler_110_111,angles(jj),4000,1);
end

figure()
plot(angles,Ni3Al_SAW_111(1,:),'k-','LineWidth',1.25);
% hold on
% plot(angles,Ni3Al_SAW_001(2,:),'r-','LineWidth',1.25);
set(gca,...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',16,...
    'FontName','Helvetica',...
    'LineWidth',1.25)
ylabel({'SAW speed [m/s]'},...
    'FontUnits','points',...
    'FontSize',20,...
    'FontName','Helvetica')
xlabel({'Relative surface angle [degrees]'},...
    'FontUnits','points',...
    'FontSize',20,...
    'FontName','Helvetica')
title('Ni3Al SAW speeds on (111)')

% Ni3Al_SAW_001=zeros(2,61);
% angles=45:135;
% for jj=1:length(angles)
%     display(1*jj)
%     [Ni3Al_SAW_001(:,jj),~]=getSAW(C_mat,den,euler_100_001,angles(jj),4000,1);
% end
% 
% 
% figure()
% plot(angles,Ni3Al_SAW_001(1,:),'k-','LineWidth',1.25);
% % hold on
% % plot(angles,Ni3Al_SAW_001(2,:),'r-','LineWidth',1.25);
% set(gca,...
%     'FontUnits','points',...
%     'FontWeight','normal',...
%     'FontSize',16,...
%     'FontName','Helvetica',...
%     'LineWidth',1.25)
% ylabel({'SAW speed [m/s]'},...
%     'FontUnits','points',...
%     'FontSize',20,...
%     'FontName','Helvetica')
% xlabel({'Relative surface angle [degrees]'},...
%     'FontUnits','points',...
%     'FontSize',20,...
%     'FontName','Helvetica')
% title('Ni3Al SAW speeds on (100)')


%% Functions

% C_modifi.m
function newC=C_modifi(C,a)
%This is used to transform C in the new coordinate frame based on transformation matrix a
% C: input elastic constants;
% a: transformation matrix

newC=zeros(3,3,3,3);
for ip=1:3;
    for jp=1:3;
        for kp=1:3;
            for lp=1:3;
                for i=1:3;
                    for j=1:3;
                        for k=1:3;
                            for l=1:3;
                                newC(ip,jp,kp,lp)=newC(ip,jp,kp,lp)+a(ip,i)*a(jp,j)*a(kp,k)*a(lp,l)*C(i,j,k,l);
                            end
                        end
                    end
                end
            end
        end
    end
end

% deltaij.m
function f=deltaij(i,j)
%Dirac delta function
if i==j
    f=1;
else
    f=0;
end

% euler_to_cart.m
function [x,y,z]=euler_to_cart(alpha,beta,gamma,mag)

x=mag.*(cos(gamma).*cos(alpha)-cos(beta).*sin(alpha).*sin(gamma));
y=mag.*(-sin(gamma).*cos(alpha)-cos(beta).*sin(alpha).*cos(gamma));
z=mag.*(sin(beta).*sin(alpha));

end

% Euler2matrix.m
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

% getBulk.m
function [v]=getBulk(C,n,den)

n_norm=n./sqrt(dot(n,n));

Gamma=getGamma_ik(C,n_norm);

velocities=sqrt((eig(Gamma))/den);

v=sort(velocities,'descend');

end

% getCijkl.m
function C=getCijkl(mat)
C=zeros(3,3,3,3); %Preallocation of C

if (strcmp(mat.class,'cubic'))
    C11=mat.C11;
    C12=mat.C12;
    C44=mat.C44;
    [C(1,1,1,1),C(2,2,2,2),C(3,3,3,3)]=deal(C11);
    [C(1,1,2,2),C(1,1,3,3),C(2,2,3,3)]=deal(C12);
    [C(2,3,2,3),C(1,3,1,3),C(1,2,1,2)]=deal(C44);
     
elseif (strcmp(mat.class,'tetragonal'))
    C11=mat.C11;               %*****
    C12=mat.C12;               %*****
    C13=mat.C13;                %*****
    C33=mat.C33;
    C44=mat.C44;
    C66=mat.C66;
    [C(1,1,1,1),C(2,2,2,2)]=deal(C11);
    [C(1,1,2,2)]=deal(C12);
    [C(1,1,3,3),C(2,2,3,3)]=deal(C13);
    [C(3,3,3,3)]=deal(C33);
    [C(2,3,2,3),C(1,3,1,3)]=deal(C44);
    [C(1,2,1,2)]=deal(C66);
    
elseif (strcmp(mat.class,'hexagonal'))
    C11=mat.C11;
    C12=mat.C12;
    C13=mat.C13;
    C33=mat.C33;
    C44=mat.C44;
    C66=1/2*(C11-C12);
    [C(1,1,1,1),C(2,2,2,2)]=deal(C11);
    [C(1,1,2,2)]=deal(C12);
    [C(1,1,3,3),C(2,2,3,3)]=deal(C13);
    [C(3,3,3,3)]=deal(C33);
    [C(2,3,2,3),C(1,3,1,3)]=deal(C44);
    [C(1,2,1,2)]=deal(C66);
    
    
else
    disp('Parameter mismatch happens. Please check database')
    return;
end


    for i=1:3;
        for j=1:3;
            for k=1:3;
                for l=1:3;
                    if C(i,j,k,l)~=0;
                        [C(j,i,k,l),C(i,j,l,k),C(j,i,l,k),C(k,l,i,j)]=deal(C(i,j,k,l));
                    end
                end
            end
        end
    end

% getDensity.m
function density=getDensity(mat)
density=mat.density;

% getDisplacement.m 

function [depth,v_displace,h_displace] = getDisplacement(C,rho,Euler,deg,grat,plotty)
%Function takes the model from Royer (1984) to calculate anisotropic
%surface acoustic wave displacements using the method of SAW speed
%calculation from Du (2017)

sampling=4000; %this is standard, can move up to 40000 for higher fidelty
[v_R,~,~]=getSAW(C,rho,Euler,deg,sampling,0); %v_R in m/s

depth=0:0.01:(grat*10);
depth_calc=depth*10^(-6);

%grating wave vector
K=2*pi/(grat*10^(-6));

%code snippet stolen from Du for rotation control

%Align the y axis to the direction in interest
MM=[cosd(deg) cosd(90-deg) 0;
    cosd(90+deg) cosd(deg) 0;
    0 0 1];                   %This rotation matrix is equal to rotate axes anti-clockwise by deg

RR=[0 1 0;
    0 0 1;
    1 0 0]; %transformation matrix from Du convention to Royer convention, but I think this shouldn't matter for cubic materials

C=C_modifi(C,(RR*Euler2matrix(Euler(1),Euler(2),Euler(3))*MM)');% Express C in the aligned sample frame
% C=C_modifi(C,(Euler2matrix(Euler(1),Euler(2),Euler(3))*MM)');% Express C in the aligned sample frame

%are these being indexed right?
%cij all in Pa
c11=C(1,1,1,1);
c12=C(1,1,2,2);
c22=C(2,2,2,2);
c66=C(1,2,1,2);
c0=c11-c12^2/c22;

%assign xi_R from Green's function calculation
xi_R=rho*v_R^2;

%define the polynomial from which you get the important roots
pol4=c22*c66;
pol2=c22*(c11-xi_R)+c66*(c66-xi_R)-(c12+c66)^2;
pol0=(c11-xi_R)*(c66-xi_R);

%take those roots to get q_sq values, two of them
q_sq=roots([pol4 pol2 pol0]);

a0=sqrt(c0/xi_R-1);

if isreal(q_sq)
    chi=abs(sqrt(q_sq));
    a=(c11-xi_R-c66.*chi.^2)./((c12+c66).*chi);
        
    u_x=exp(-chi(1)*K*depth_calc)-sqrt(a(1)/a(2))*exp(-chi(2)*K*depth_calc);
    u_z=a0*(exp(-chi(2)*K*depth_calc)-sqrt(a(1)/a(2))*exp(-chi(1)*K*depth_calc));
else
    g=abs(real(sqrt(q_sq(1))));
    h=abs(imag(sqrt(q_sq(1))));
      
    comp1=c22*(c0-xi_R)+c12*xi_R;
    comp2=c22*(c0-xi_R)-c12*xi_R;
    alpha=atan((comp1/comp2)*(h/g));
    
    u_x=2*exp(-h*K*depth_calc).*cos(g*K*depth_calc+alpha);
    u_z=2*a0*exp(-h*K*depth_calc).*cos(g*K*depth_calc-alpha);
end

v_displace=u_z/u_z(1);
h_displace=u_x/u_z(1);

% if plotty
%     figure()
%     plot(depth/grat,v_displace,'r-','LineWidth',1.25)
%     hold on
% %     plot(depth/grat,h_displace,'b-','LineWidth',1.25)
% %     hold on
%     plot([0 depth(end)],[0 0],'k--','LineWidth',1.25)
% %     legend('Vertical','Horizontal')
%     xlim([0 4])
%     set(gca,...
%         'FontUnits','points',...
%         'FontWeight','normal',...
%         'FontSize',16,...
%         'FontName','Helvetica',...
%         'LineWidth',1.25)
%     ylabel({'SAW Displacement [a.u.]'},...
%         'FontUnits','points',...
%         'FontSize',20,...
%         'FontName','Helvetica')
%     xlabel({'Depth [z/\Lambda]'},... %['Depth [' 956 'm]']
%         'FontUnits','points',...
%         'FontSize',20,...
%         'FontName','Helvetica')
% end

end

% getDisplacementIso.m
function [depth,v_displace,h_displace] = getDisplacementIso(E,nu,rho,grat,plotty)
%Methods takes the SAW displacement profiles for isotropic solids based on
%a model found in full in Lin Thesis (2014) using the SAW speed calculation
%from Malischewsky's (2005) polynomial solution to Rayleigh's equation

depth=0:0.01:(grat*10);
depth_calc=depth*10^(-6);

mu=E/(2*(1+nu)); %these define the Lame constants, this is just the shear modulus
lambda=(E*nu)/((1+nu)*(1-2*nu)); %this is the other Lame constant

v_S=sqrt(mu/rho);
v_P=sqrt((lambda+2*mu)/rho);
v_R=v_S*(0.874+0.196*nu-0.043*nu^2-0.055*nu^3); %this is Malischewsky's formula

K=2*pi/(grat*10^(-6)); %grating wave vector

q=sqrt(1-v_R/v_P)*K;
s=sqrt(1-v_R/v_S)*K;

u_z=-q*exp(-q*depth_calc)+((2*K^2*q)/(s^2+K^2))*exp(-s*depth_calc);
u_x=(-K*exp(-q*depth_calc)+((2*K*q*s)/(s^2+K^2))*exp(-s*depth_calc))*-1;

v_displace=u_z/u_z(1);
h_displace=u_x/u_z(1);

if plotty
    figure()
    plot(depth/grat,v_displace,'r-','LineWidth',1.25)
    hold on
    plot(depth/grat,h_displace,'b-','LineWidth',1.25)
    hold on
    plot([0 depth(end)],[0 0],'k--','LineWidth',1.25)
    legend('Vertical','Horizontal')
    xlim([0 4])
    set(gca,...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',16,...
        'FontName','Helvetica',...
        'LineWidth',1.25)
    ylabel({'SAW Displacement [a.u.]'},...
        'FontUnits','points',...
        'FontSize',20,...
        'FontName','Helvetica')
    xlabel({'Depth [z/\Lambda]'},... %['Depth [' 956 'm]']
        'FontUnits','points',...
        'FontSize',20,...
        'FontName','Helvetica')
end

% [~,ave_ind_1]=min(abs(depth-grat));
% [~,ave_ind_2]=min(abs(depth-grat/2));
% total=sum(v_displace);
% 
% grat_ave=sum(v_displace(1:ave_ind_1))/total
% half_ave=sum(v_displace(1:ave_ind_2))/total

end

% getGamma_ik.m
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

% getSAW.m
function [v,index,intensity]=getSAW(C,rho,Euler,deg,sampling,psaw,varargin)
%index:direction of SAW referred to crystalline coordinate frame
%v: SAW velocity
%C: fourth-tensor elastic constants
%rho: density
%Euler:Euler angle describing orientation in radian
%deg: the angle rotated on surface plane referred to y direction, 0<=deg<180 in degree
%Sampling:resolution in k space, usually 400 or 4000

%Check Euler range
if any(Euler > 9)||any(Euler<-9)
    warning('Euler angles are expressed in radian not degree. Please check! ');
end

% Check deg range
if deg>180
    warning('Check convention, deg should be between 0 and 180 in degree')
end

if nargin<6
    psaw=0;
end

%Set sampling parameter
if sampling==40000
    T=20*10^-14;
elseif sampling==4000
    T=20*10^-13;
elseif sampling==400
    T=20*10^-12;
else
    T=20*10^-12;
    disp('samling is not a value often used')
end

lambda=7*10^-6; %m
k0=2*pi/lambda;
w0=2*pi/T;
w=w0+complex(0,0.000001*w0);

% Preallocate
F=zeros(3,3);
B=zeros(3,3);
M=zeros(3,3);
N=zeros(3,3);
POL=cell(3,3);
pp=zeros(1,3);
S=cell(1,3);
x=zeros(1,3);
y=zeros(1,3);
z=zeros(1,3);
A=zeros(3,3);
R=zeros(3,3);
I=zeros(3,3);
a=zeros(1,3);
G33=zeros(1,sampling);

%Align the y axis to the direction in interest
MM=[cosd(deg) cosd(90-deg) 0;
    cosd(90+deg) cosd(deg) 0;
    0 0 1];                   %This rotation matrix is equal to rotate axes anti-clockwise by deg

C=C_modifi(C,(Euler2matrix(Euler(1),Euler(2),Euler(3))*MM)');% Express C in the aligned sample frame
index=(Euler2matrix(Euler(1),Euler(2),Euler(3))*MM)*[0;1;0]; % (Surface rotation has been considered) Get the direction in interest expressed in crystalline coordinate frame. (Sample is old, Crystal is new

%evaluate the initial k
for nx=1
    for ny=1:sampling
        
        k(1)=nx*k0;
        k(2)=ny*k0;
        
        %evaluate the quadratic coefficient
        for i=1:3
            for j=1:3
                F(i,j)=C(i,3,3,j);
            end
        end
        
        %evaluate the linear coefficient
        for i=1:3
            for l=1:3
                B(i,l)=0;
            end
        end
        
        for i=1:3
            for l=1:3
                for u=1:2
                    B(i,l)=B(i,l)-k(u)*C(i,u,3,l);
                end
                for v=1:2
                    B(i,l)=B(i,l)-k(v)*C(i,3,v,l);
                    
                end
                
                M(i,l)=B(i,l)*complex(0,1);
            end
        end
        
        % Evaluate the constant coefficient
        for i=1:3
            for l=1:3
                N(i,l)=0;
            end
        end
        
        for i=1:3
            for l=1:3
                N(i,l)=N(i,l)+rho*w^2*deltaij(i,l);
                for u=1:2
                    for v=1:2
                        N(i,l)=N(i,l)-C(i,u,v,l)*k(u)*k(v);
                    end
                end
            end
        end
        
        % Get the final expression
        for i=1:3
            for j=1:3
                POL{i,j}=[F(i,j) M(i,j) N(i,j)];
            end
        end
        Poly=conv(conv(POL{1,1},POL{2,2}),POL{3,3})+conv(conv(POL{1,2},POL{2,3}),POL{3,1})+conv(conv(POL{1,3},POL{2,1}),POL{3,2})-conv(conv(POL{1,1},POL{2,3}),POL{3,2})-conv(conv(POL{1,2},POL{2,1}),POL{3,3})-conv(conv(POL{1,3},POL{2,2}),POL{3,1});
        ppC=roots(Poly);
        cont=0;
        for i=1:length(ppC)
            if real(ppC(i)>0)
                cont=cont+1;
                pp(cont)=ppC(i);
            end
        end
        
        
        % Solve to get A;;;;Notice imaginary number
        for i=1:3
            S{i}=F*pp(i)^2+M*pp(i)+N;
            temp=S{i};
            
            [NA,NB,NC]=svd(temp);
            Sol=NC(:,end);
            x(i)=Sol(1);
            y(i)=Sol(2);
            z(i)=Sol(3);
            
            
        end
        
        % %Choose the first set as the solution of A; First index: the number of set;
        % %Second: the order of the components
        for i=1:3
            A(i,1)=x(i);
            A(i,2)=y(i);
            A(i,3)=z(i);
        end
        
        if(nx==0)&&(ny==0)
            A=eye(3);
        end
        
        %boundary conditions to determine a
        for i=1:3
            for r=1:3
                R(i,r)=0;
                I(i,r)=0;
            end
        end
        
        for i=1:3
            for r=1:3
                for l=1:3
                    R(i,r)=R(i,r)+C(i,3,3,l)*pp(r)*A(r,l);
                    for u=1:2
                        I(i,r)=I(i,r)+C(i,3,u,l)*k(u)*A(r,l);
                    end
                end
            end
        end
        Comb=-R+I*complex(0,1);
        del=[0 0 1]';
        
        % Solve a by using determinant operatios
        for r=1:3
            Aug=Comb;
            Aug(:,r)=del;
            a(r)=det(Aug)/det(Comb);
        end
        
        %get G33 value
        G33(nx,ny)=0;
        for r=1:3
            G33(nx,ny)=G33(nx,ny)+a(r)*A(r,3);
        end
    end
end

inc=1;
xx=1:sampling;
yy=real(G33(1,:));
xnew=1:inc:sampling;
ynew=spline(xx,yy,xnew);
[YYnew]=H_L_Peak(ynew,psaw);
Num=1+inc*YYnew;
slownessnew=Num*k0/real(w);
v=1./slownessnew;

%Block to find the relative intensity of each of the SAW and PSAW modes
intensity=zeros(1,length(YYnew));

for jj=1:length(YYnew)
    pos_ind=YYnew(jj)+1;
    go=1;
    loop_idx=pos_ind;
    while go
        if ynew(loop_idx-2)<ynew(loop_idx-1)
            go=0;
            neg_ind=loop_idx-1;
        else
            loop_idx=loop_idx-1;
        end
    end
    intensity(jj)=ynew(neg_ind)-ynew(pos_ind);
end

%if required, draw the displacement-slowness profile
if nargin>6&&strcmp(varargin{1},'draw')
    figure() 
    hax=axes;
    ny=1:sampling;
    Nsx=1;
    plot(ny*k0/real(w),real(G33(Nsx,ny)),'b','LineWidth',2);
    hold on
    line([slownessnew(1) slownessnew(1)],get(hax,'YLim'),'color','red')
    hold on
    if psaw
        line([slownessnew(2) slownessnew(2)],get(hax,'YLim'),'color','red')
        hold on
    end
    set(gca,...
            'FontUnits','points',...
            'FontWeight','normal',...
            'FontSize',16,...
            'FontName','Times',...
            'LineWidth',1)
        ylabel({'Displacement (arb. unit)'},...
            'FontUnits','points',...
            'interpreter','latex',...
            'FontSize',20,...
            'FontName','Times')
        xlabel({'Slowness (s/m)'},...
            'FontUnits','points',...
            'interpreter','latex',...
            'FontSize',20,...
            'FontName','Times')
end

% H_L_Peak.m
function [y]=H_L_Peak(var,psaw)
%This function is to return the RW peak location

%y: RW peak slowness position
%var: displacement profile

%Find SAW by finding positions in displacement vs slowness that have a
%negative instantaneous slope
PeakTotal=find((diff(var))<0);
if isempty(PeakTotal)
    y=0;
elseif psaw
    y=[PeakTotal(end) PeakTotal(end-1)];
else
y=(PeakTotal(end));
end

end

% Matrix2Euler.m
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





%% Databases 
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

% material_database.m
%The paper 'Elastic constants of single crystal' is put in short E

Si=struct('fomular','Si', 'C11',165.7e9, 'C12',63.9e9, 'C44',79.56e9, 'density',2.329e3, 'class','cubic');%data is from paper in folder data base 
Ni=struct('fomular','Ni', 'C11',248.1e9, 'C12',154.9e9, 'C44',124.2e9, 'density',8.91e3, 'class','cubic');%data is from paper paper E
%W=struct('fomular','W', 'C11',522.39e9, 'C12',204.37e9, 'C44',160.83e9, 'density',19.257e3, 'class','cubic');%data is from paper paper E
% Al=struct('fomular','Al', 'C11',106.75e9, 'C12',60.41e9, 'C44',28.34e9, 'density',2.697e3, 'class','cubic');%data is from paper paper E
Sn=struct('fomular','Sn', 'C11',75.29e9, 'C12',61.56e9, 'C13',44.00e9, 'C33',95.52e9, 'C44',21.93e9, 'C66',23.36e9, 'density',7.29e3, 'class','tetragonal');%data is from paper paper E
Co=struct('fomular','Co', 'C11',307.1e9, 'C12',165.0e9, 'C13',102.7e9, 'C33',358.1e9, 'C44',75.5e9, 'density',8.836e3,'class','hexagonal');
AlN=struct('fomular','AlN', 'C11',464e9, 'C12',149e9, 'C13',116e9, 'C33',409e9, 'C44',128e9, 'density',3.26e3,'class','hexagonal');%[6]
Random=struct('fomular','Random', 'C11',195.1e9, 'C12',154.9e9, 'C44',124.2e9, 'density',8.91e3, 'class','cubic');
%AlN=struct('fomular','AlN', 'C11',396e9, 'C12',137e9, 'C13',108e9, 'C33',373e9, 'C44',116e9, 'density',3.26e3,'class','hexagonal');%[6]
%AlN=struct('fomular','AlN', 'C11',369e9, 'C12',145e9, 'C13',120e9,'C33',395e9, 'C44',96e9, 'density',3.26e3,'class','hexagonal');%[9]


% Elastic constants added by CAD after 08/2017

%Aggregated from Simmons and Wang for RT data
Ge=struct('formular','Ge','C11',129.35e9,'C12',48.65e9,'C44',66.94e9,'density',5.323e3,'class','cubic');
%Elastic constants for Nb from Carroll J. Appl. Phys. (1965)
Nb=struct('formular','Nb','C11',245.6e9,'C12',138.7e9,'C44',29.30e9,'density',8.5605e3','class','cubic');
%^ check those
%Elastic contants for Cu from Overton (1954)
Cu=struct('formular','Cu','C11',168.4e9,'C12',121.4e9,'C44',75.39e9,'density',8.96e3,'class','cubic');
Cu_00dpa=struct('formular','Cu','C11',165.0e9,'C12',119.0e9,'C44',73.51e9,'density',8.96e3,'class','cubic');
Cu_90dpa=struct('formular','Cu_90dpa','C11',142.7e9,'C12',102.9e9,'C44',52.93e9,'density',7.81e3,'class','cubic');
%Elastic constants for W aggregated from sources listed in Dennett (2016)
W=struct('formular','W','C11',522.796e9,'C12',203.468e9,'C44',160.668e9,'density',19.25e3,'class','cubic');
%Elastic constants from Gerlich and Fisher 1969, this data is available for
%other temps, too see docs for APL paper 2017
Al_RT=struct('fomular','Al', 'C11',106.49e9, 'C12',60.39e9, 'C44',28.28e9, 'density',2.700e3, 'class','cubic');

%Aggregated from Simmons and Wang for 273, 280, and 280K data
Mo=struct('formular','Mo','C11',456.77e9,'C12',166.11e9,'C44',112.3267e9,'density',10.28e3,'class','cubic');

%Aggregated from Simmons and Wang for 300 and 298K data
Fe=struct('formular','Fe','C11',229.03e9,'C12',135.81e9,'C44',116.78e9,'density',7.874e3,'class','cubic');

%From Simmons and Wang for 300K, single source
Pd=struct('formular','Pd','C11',227.10e9,'C12',176.04e9,'C44',71.73e9,'density',12.038e3,'class','cubic');

%From Letbetter "PREDICTED MONOCRYSTAL ELASTIC CONSTANTS OF 304-TYPE STAINLESS STEEL" Physica 128B (1985) 1-4 
SS304=struct('formular','SS304','C11',209e9,'C12',133e9,'C44',121e9,'density',7.95e3,'class','cubic');

%Some constants for UO_2 from a couple of sources
%(1979) Fritz
UO2_1=struct('formular','UO2_1','C11',389.3e9,'C12',118.7e9,'C44',59.7e9,'density',10.97e3,'class','cubic');
%(1965) Wachtman
UO2_2=struct('formular','UO2_2','C11',396e9,'C12',121e9,'C44',64.1e9,'density',10.97e3,'class','cubic');


% Cu=struct('fomular','Cu','indepC',[169.0,122.0,75.3],'density',8.96,'class','cubic'); %data is from paper 15M
% Fe=struct('fomular','Fe','indepC',[226,  140,  116],'density',7.8672,'class','cubic');% data is from paper E
% 
% Quartz=struct('fomular','SiO2','indepC',[78.5,16.1],'density',2.2,'class','isotropic');
% GaAs=struct('fomular','GaAs','indepC',[119, 53.8, 59.4],'density',5.5,'class','cubic');
% GaP=struct('fomular','GaP','indepC',[141.2,62.53,70.47],'density',4.1297,'class','cubic');
% KCl=struct('fomular','KCl','indepC',[40.69,7.11,6.31],'density',1.984,'class','cubic');
% TiO2=struct('fomular','TiO2','indepC',[266,173,136,0,470,124,189],'density',4.3,'class','tetragonal');
% TeO2=struct('fomular','TeO2','indepC',[532,486,212,0,1085,244,552],'density',6,'class','tetragonal');
% BSN=struct('fomular','Ba2NaNb5O15','indepC',[239,104,50,247,52,135,65,66,76],'density',5.3,'class','orthorhombic');
% Rochelle=struct('fomular','NaKC4H4O6','indepC',[28,17.4,15,41.4,19.7,39.4,8.7,2.9,9.6],'density',1.77,'class','orthorhombic');
% LiNbO3=struct('fomular','LiNbO3','indepC',[203,53,75,9,245,60],'density',4.70,'class','trigonal','subclass','3m m x1','indepD',[6.92,-4.16,-0.085,0.6]*1e-11,'indepP',[44.3 27.9]*e0);
% Ni3Sn4=struct('fomular','Ni3Sn4','indepC',[239.3, 208.7, 227, 77.9, 67, 70, 83.7, 79, 101.7, -28, 14.4, -7.4, 3.7,],'density',8.65,'class','monoclinic');

%The following data are from the paper: property of elastic surface wave

% R.W. Dickson 1969 Journal of Applied Physics for Ni3Al at 500C
Ni3Al=struct('formular','Ni3Al','C11',150.4e9,'C12',81.7e9,'C44',107.8e9,'density',7.57e3,'class','cubic');

