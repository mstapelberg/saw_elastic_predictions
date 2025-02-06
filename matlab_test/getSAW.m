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


