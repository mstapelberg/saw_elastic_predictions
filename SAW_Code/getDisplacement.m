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