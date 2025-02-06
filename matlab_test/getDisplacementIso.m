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