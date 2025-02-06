% Calculate the SAW response on {111} Cu
% over 60 degrees

run euler_database.m
run material_database.m

C_mat=getCijkl(Cu);
den=getDensity(Cu);
% C_mat_90=getCijkl(Cu_90dpa);
% den_90=getDensity(Cu_90dpa);

Cu_SAW_111=zeros(2,61);
% Cu_SAW_111_90dpa=zeros(1,61);
angles=0:60;
for jj=1:length(angles)
    display(1*jj)
    [Cu_SAW_111(:,jj),~]=getSAW(C_mat,den,euler_110_111,angles(jj),4000,1);
%     [Cu_SAW_111_90dpa(jj),~]=getSAW(C_mat_90,den_90,euler_110_111,angles(jj),4000);
end

% Cu_SAW_111=[Cu_SAW_111(15:end) Cu_SAW_111(2:15)];
% Cu_SAW_111_90dpa=[Cu_SAW_111_90dpa(15:end) Cu_SAW_111_90dpa(2:15)];

figure()
plot(angles,Cu_SAW_111(1,:),'k-','LineWidth',1.25);
hold on
plot(angles,Cu_SAW_111(2,:),'r-','LineWidth',1.25);
% hold on
% plot(angles,Cu_SAW_111_90dpa,'-','color',[1 0.5 0],'LineWidth',1.25);
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