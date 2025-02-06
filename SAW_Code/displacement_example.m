%plot a few vertical SAW displacements for anisotropic Cu and isotropic W

run material_database.m
run euler_database.m

W_E=4.10106*10^11;
W_nu=0.27945;
W_rho=19300;

grat=4.8; 

C_mat=getCijkl(Cu);
den=getDensity(Cu);


[dis_depth_cu_1,dis_v_cu_1,dis_h_cu_1]=getDisplacement(C_mat,den,euler_100_001,0,grat,0);
[dis_depth_cu_2,dis_v_cu_2,dis_h_cu_2]=getDisplacement(C_mat,den,euler_112_111,0,grat,0);
[dis_depth_w,dis_v_w,dis_h_w]=getDisplacementIso(W_E,W_nu,W_rho,grat,0);

figure()
plot(dis_depth_w/grat,dis_v_w,'k-','LineWidth',1.25)
hold on
plot(dis_depth_cu_1/grat,dis_v_cu_1,'r-','LineWidth',1.25)
hold on
plot(dis_depth_cu_2/grat,dis_v_cu_2,'-','LineWidth',1.25,'Color',[0 0 0.75])
legend({'Isotropic W','$\langle100\rangle\{001\}$ Cu','$\langle11\bar{2}\rangle\{111\}$ Cu'},'Interpreter','latex')
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