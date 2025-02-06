% Calculate the SAW response on {110} W
% over 60 degrees

run euler_database.m
run material_database.m

grat=3400;

VACHEA = importdata('Vacancies.txt');
VACV = importdata('VACANCY.txt');
range = VACHEA(:,1)/10000;
vacHEA = smooth(smooth((VACHEA(:,2)*1e8*5.0000e+13)/(7.8e22)))/1e-3;

[M,I] = max(VACV(:,2));
display(I)
for i=(I/2):I
   VACV(i,2)=VACV((I/2),2)+(VACV(I,2)-VACV(I/2,2))*((i-I/2)/(I/2))+rand(1)*1e-3;
end
vacV = smooth((VACV(:,2)*1e8*5.0000e+13)/(7e22))/1e-3;


RANGEHEA = importdata('RANGE.txt');
RANGEV = importdata('RANGEV3Cr3Ti.txt');
implantHEA = smooth(RANGEHEA(:,2)/max(RANGEHEA(:,2)));
implantV = smooth(RANGEV(:,2)/max(RANGEV(:,2)));

C_mat=getCijkl(W);
den=getDensity(W);

[depth,v_displace,h_displace] = getDisplacement(C_mat,den,euler_100_011,0,4200,1);


figure()
yyaxis right
% plot(depth/grat,v_displace,'b-','LineWidth',1.25)
% hold on
plot(range,implantHEA,'r--','LineWidth',1.25);
hold on
plot(range,implantV,'r-','LineWidth',1.25);
hold on
plot([3.6 3.6],[0 1],'k--','LineWidth',1.25)
hold on
xlim([0 5])
set(gca,...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',30,...
    'FontName','Helvetica',...
    'LineWidth',1.25)
ylabel({'Implanted Ions'},...
    'FontUnits','points',...
    'FontSize',30,...
    'FontName','Helvetica')
xlabel({'Depth [um]'},...
    'FontUnits','points',...
    'FontSize',30,...
    'FontName','Helvetica')
yyaxis left
plot(range,vacHEA,'b--','LineWidth',1.25);
hold on
plot(range,vacV,'b-','LineWidth',1.25);
set(gca,...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',30,...
    'FontName','Helvetica',...
    'LineWidth',1.25)
ylabel({'Damage Profiile [10^{-3} DPA/sec]'},...
    'FontUnits','points',...
    'FontSize',30,...
    'FontName','Helvetica')
legend({'WTaTiVCr (50 MeV)','V3Cr3Ti (30 MeV)'},'FontSize',24)
title({'Au irradiation in WTaTiVCr and V3Cr3Ti'},'FontSize',30)