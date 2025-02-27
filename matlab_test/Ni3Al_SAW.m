% Calculate the SAW response on {001} Ni3Al at 500C
% over 90 degrees

run euler_database.m
run material_database.m

C_mat=getCijkl(Ni3Al);
den=getDensity(Ni3Al);

disp(C_mat)
disp(den)

Ni3Al_SAW_111=zeros(2,61);
angles=0:60;
for jj=1:length(angles)
    display(1*jj)
    [Ni3Al_SAW_111(:,jj),~]=getSAW(C_mat,den,euler_110_111,angles(jj),4000,1);
end

% Save 111 data
data_111 = [angles', Ni3Al_SAW_111(1,:)'];
writematrix(data_111, 'Ni3Al_SAW_111_data.txt', 'Delimiter', '\t');

% Create and save 111 plot
figure()
plot(angles,Ni3Al_SAW_111(1,:),'k-','LineWidth',1.25);
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
saveas(gcf, 'Ni3Al_SAW_111_plot.png')
close(gcf)

% Uncomment to calculate and save 001 data
% Ni3Al_SAW_001=zeros(2,61);
% angles_001=45:135;
% for jj=1:length(angles_001)
%     display(1*jj)
%     [Ni3Al_SAW_001(:,jj),~]=getSAW(C_mat,den,euler_100_001,angles_001(jj),4000,1);
% end
% 
% % Save 001 data
% data_001 = [angles_001', Ni3Al_SAW_001(1,:)'];
% writematrix(data_001, 'Ni3Al_SAW_001_data.txt', 'Delimiter', '\t');
% 
% % Create and save 001 plot
% figure()
% plot(angles_001,Ni3Al_SAW_001(1,:),'k-','LineWidth',1.25);
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
% saveas(gcf, 'Ni3Al_SAW_001_plot.png')
% close(gcf)