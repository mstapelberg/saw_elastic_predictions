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



