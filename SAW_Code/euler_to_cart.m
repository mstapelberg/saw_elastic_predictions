function [x,y,z]=euler_to_cart(alpha,beta,gamma,mag)

x=mag.*(cos(gamma).*cos(alpha)-cos(beta).*sin(alpha).*sin(gamma));
y=mag.*(-sin(gamma).*cos(alpha)-cos(beta).*sin(alpha).*cos(gamma));
z=mag.*(sin(beta).*sin(alpha));

end