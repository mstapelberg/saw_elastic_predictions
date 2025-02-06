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
                               