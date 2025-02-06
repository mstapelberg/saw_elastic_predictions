function C=getCijkl(mat)
C=zeros(3,3,3,3); %Preallocation of C

if (strcmp(mat.class,'cubic'))
    C11=mat.C11;
    C12=mat.C12;
    C44=mat.C44;
    [C(1,1,1,1),C(2,2,2,2),C(3,3,3,3)]=deal(C11);
    [C(1,1,2,2),C(1,1,3,3),C(2,2,3,3)]=deal(C12);
    [C(2,3,2,3),C(1,3,1,3),C(1,2,1,2)]=deal(C44);
     
elseif (strcmp(mat.class,'tetragonal'))
    C11=mat.C11;               %*****
    C12=mat.C12;               %*****
    C13=mat.C13;                %*****
    C33=mat.C33;
    C44=mat.C44;
    C66=mat.C66;
    [C(1,1,1,1),C(2,2,2,2)]=deal(C11);
    [C(1,1,2,2)]=deal(C12);
    [C(1,1,3,3),C(2,2,3,3)]=deal(C13);
    [C(3,3,3,3)]=deal(C33);
    [C(2,3,2,3),C(1,3,1,3)]=deal(C44);
    [C(1,2,1,2)]=deal(C66);
    
elseif (strcmp(mat.class,'hexagonal'))
    C11=mat.C11;
    C12=mat.C12;
    C13=mat.C13;
    C33=mat.C33;
    C44=mat.C44;
    C66=1/2*(C11-C12);
    [C(1,1,1,1),C(2,2,2,2)]=deal(C11);
    [C(1,1,2,2)]=deal(C12);
    [C(1,1,3,3),C(2,2,3,3)]=deal(C13);
    [C(3,3,3,3)]=deal(C33);
    [C(2,3,2,3),C(1,3,1,3)]=deal(C44);
    [C(1,2,1,2)]=deal(C66);
    
    
else
    disp('Parameter mismatch happens. Please check database')
    return;
end


    for i=1:3;
        for j=1:3;
            for k=1:3;
                for l=1:3;
                    if C(i,j,k,l)~=0;
                        [C(j,i,k,l),C(i,j,l,k),C(j,i,l,k),C(k,l,i,j)]=deal(C(i,j,k,l));
                    end
                end
            end
        end
    end
    
