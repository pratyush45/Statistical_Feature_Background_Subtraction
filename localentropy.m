function [LE]=localentropy(C,max_val_X)

LE=0;
for i=1:max_val_X
    for j=1:max_val_X
        if (C(i,j)~=0)
            LE=LE-C(i,j).*log10(C(i,j));
        end
    end
end
%H=-H;
            
    
 

