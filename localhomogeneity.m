function [LH]=localhomogeneity(C,max_val_X)

LH=0;
 for i=1:max_val_X
    for j=1:max_val_X
        LH = LH + (1/(1+abs(i-j)^2)).* C(i,j);
    end
end

