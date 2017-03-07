function [comatx] = glcmcal(X,offset,siz_kr,siz_X,max_val_X)
tic;
% siz_kr = 5; % Size of Kernel
% offset = [-1 0]; % Distance and Direction of Search
% Direction Convention
% [1 0] = Direction down starting from [1 1]
% [0 1] = Direction right starting from [1 1]
% [-1 0] = Direction up starting from [siz_kr siz_kr]
% [0 -1] = Direction left starting from [siz_kr siz_kr]
offset_dir = offset./max(abs(offset)); % Direction of Search
% X=[1 3 0 2 ; 2 0 0 2 ; 1 0 1 3 ;0 1 1 3 ; 3 1 2 1]; % Input Matrix
% siz_X = size(X);% Size of "X"
flr = floor(siz_kr/2);
% max_val_X = max(max(X)); % Largest Value in Matrix "X"
padd_X= padarray(X,[flr flr],0,'both'); % Padding the Matrix "X" with zeros
comatx = zeros(max_val_X+1,max_val_X+1,siz_X(1)*siz_X(2)); % Defining the size of Cooccurence MAtrix
comatx_no = 1;% Counter for Cooccurence Matrix
k = [0 0];% Shift after reaching boundary of a matrix (always opposite to offset_dir variable)
iter_lr = flr + 1; % Iteration Variable for Outer Loop
iter_lim = ((siz_kr-1)-max(abs(offset))+1)*siz_kr; % iteration Variable for loop parsing the matrix "X"

if(offset_dir(1) == 1)
    tmp_cor = [1 1];
    k(2) = 1;
elseif(offset_dir(1) == -1)
    tmp_cor = [siz_kr siz_kr];
    k(2) = -1;
elseif (offset_dir(2) == 1)
    tmp_cor = [1 1];
    k(1) = 1;
elseif (offset_dir(2) == -1)
    tmp_cor = [siz_kr siz_kr];
    k(1) = -1;    
end

    %Outer Loop
    for i=iter_lr:iter_lr + (siz_X(1)-1)
        for j=iter_lr:iter_lr + (siz_X(2)-1)
            tmp = padd_X(i-flr:i+flr, j-flr:j+flr);
            tmp_x = tmp_cor(1);
            tmp_y = tmp_cor(2);
            k_t = k;
            % Loop Parsing the Matrix
            for l = 1:iter_lim
                tmp_x_off = tmp_x + offset(1);
                tmp_y_off = tmp_y + offset(2);
                comatx(tmp(tmp_x,tmp_y) + 1,tmp(tmp_x_off,tmp_y_off) + 1,comatx_no) = comatx(tmp(tmp_x,tmp_y) + 1,tmp(tmp_x_off,tmp_y_off) + 1,comatx_no) + 1; 
                 if(tmp_x_off+offset_dir(1) > siz_kr || tmp_x_off+offset_dir(1) < 1 || tmp_y_off+offset_dir(2) > siz_kr || tmp_y_off+offset_dir(2) < 1) 
                    % The above condition should be independent of x and y coordinates
                    tmp_x = tmp_cor(1) + k_t(1);
                    tmp_y = tmp_cor(2) + k_t(2);
                    k_t = k_t + k;
                 else                                   
                    tmp_x = tmp_x + offset_dir(1);
                    tmp_y = tmp_y + offset_dir(2);
                 end            
            end
            comatx_no = comatx_no + 1; % keep this line before the "end" of "j" loop
        end
    end
toc
end


    
