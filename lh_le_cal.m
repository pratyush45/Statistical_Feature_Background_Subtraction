function [LH,LE] = lh_le_cal(X,offset,siz_kr,siz_X,max_val_X)
sprintf('lh_le_cal : begin')
tic;
% Direction Convention
% [1 0] = Direction down starting from [1 1]
% [0 1] = Direction right starting from [1 1]
% [-1 0] = Direction up starting from [siz_kr siz_kr]
% [0 -1] = Direction left starting from [siz_kr siz_kr]
offset_dir = offset./max(abs(offset)); % Direction of Search
flr = floor(siz_kr/2);
padd_X= padarray(X,[flr flr],0,'both'); % Padding the Matrix "X" with zeros
comatx = zeros(max_val_X+1,max_val_X+1); % Declaring and Defining the size of Cooccurence Matrix
LH = zeros(siz_X(1),siz_X(2)); % Declaring and Defining the size of Local Homogenity Matrix
LE = zeros(siz_X(1),siz_X(2)); % Declaring and Defining the size of Local Entropy Matrix
k = [0 0];% Shift after reaching boundary of a matrix (always opposite to offset_dir variable)
iter_lr = flr + 1; % Iteration Variable for Outer Loop
iter_lim = ((siz_kr-1)-max(abs(offset))+1)*siz_kr; % Iteration Variable for loop parsing the matrix "X"

% Setting the shift variable
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
	
	% Raw Implementation of graycomatrix function
	% TODO : Diagonal support for finding GLCM
    % Outer Loop
    for i=iter_lr:iter_lr + (siz_X(1)-1)
        for j=iter_lr:iter_lr + (siz_X(2)-1)
            tmp = padd_X(i-flr:i+flr, j-flr:j+flr);
            tmp_x = tmp_cor(1);
            tmp_y = tmp_cor(2);
            k_t = k; % Resetting the shift variable
            % Loop Parsing the Matrix
            for l = 1:iter_lim
                tmp_x_off = tmp_x + offset(1);
                tmp_y_off = tmp_y + offset(2);
                comatx(tmp(tmp_x,tmp_y) + 1,tmp(tmp_x_off,tmp_y_off) + 1) = comatx(tmp(tmp_x,tmp_y) + 1,tmp(tmp_x_off,tmp_y_off) + 1) + 1; 
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
            comatx(:,:) = comatx(:,:) + comatx(:,:).'; % Symmetric Cooccurrence Matrix
            comatx_sum = sum(sum(comatx(:,:)));
            C = double(comatx(:,:))./comatx_sum;  
            LH(i-flr,j-flr) = localhomogeneity(C,max_val_X+1); % Local Homogenity Calculation
            LE(i-flr,j-flr) = localentropy(C,max_val_X+1); % Local Entropy Calculation
            comatx = zeros(max_val_X+1,max_val_X+1); % Resetting the values in Cooccurrence Matrix
        end
    end
toc
sprintf('lh_le_cal : end')
end


    
