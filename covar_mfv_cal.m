function [covar_mat,mfv_mat] = covar_mfv_cal(int_mat,hom_mat,ent_mat,siz_kr,siz_input)
sprintf('covar_cal : begin')
tic;

% Relevant Variables are Declared and Defined
flr = floor(siz_kr./2);
tmp_var = 0;

% Padding the Matrix [Intensity Homogenity Entropy] with zeros
padd_X = zeros(siz_input(1)+(flr(1)*2),siz_input(2)+(flr(2)*2),3);
padd_X(:,:,1) = padarray(int_mat,[flr(1) flr(2)],0,'both');
padd_X(:,:,2) = padarray(ent_mat,[flr(1) flr(2)],0,'both');
padd_X(:,:,3) = padarray(hom_mat,[flr(1) flr(2)],0,'both');

iter_lr = [flr(1)+1 flr(2)+1]; % Iteration Variable for Outer Loop
covar_mat = zeros(max(siz_kr),max(siz_kr),siz_input(1),siz_input(2)); % Covariance Matrix
mfv_mat = zeros(1,3,siz_input(1),siz_input(2)); % Mean Feature Vector Matrix

% Matrix for temporary use
tmp = zeros(1,3,siz_kr(1),siz_kr(2)); 

    %Outer Loop
    for i=iter_lr(1):iter_lr(1) + (siz_input(1)-1)        
        for j=iter_lr(2):iter_lr(2) + (siz_input(2)-1) 
            
            % 2 Loops in [Intensity Homogenity Entropy] for each pixel(i,j)
            % This Loop Performs Operation Similar to (f(k) - U(x,y))
            for k=1:3 
                tmp_var = padd_X(i-flr(1):i+flr(1),j-flr(2):j+flr(2),k);
                mfv_mat(1,k,i-flr(1),j-flr(2)) = sum(sum(tmp_var))./(siz_kr(1)*siz_kr(2));
                tmp(1,k,:,:) = tmp_var -  mfv_mat(1,k,i-flr(1),j-flr(2)); 
            end
            % This Loop Performs Operation Similar to sum(X(k)*X(k)') where
            % X(k) = (f(k) - U(x,y))
            for m=1:siz_kr(1) 
                for n=1:siz_kr(2)
                      covar_mat(:,:,i-flr(1),j-flr(2)) = covar_mat(:,:,i-flr(1),j-flr(2)) + tmp(:,:,m,n).'*tmp(:,:,m,n);
                end
            end
            % Dividing the summation by region X*Y
            covar_mat(:,:,i-flr(1),j-flr(2)) = covar_mat(:,:,i-flr(1),j-flr(2))./(siz_kr(1)*siz_kr(2));
        end   
    end
toc
sprintf('covar_cal : end')
end


    
