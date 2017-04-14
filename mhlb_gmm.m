function[fg,mfv_mat_mf,covar_mat_mf,w_mf] = mhlb_gmm(mfv_mat,covar_mat,mfv_mat_mf,covar_mat_mf,w_mf,...
    siz_X,no_of_model_frames,no_of_bg_frames,alpha,Tp,Tb)
sprintf('mhlb_gmm : begin')
tic;
fg = zeros(siz_X(1),siz_X(2)); % Output Frame
mhlb_dist = zeros(1,no_of_model_frames); % Mahalanobis Distance of Each Current Pixel wrt Model Frames

    for i=1:siz_X(1)
        for j=1:siz_X(2)
			for k=1:no_of_model_frames
				
				match = 0;
				
				% Mahanalobis Distance               
				tmp_mat = mfv_mat(1,:,i,j) - mfv_mat_mf(1,:,i,j,k); % Operation similar to (U(i,j)-Um(i,j))
				inv_mat_mf = inv(covar_mat_mf(:,:,i,j,k));
				tmp_var = sqrt(abs(tmp_mat*inv_mat_mf*tmp_mat.'));
				if ~(isnan(tmp_var) || tmp_var == inf) % Discarding Extreme Values
					if ~(tmp_var > 255 || tmp_var < 0.001) % Discarding Unecessary Small or Large Values
						mhlb_dist(k) = tmp_var;
					end
				end
				
			% Code for GMM Implementation begins here
				
				if(mhlb_dist(k)) <= Tp
					
					match = 1;
					
					% Update Weight, Mean Feature Vector and Covariance Matrix
					w_mf(1,k,i,j) = (1-alpha)*w_mf(1,k,i,j) + alpha;
					p = alpha/w_mf(1,k,i,j);
					mfv_mat_mf(1,:,i,j,k) = (1-p)*mfv_mat_mf(1,:,i,j,k) + p*mfv_mat(1,:,i,j);
					covar_mat_mf(:,:,i,j,k) = covar_mat_mf(:,:,i,j,k).*(1-p) + covar_mat(:,:,i,j).*p;
				else
					w_mf(1,k,i,j) = (1-alpha)*w_mf(1,k,i,j); % Weight is slightly lower
				end
			end
			%mhlb_dist
			w_mf(1,:,i,j) = w_mf(1,:,i,j)./sum(w_mf(1,:,i,j));
			
			if (match == 0)
				[~, min_w_index] = min(w_mf(1,:,i,j));  
				mfv_mat_mf(1,:,i,j,min_w_index) = mfv_mat(1,:,i,j);
				covar_mat_mf(:,:,i,j,min_w_index) = covar_mat(:,:,i,j);
			end
			
			% sort weight values in decreasing order			
            [~,rank_ind] = sort(w_mf(1,:,i,j),'descend');
			
			% Foreground Detection
			
			match = 0;
			k = 1;

			while ((match == 0) && (k <= no_of_bg_frames))
			
				if (w_mf(1,rank_ind(k),i,j) >= Tb)
					if (mhlb_dist(rank_ind(k)) <= Tp)
						fg(i,j) = 0;
						match = 1;
					else
						fg(i,j) = 255;     
					end
				end
				k = k+1;
			end
		end
    end
toc
sprintf('mhlb_gmm : end')
end
