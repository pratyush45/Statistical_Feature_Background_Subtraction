clc;
clear all;
close all;
t1 = cputime; % Begin Time of the Computation
warning('off','all'); % Turning off all warnings
% Declaration of Relevant Variables
no_of_images = 1180; % Number of Images to consider for computation
model_frame_no = 1; % Begin Frame Number for a dataset
no_of_model_frames = 3; % Number of Model Frames
no_of_bg_frames = 1; % Number of Background Frames to consider
inv_req = 1; % Inverse Matrix Flag ('0' = False,'1' = True)
env = {'baseline','highway','pedestrians';'dynamic background','canoe','fountain01'};
prompt = ' 1.baseline (1.highway 2.pedestrians) \n 2.dynamic background (1.canoe 2.fountain01)';
d_choice = input(strcat(prompt,'\n =')); % dataset choice
e_choice = input(' =') + 1; % environment choice
dataset_path = 'D:\\Education\\Project\\Results\\Testing\\dataset\\';  % Dataset Path (Dataset from changedetection.net)
norm = 32; % Normalizing the values in an Image Matrix
% Direction Convention for offset Variable
% [1 0] = Direction down starting from [1 1]
% [0 1] = Direction right starting from [1 1]
% [-1 0] = Direction up starting from [siz_kr siz_kr]
% [0 -1] = Direction left starting from [siz_kr siz_kr]
offset = [0 2]; % Direction of parsing the Image Matrix
norm_max_X = floor(255/norm); % Max Value present in Image Matrix
tmp_var = 0; % Temporary Variable

% GMM Initialization Constants
Tp = 4; % Distance Threshold
Tb = 0.8; % Weight Threshold
alpha = 0.005; % Learning Rate

% Must Be Odd Values
siz_kr = 3; % Size of Kernel for parsing each pixel in the Image Matrix in "glcmcal" function
region_XY = [3 3]; % Region X * Y over which Covariance for a pixel is computed

% Determining Size of an Image Matrix
I_n = double(rgb2gray(imread(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
    env(d_choice,e_choice),'\\input\\in%06d.jpg')),model_frame_no))));
siz_X = size(I_n);

% Choosing ROI in a frame
init_xy = [1 80]; 
siz_X = [60 80];

I_mf(:,:,1) = I_n(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1);

% Declaration and Size Definition of required Matrix
LH_mf = zeros(siz_X(1),siz_X(2),no_of_model_frames); % Local Homogenity
LE_mf = zeros(siz_X(1),siz_X(2),no_of_model_frames); % Local Entropy
covar_mat_mf = zeros(max(region_XY),max(region_XY),siz_X(1),siz_X(2),no_of_model_frames); % Covariance matrix
mfv_mat_mf = zeros(1,3,siz_X(1),siz_X(2),no_of_model_frames); % Mean feature Vector Matrix 
mhlb_mat = zeros(1,no_of_model_frames,siz_X(1),siz_X(2)); % Mahalanobis Distance Matrix
inv_mat_mf = zeros(max(region_XY),max(region_XY),siz_X(1),siz_X(2),no_of_model_frames); % Inverse Matrix of Model Frames
w_mf = ones(1,no_of_model_frames,siz_X(1),siz_X(2)).*(1/no_of_model_frames); % Weight Assigned to each pixel in respective model frames
tmp_mat = zeros(1,3,siz_X(1),siz_X(2)); % Temporary Matrix used in Mahalanobis Distance Computation

% Input Arguments of lh_le_cal in order :
% (Input Matrix,Offset Direction and Magnitude,Size of Kernel,Size of Input Matrix,Normalized Max value in Input Matrix)
[LH_mf(:,:,1),LE_mf(:,:,1)] = lh_le_cal(floor(I_mf(:,:,1)./norm),offset,siz_kr,siz_X,norm_max_X); 

% Input Arguments of covar_mfv_cal in order :
% (Input,Homogenity Matrix,Entropy Matrix,Region XY,Size of Input,Inverse Matrix Flag)
[covar_mat_mf(:,:,:,:,1),mfv_mat_mf(1,:,:,:,1),inv_mat_mf(:,:,:,:,1)] = covar_mfv_cal(I_mf(:,:,1),LH_mf(:,:,1),LE_mf(:,:,1),region_XY,siz_X,inv_req);

for n = 2:no_of_model_frames
  I_n = double(rgb2gray(imread(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
      env(d_choice,e_choice),'\\input\\in%06d.jpg')),(model_frame_no + n-1)))));
  I_mf(:,:,n) = I_n(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1);
  [LH_mf(:,:,n),LE_mf(:,:,n)] = lh_le_cal(floor(I_mf(:,:,n)./norm),offset,siz_kr,siz_X,norm_max_X); 
  [covar_mat_mf(:,:,:,:,n),mfv_mat_mf(1,:,:,:,n),inv_mat_mf(:,:,:,:,n)] = covar_mfv_cal(I_mf(:,:,n),LH_mf(:,:,n),LE_mf(:,:,n),region_XY,siz_X,inv_req);
end

% Create Video Object
aviobj=VideoWriter(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
         env(d_choice,e_choice),'\\StatFeat\\StatFeat%06d.avi')),no_of_bg_frames));
open(aviobj);

inv_req = 0;

for n=no_of_model_frames + 1:no_of_images
    
    % Calculation of Covariance Matrix and Mean Feature Vector of Current Frame
    I_n = double(rgb2gray(imread(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
        env(d_choice,e_choice),'\\input\\in%06d.jpg')),(model_frame_no + n-1)))));
    I = I_n(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1);
    [LH,LE] = lh_le_cal(floor(I./norm),offset,siz_kr,siz_X,norm_max_X); 
    [covar_mat,mfv_mat] = covar_mfv_cal(I,LH,LE,region_XY,siz_X,inv_req);
    fg = zeros(siz_X(1),siz_X(2)); % Output Frame
    
    for i=1:siz_X(1)
        for j=1:siz_X(2)
            
            % GMM Implementation
            
            match = 0;
            
            for k=1:no_of_model_frames
                
                 % Mahanalobis Distance               
                 tmp_mat = mfv_mat(1,:,i,j) - mfv_mat_mf(1,:,i,j,k); % Operation similar to (U(i,j)-Um(i,j))
                 inv_mat_mf(:,:,i,j,k) = inv(covar_mat_mf(:,:,i,j,k));
                 tmp_var = abs(sqrt(tmp_mat*inv_mat_mf(:,:,i,j,k)*tmp_mat.'));
                 if ~(isnan(tmp_var) || tmp_var == inf) % Discarding Extreme Values
                    if ~(tmp_var > 255 || tmp_var < 0.001) % Discarding Unecessary Small or Large Values
                        mhlb_mat(1,k,i,j) = tmp_var;
                    end
                 end
                 
                 if(mhlb_mat(1,k,i,j)) <= Tp
                     
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
            
            w_mf(1,:,i,j) = w_mf(1,:,i,j)./sum(w_mf(1,:,i,j));
            
            if (match == 0)
                [min_w, min_w_index] = min(w_mf(1,:,i,j));  
                mfv_mat_mf(1,:,i,j,min_w_index) = mfv_mat(1,:,i,j);
                covar_mat_mf(:,:,i,j,min_w_index) = covar_mat(:,:,i,j);
            end
            
            % sort weight values in decreasing order
            
            rank = w_mf(1,:,i,j); 
            rank_ind = [1:1:no_of_model_frames];
            
            for m=2:no_of_model_frames               
                for r=1:(m-1)
                    
                    if (rank(m) > rank(r))                     
                        % swap max values
                        tmp_var = rank(r);  
                        rank(r) = rank(m);
                        rank(m) = tmp_var;
                        
                        % swap max index values
                        tmp_var = rank_ind(r);  
                        rank_ind(r) = rank_ind(m);
                        rank_ind(m) = tmp_var;    

                    end
                end
            end
            
            % Foreground Detection
            
            match = 0;
            k = 1;
            
            fg(i,j) = 0;
            while ((match == 0) && (k <= no_of_bg_frames))

                if (w_mf(1,rank_ind(k),i,j) >= Tb)
                    if (mhlb_mat(1,k,i,j) <= Tp)
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
    writeVideo(aviobj,uint8(fg));
    imwrite(uint8(fg),sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
         env(d_choice,e_choice),'\\StatFeat\\bin%06d.png')),n),'PNG');
end
       
close(aviobj);
t2 = cputime; % End Time of Computation
sprintf('Time Elapsed : %.4f',t2-t1) % Time Taken for the Computation