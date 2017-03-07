clc;
clear all;
close all;
t1 = cputime; % Begin Time of the Computation
warning('off','all'); % Turning off all warnings
% Declaration of Relevant Variables
no_of_images = 2; % Number of Images to consider for computation
model_frame_no = 10; % Begin Frame Number for a dataset
no_of_model_frames = 1; % Number of Model Frames
inv_req = 0; % Inverse Matrix Flag ('0' = False,'1' = True)
env = {'baseline','highway','pedestrians';'dynamic background','canoe','fountain01'};
prompt = ' 1.baseline (1.highway 2.pedestrians) \n 2.dynamic background (1.canoe 2.fountain01)';
d_choice = input(strcat(prompt,'\n =')); % dataset choice
e_choice = input(' =') + 1; % environment choice
norm = 32; % Normalizing the values in an Image Matrix
% Direction Convention for offset Variable
% [1 0] = Direction down starting from [1 1]
% [0 1] = Direction right starting from [1 1]
% [-1 0] = Direction up starting from [siz_kr siz_kr]
% [0 -1] = Direction left starting from [siz_kr siz_kr]
offset = [0 2]; % Direction of parsing the Image Matrix
max_val_X = floor(255/norm); % Max Value present in Image Matrix
tmp_var = 0; % Temporary Variable

% Must Be Odd Values
siz_kr = 3; % Size of Kernel for parsing each pixel in the Image Matrix in "glcmcal" function
region_XY = [3 3]; % Region X * Y over which Covariance for a pixel is computed

% Determining Size of an Image Matrix
I_t(:,:,1) = floor(double(rgb2gray(imread(sprintf(char(strcat('D:\\Education\\Project\\Results\\Testing\\dataset\\',env(d_choice,1),'\\',env(d_choice,e_choice),'\\input\\in%06d.jpg')),model_frame_no))))./norm);
% siz_X = size(I(:,:,1));

% Choosing ROI in a frame
init_xy = [36 8]; 
siz_X = [3 3];
I(:,:,1) = I_t(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1,1);

% Declaration and Size Definition of required Matrix
LH = zeros(siz_X(1),siz_X(2),no_of_images); % Local Homogenity
LE = zeros(siz_X(1),siz_X(2),no_of_images); % Local Entropy
covar_mat = zeros(max(region_XY),max(region_XY),siz_X(1),siz_X(2),no_of_images); % Covariance matrix
mfv_mat = zeros(1,3,siz_X(1),siz_X(2),no_of_images); % Mean feature Vector Matrix 
mhlb_mat = zeros(siz_X(1),siz_X(2),(no_of_images - no_of_model_frames)*no_of_model_frames); % Mahalanobis Distance Matrix
inv_mat_mf = zeros(max(region_XY),max(region_XY),siz_X(1),siz_X(2),no_of_model_frames); % Inverse Matrix of Model Frames
tmp_mat = zeros(1,3,siz_X(1),siz_X(2)); % Temporary Matrix used in Mahalanobis Distance Computation

% Input Arguments of lh_le_cal in order :
% (Input Matrix,Offset Direction and Magnitude,Size of Kernel,Size of Input Matrix,Max value in Input Matrix)
[LH(:,:,1),LE(:,:,1)] = lh_le_cal(I(:,:,1),offset,siz_kr,siz_X,max_val_X); % Choosing ROI in a frame

for n = 2:no_of_images
  I_t(:,:,n) = floor(double(rgb2gray(imread(sprintf(char(strcat('D:\\Education\\Project\\Results\\Testing\\dataset\\',env(d_choice,1),'\\',env(d_choice,e_choice),'\\input\\in%06d.jpg')),(model_frame_no + n-1)))))./norm);
  I(:,:,n) = I_t(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1,n);
  [LH(:,:,n),LE(:,:,n)] = lh_le_cal(I(:,:,n),offset,siz_kr,siz_X,max_val_X); 
end

% Covariance Matrix Generation For Model Frames
inv_req = 1;
for n=1:no_of_model_frames
    [covar_mat(:,:,:,:,n),mfv_mat(1,:,:,:,n),inv_mat_mf(:,:,:,:,n)] = covar_mfv_cal(I(:,:,n),LH(:,:,n),LE(:,:,n),region_XY,siz_X,inv_req);
end

% Mahanalobis Distance
inv_req = 0;
for n=no_of_model_frames + 1:no_of_images
    % Covariance and Mean feature vector for current frame
    [covar_mat(:,:,:,:,n),mfv_mat(1,:,:,:,n)] = covar_mfv_cal(I(:,:,n),LH(:,:,n),LE(:,:,n),region_XY,siz_X,inv_req);
    for k=1:no_of_model_frames
        tmp_mat = mfv_mat(1,:,:,:,n) - mfv_mat(1,:,:,:,k); % Operation similar to (U(i,j)-Um(i,j))
        for i=1:siz_X(1)
            for j=1:siz_X(2)
                 tmp_var = abs(sqrt(tmp_mat(:,:,i,j)*inv_mat_mf(:,:,i,j,k)*tmp_mat(:,:,i,j).'));
                 if ~(isnan(tmp_var) || tmp_var == inf) % Discarding Extreme Values
                    mhlb_mat(i,j,(n-no_of_model_frames)*k) = tmp_var;
                 end
            end
        end
    end
end     

t2 = cputime; % End Time of Computation
sprintf('Time Elapsed : %.4f',t2-t1) % Time Taken for the Computation