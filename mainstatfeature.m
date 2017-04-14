clc;
clear all;
close all;
t1 = cputime; % Begin Time of the Computation
warning('off','all'); % Turning on all warnings
% Declaration of Relevant Variables
no_of_images = 1180; % Number of Images to consider for computation
model_frame_no = 1; % Begin Frame Number for a dataset
no_of_model_frames = 2; % Number of Model Frames
no_of_bg_frames = 1; % Number of Background Frames to consider
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
Tp = 3; % Distance Threshold
Tb = 0.9; % Weight Threshold
alpha = 0.1; % Learning Rate

% Must Be Odd Values
siz_kr = 3; % Size of Kernel for parsing each pixel in the Image Matrix in "glcmcal" function
region_XY = [3 3]; % Region X * Y over which Covariance for a pixel is computed

% Determining Size of an Image Matrix
I_n = double(rgb2gray(imread(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
    env(d_choice,e_choice),'\\input\\in%06d.jpg')),model_frame_no))));
siz_X = size(I_n);

% Choosing ROI in a frame
init_xy = [80 80]; 
siz_X = [90 160];

I = I_n(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1);

% Declaration and Size Definition of required Matrix
LH = zeros(siz_X(1),siz_X(2)); % Local Homogenity
LE = zeros(siz_X(1),siz_X(2)); % Local Entropy
covar_mat_mf = zeros(max(region_XY),max(region_XY),siz_X(1),siz_X(2),no_of_model_frames); % Covariance matrix
mfv_mat_mf = zeros(1,3,siz_X(1),siz_X(2),no_of_model_frames); % Mean feature Vector Matrix 
w_mf = ones(1,no_of_model_frames,siz_X(1),siz_X(2)).*(1/no_of_model_frames); % Weight Assigned to each pixel in respective model frames

% Input Arguments of lh_le_cal in order :
% (Input Matrix,Offset Direction and Magnitude,Size of Kernel,Size of Input Matrix,Normalized Max value in Input Matrix)
[LH,LE] = lh_le_cal(floor(I./norm),offset,siz_kr,siz_X,norm_max_X); 

% Input Arguments of covar_mfv_cal in order :
% (Input,Homogenity Matrix,Entropy Matrix,Region XY,Size of Input,Inverse Matrix Flag)
[covar_mat_mf(:,:,:,:,1),mfv_mat_mf(1,:,:,:,1)] = covar_mfv_cal(I,LH,LE,region_XY,siz_X);

for n = 2:no_of_model_frames
  I_n = double(rgb2gray(imread(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
      env(d_choice,e_choice),'\\input\\in%06d.jpg')),(model_frame_no + n-1)))));
  I = I_n(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1);
  [LH,LE] = lh_le_cal(floor(I./norm),offset,siz_kr,siz_X,norm_max_X); 
  [covar_mat_mf(:,:,:,:,n),mfv_mat_mf(1,:,:,:,n)] = covar_mfv_cal(I,LH,LE,region_XY,siz_X);
end

% Create Video Object
aviobj=VideoWriter(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
         env(d_choice,e_choice),'\\StatFeat\\StatFeat%06d.avi')),no_of_bg_frames));
open(aviobj);

for n=no_of_model_frames + 1:no_of_images
    
    % Calculation of Covariance Matrix and Mean Feature Vector of Current Frame
    I_n = double(rgb2gray(imread(sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
        env(d_choice,e_choice),'\\input\\in%06d.jpg')),(model_frame_no + n-1)))));
    I = I_n(init_xy(1):init_xy(1)+siz_X(1)-1,init_xy(2):init_xy(2)+siz_X(2)-1);
    [LH,LE] = lh_le_cal(floor(I./norm),offset,siz_kr,siz_X,norm_max_X); 
    [covar_mat,mfv_mat] = covar_mfv_cal(I,LH,LE,region_XY,siz_X);
    
    % Input Arguments of mhlb_gmm in order :
    % (mfv_mat(current frame),covar_mat(current frame),mfv_mat_mf(model frame),
    % covar_mat_mf(model frame),w_mf(weight of model frames),size of input,
    % no. of model frames,no. of background frames,learning rate,
    % distance threshold,weight threshold)
    [fg,mfv_mat_mf,covar_mat_mf,w_mf] = mhlb_gmm(mfv_mat,covar_mat,mfv_mat_mf,covar_mat_mf,w_mf,...
        siz_X,no_of_model_frames,no_of_bg_frames,alpha,Tp,Tb);
    writeVideo(aviobj,uint8(fg));
    imwrite(uint8(fg),sprintf(char(strcat(dataset_path,env(d_choice,1),'\\',...
         env(d_choice,e_choice),'\\StatFeat\\bin%06d.png')),n),'PNG');
     
end
       
close(aviobj);
t2 = cputime; % End Time of Computation
sprintf('Time Elapsed : %.4f',t2-t1) % Time Taken for the Computation