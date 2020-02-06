clear opts % opts is a struct contains some default setting such as the image path and data path, etc. 
rootpath='/home/chris/Desktop/cv_TA/PG_BOW_DEMO-master/'; % change according to your workspace path
%% change these paths to image, data and label location
images_set=strcat(rootpath,'image');
data=strcat(rootpath,'data');
%%
opts.imgpath=images_set; % image path
opts.datapath=data;
% opts.labelspath=labels;
opts.nimages = 400; % number of images in total 
opts.image_names = strcat(rootpath,'data/global/image_names.mat');
%%
% local and global data paths
opts.localdatapath=sprintf('%s/local',opts.datapath);
opts.globaldatapath=sprintf('%s/global',opts.datapath);
%% Descriptors
descriptor_opts.type='sift';                                                     % name descriptor
descriptor_opts.name=['des',descriptor_opts.type]; % output name (combines detector and descrtiptor name)
descriptor_opts.patchSize=16;                                                   % normalized patch size
descriptor_opts.gridSpacing=8; 
descriptor_opts.maxImageSize=1000;
GenerateSiftDescriptors(opts,descriptor_opts);