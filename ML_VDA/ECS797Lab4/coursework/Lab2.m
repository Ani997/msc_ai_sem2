function Lab2()
%------------------------------------------------------------LAB2-------------------------------------------------------------
% Implicit Shape Model for action detection (for static actions => the bounding box is stationary)
%-----------------------------------------------------------------------------------------------------------------------------
% The local descriptor votes for the spatial center of the action and
% start and end frame of the action
% Lab2.m       ----->   main file
% ./data/      ----->   contains training descriptors
% ./test_fea/  ----->   contains test descriptors (structures)

% run load  data/struct_class1.mat to view the structure containing training descriptors
%{
eg: mystruct contains all the training local descriptors from a class
mystruct =  

          patches: [162x2090 double] -------->     HOG/HOF descriptors (No of descriptors = 2090 ; dimns = 162
       spa_center: [2x2090 double]   -------->     [xc;yc]
       spa_offset: [2x2090 double]   -------->     [x-xc;y-yc] where (x,y) are the spatial locations of the descriptor
       hei_wid_bb: [2x2090 double]   -------->     height and width of the bounding box
         location: [2x2090 double]   -------->     spatial locations of the descriptor 
        spa_scale: [1x2090 double]   -------->     spatial scale
        tem_scale: [1x2090 double]   -------->     temporal scale
       tem_offset: [1x2090 double]   -------->     frame - frame_center
    st_end_offset: [2x2090 double]   -------->     offsets for start and end frame = [frame;frame] - [start_frame;end_frame]
     frame_center: [1x2090 double]   -------->     frame;frame - frame_center
     st_end_frame: [2x2090 double]   -------->     start and end frame of the action
        frame_num: [1x2090 double]   -------->     frame number
          seq_num: [1x2090 double]   -------->     N/A
            label: [1x2090 double]   -------->     N/A
            class: [1x2090 double]   -------->     class 1 => boxing; class 2 => handclapping;  class 3 = handwaving
%}
train_ism();    % You can change the size of the codebook here
% This generates the codebook and spatial occurance distribution for the classes and are saved in data/class1,2,3 ...
%-----------------------------------------------------------------------------------------------------------------------------
% Testing:
%-----------------------------------------------------------------------------------------------------------------------------
%  load data/test_class1.mat contains index of the test actions

ism_test();    % Testing ism

recall_prec_curve();        % Generating recall-precision curves