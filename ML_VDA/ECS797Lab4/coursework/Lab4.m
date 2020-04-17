function Lab4()
%------------------------------------------------------------LAB2-------------------------------------------------------------
% Implicit Shape Model for action detection (for static actions => the bounding box is stationary)
%-----------------------------------------------------------------------------------------------------------------------------
% The local descriptor votes for the spatial center of the action and
% start and end frame of the action
% Lab4.m       ----->   main file
% ./data/      ----->   contains training descriptors
% ./test_fea/  ----->   contains test descriptors (structures)

% run load  data/struct_class1.mat to view the structure containing training descriptors
%{
  
Example: DataStructure contains all the training local descriptors from a class

NoSTIPS is the numnber of Space Time Interest Points (STIPs) detected in the image sequences of the class in question.

DataStructure =  

       patches: [162x NoSTIPs double]   -------->     HOG/HOF descriptors (NoSTIPs ; dimns = 162
       spa_center: [2xNoSTIPs double]   -------->     [xc;yc]
       spa_offset: [2xNoSTIPs double]   -------->     [x-xc;y-yc] where (x,y) are the spatial locations of the descriptor
       hei_wid_bb: [2xNoSTIPs double]   -------->     height and width of the bounding box
         location: [2xNoSTIPs double]   -------->     spatial locations of the descriptor 
        spa_scale: [1xNoSTIPs double]   -------->     spatial scale
        tem_scale: [1xNoSTIPs double]   -------->     temporal scale
       tem_offset: [1xNoSTIPs double]   -------->     frame - frame_center
    st_end_offset: [2xNoSTIPs double]   -------->     offsets for start and end frame = [frame;frame] - [start_frame;end_frame]
     frame_center: [1xNoSTIPs double]   -------->     frame;frame - frame_center
     st_end_frame: [2xNoSTIPs double]   -------->     start and end frame of the action
        frame_num: [1xNoSTIPs double]   -------->     frame number
          seq_num: [1xNoSTIPs double]   -------->     N/A
            label: [1xNoSTIPs double]   -------->     N/A
            class: [1xNoSTIPs double]   -------->     class 1 => boxing; class 2 => handclapping;  class 3 = handwaving
%}


% We use the terms dictionary and codewords interchangebly in this document
train_ism();    
% This generates the codebook and spatial occurance distribution (Vote maps) for the classes and are 
%saved in the folders data/class1,2,3 ...
%-----------------------------------------------------------------------------------------------------------------------------
% Testing:
%-----------------------------------------------------------------------------------------------------------------------------
%  load data/test_class1.mat contains index of the test actions

test_ism();    % Testing ism

recall_prec_curve();        % Generating recall-precision curves
er = eval_error(); 
fprintf('Mis-sclassification rate = %f\n', er);