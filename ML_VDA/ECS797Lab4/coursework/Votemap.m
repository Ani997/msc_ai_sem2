% This function computes the voting maps for one class.
 
function DataStructureVotemap = Votemap(DataStructure, Dictionary, flag_mat)
% Dictionary           : Dictionary for each class
% DataStructureVotemap : Data structure used for storing Voting maps 
% DataStructure        : Input Data structure  (is shown in Lab4.m)
% flag_mat             : is a matrix with dimension K x NoSTIPs. Each entry of the matrix a_{ij} = 1 if distance(codeword_i,descriptor_j) < dist_thresh  

%-----------------------------------------------------------------------------------------------------
%To construct DataStructureVotemap For each descriptor matched to a codebook,
% you need to store the spatial offset, offsets for start and end frames, spatial scale, temporal scale, 
% height and width of the bounding box(given by the element DataStructure.hei_wid_bb) for the codeword. 
%-----------------------------------------------------------------------------------------------------
  
  
  dictionary_size = size(Dictionary,2);
  spa_offset = DataStructure.spa_offset;   % Spatial offset % offset = [x y] - [x_c y_c]
  spa_scale = DataStructure.spa_scale;      
  tem_scale = DataStructure.tem_scale;    
  st_end_offset = DataStructure.st_end_offset;
  hei_wid_bb = DataStructure.hei_wid_bb;
  DataStructureVotemap = [];
  
  
  tot_patches = sum(flag_mat,2);
  
  for i = 1:dictionary_size
    DataStructureVotemap.offset(i).tot_cnt = tot_patches(i);   % Count of the total number of patches for a codeword
       
    if(tot_patches(i)>0)
      DataStructureVotemap.offset(i).spa_offset = spa_offset(:,flag_mat(i,:))./[spa_scale(:,flag_mat(i,:));spa_scale(:,flag_mat(i,:))]; 
      DataStructureVotemap.offset(i).st_end_offset = st_end_offset(:,flag_mat(i,:))./[tem_scale(:,flag_mat(i,:));tem_scale(:,flag_mat(i,:))];
      DataStructureVotemap.offset(i).spa_scale = spa_scale(:,flag_mat(i,:));
      DataStructureVotemap.offset(i).temp_scale = tem_scale(:,flag_mat(i,:));
      DataStructureVotemap.offset(i).hei_wid_bb = hei_wid_bb(:,flag_mat(i,:))./[spa_scale(:,flag_mat(i,:));spa_scale(:,flag_mat(i,:))];
    end
  end
