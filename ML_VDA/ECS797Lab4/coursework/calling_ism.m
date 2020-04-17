function calling_ism(THRESH, class, dir_nam, K)
% THRESH : used to set the threshold for the distance between the
%          dictionary atom and the discriptor
% class  : indicates the class parameter
% dir_nam : Name of the directory
% K:    Dictionary/codebook size
  path = cd;
  path = [path,'/data/',dir_nam]; 
  aa = path;
  system(['mkdir',' ',aa,'/']);
  load(sprintf('data/struct_class%d',class));

  [Dictionary,DataStructureVotemap] = ism(DataStructure, K, THRESH);

  save([aa,'/Dictionary'],'Dictionary');    
  save([aa,'/DataStructureVotemap'],'DataStructureVotemap');    % save the Dictionary and DataStructureVotemap information for each class
  clear Dictionary DataStructureVotemap 