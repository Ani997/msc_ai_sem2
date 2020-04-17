function [Dictionary, DataStructureVotemap] = ism(DataStructure,K,THRESH)
% Dictionary           : Dictionary for each class
% DataStructureVotemap : Data structure used for storing Voting maps
% K                    : Dictionary/codebook size
% DataStructure        : Input Data structure
DataStructure.patches = normalize_norm(DataStructure.patches);   % normalizing i/p descriptor such that a'*a = 1
while(1)
    [~, tempD] = kmeans(DataStructure.patches',K);          % kmeans codebook
    Dictionary = tempD';
    Dictionary=Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));                      % Unit length bases
    clear tempD IDX
    if(size(Dictionary,2)==K)
        break;
    end 
end
 

eucl_mat = compute_dist(DataStructure.patches',Dictionary');     % Compute Eucleadian distance between the codewords and the i/p descriptors
dist_thresh = THRESH*(sum(sum(eucl_mat))/numel(eucl_mat));       % dist_thresh is the Threshold distance for matching descriptor with codewords
flag_mat = sparse(eucl_mat < dist_thresh);                       % flag_mat is a matrix with dimension K x NoSTIPs. Each entry of the matrix a_{ij} = 1 if distance(codeword_i,descriptor_j) < dist_thresh

DataStructureVotemap = Votemap(DataStructure, Dictionary, flag_mat);      % computing spatial occurence distribution




function D = compute_dist(X,B)
nframe=size(X,1);
nbase=size(B,1);
XX = sum(X.*X, 2);
BB = sum(B.*B, 2);
D  = repmat(XX, 1, nbase)-2*X*B'+repmat(BB', nframe, 1);
D = D';