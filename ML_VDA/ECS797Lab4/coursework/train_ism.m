function train_ism()
codebook_sz = [550 500 650];  % codebook size of class 1,2 & 3 respectively. 
                                % You can change the size of the codebook here
% codebook_sz = [10 15 5]; 
for class_ind = 1:3
    dir_nam = ['class', int2str(class_ind)];
    THRESH = 0.24;            % Variable that helps sets the threshold used for matching codebook with descriptors
    fprintf('\n Training class %d', class_ind);
    calling_ism(THRESH, class_ind, dir_nam, codebook_sz(class_ind));  % training ism
end
fprintf('\n'); 