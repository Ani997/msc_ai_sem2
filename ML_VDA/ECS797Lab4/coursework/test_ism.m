function test_ism()
 
THRESH = 0.22;             % used to set the threshold for the distance between the codeword and the discriptor 
TEM_MEANSHIFT_WIDTH = 0.20; % Meanshift kernel width. This is a parameter that controls the way that local maxima are found at the voting maps.
                           % you can decrease this value to increase the number of generated hypothesis
a = 1.8; b = 3;            % a and b are the lower and upper limit for the selection of the threshold (for the distance between the codeword and the discriptor)
                           % you can change the value 1.8 <= a <= 2.4 to change the number of descriptors matched to the codeword
fea_path = 'test_fea/';    % test feature path
NO_OF_POINTS = 1;          % threshold for the number points in a hypothesis (To reject the hypothesis with no.of points < NO_OF_POINTS)
LOW_THRESH = 0;            % threshold for the confidence of a hypothesis (To reject the hypothesis with confidence < LOW_THRESH)


for i = 1:3
    load(sprintf('data/class%d/Dictionary',i));
    struct_Dictionary_array.array(i).Dictionary = Dictionary;      
    load(sprintf('data/class%d/DataStructureVotemap',i));
    DataStructureVotemap_array.array(i).DataStructureVotemap = DataStructureVotemap;
    clear DataStructureVotemap Dictionary
end

pred = []
for i = 1:3             % Test each class
    fprintf('\nTesting class %d',i);
    load(sprintf('data/test_class%d',i));  %Example: test_class1 contains the index of the features used for class 1
    sz_test_array = size(test_class,2);
    for j = 1:sz_test_array    
        fprintf('.');
        seq_no = test_class(j);
        load([fea_path,'struct_fea',int2str(seq_no)]);
        temp_array_TP_FP_mat = [];
        for k = 1:3        %  Compare the descriptors with dictionaries from all the and choose the one with the maximum response
            Dictionary = struct_Dictionary_array.array(k).Dictionary;
            DataStructureVotemap = DataStructureVotemap_array.array(k).DataStructureVotemap;
            TP_FP_mat = ism_test_voting(struct_fea,THRESH,0,TEM_MEANSHIFT_WIDTH, LOW_THRESH,NO_OF_POINTS,DataStructureVotemap,Dictionary,a,b);
            % TP_FP_mat is a matrix with dimensions equal to 2 x number of hypotheses. 
            % The first row TP_FP_mat(1, :) is the overlap with the ground truth 
            % The second row TP_FP_mat(2, :) is the number of votes for the hypotheses.
            
            if(~isempty(TP_FP_mat))
                temp_cnt = size(TP_FP_mat,2);
                temp_array_TP_FP_mat = [temp_array_TP_FP_mat [TP_FP_mat;k*ones(1,temp_cnt)]];
            end
        end
        [~,sort_ind] = sort(temp_array_TP_FP_mat(2,:),'descend');
        temp_array_TP_FP_mat = temp_array_TP_FP_mat(:,sort_ind);
        struct_TP_FP.class(i).seq(j).array = temp_array_TP_FP_mat;
        pred = [pred temp_array_TP_FP_mat(3,1)];
    end      
end
save('pred','pred')
save('struct_TP_FP','struct_TP_FP');
fprintf('\n');