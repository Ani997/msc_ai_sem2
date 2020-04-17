function ism_test()
THRESH = 0.22;  
width = 0.2;  % Meanshift kernel width
a = 2.2; b = 3;
fea_path = 'test_fea/';  %test feature path
NO_OF_POINTS = 1;     
LOW_THRESH = 0;

 
for i = 1:3
    load(sprintf('data/class%d/D',i));
    struct_D_array.array(i).D = D;
    load(sprintf('data/class%d/struct_cb',i));
    struct_cb_array.array(i).struct_cb = struct_cb;
    clear struct_cb D
end


for i = 1:3             % Test each class
    fprintf('\nTesting class %d',i);
    load(sprintf('data/test_class%d',i));
    sz_test_array = size(test_class,2);
    for j = 1:sz_test_array    
        fprintf('.');
        seq_no = test_class(j);
        load([fea_path,'struct_fea',int2str(seq_no)]);
        temp_array_TP_FP_mat = [];
        for k = 1:3        %  Compare the descriptors with dictionaries from all the and choose the one with the maximum response
            D = struct_D_array.array(k).D;
            struct_cb = struct_cb_array.array(k).struct_cb;
            TP_FP_mat = ism_test_voting(struct_fea,THRESH,0,width,LOW_THRESH,NO_OF_POINTS,struct_cb,D,a,b);
            if(~isempty(TP_FP_mat))
                temp_cnt = size(TP_FP_mat,2);
                temp_array_TP_FP_mat = [temp_array_TP_FP_mat [TP_FP_mat;k*ones(1,temp_cnt)]];
            end
        end
        [~,sort_ind] = sort(temp_array_TP_FP_mat(2,:),'descend');
        temp_array_TP_FP_mat = temp_array_TP_FP_mat(:,sort_ind);
        struct_TP_FP.class(i).seq(j).array = temp_array_TP_FP_mat;
    end      
end
save('struct_TP_FP','struct_TP_FP');
fprintf('\n');