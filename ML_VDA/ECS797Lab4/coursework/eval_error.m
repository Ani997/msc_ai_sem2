function er = eval_error()
% calculate the misclassification error which is
% FP+FN/Total
load('struct_TP_FP');
count = 0;
num_miss = 0;
for i = 1:3
    class_size = size(struct_TP_FP.class(i).seq,2);   
    for j = 1:class_size
        % from reall_prec_curv.m, get temp_ind_pos and temp_class
        temp_ind_pos = ((struct_TP_FP.class(i).seq(j).array(1,:)==1) & (struct_TP_FP.class(i).seq(j).array(3,:)==i));
        temp_class = struct_TP_FP.class(i).seq(j).array(3,temp_ind_pos);
        % FP found
        if cumsum(temp_ind_pos) == 0
            num_miss = num_miss + 1;
        % FN 
        elseif temp_class(1) ~= i 
            num_miss = num_miss + 1;
        end
        count = count + 1;
    end
end
er = num_miss/count;