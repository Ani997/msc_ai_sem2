function struct_bb = bb_estimation(bb,hough_array,hyp_center,hyp_tem,SPA_OFFSHOOT,TMP_OFFSHOOT)
% hyp_center      : hypothesis (spatial center) 
% hyp_tem         : hypothesis (temporal OR START AND END FRAME)
% hough_array     : refer houghvoting.m
% bb              : refer ism_test_voting.m
% SPA_OFFSHOOT    : refer ism_test_voting.m
% TMP_OFFSHOOT    : refer ism_test_voting.m
  
%struct_bb is a structure with fields
%
%         array_bb: [2xNoHYPS double]    ------> estimated bounding box
%              cnt: NoHYPS               ------> No. of hypothesis
%     array_center: [2xNoHYPS double]    ------> same as hyp_center
%        array_tem: [2xNoHYPS double]    ------> same as hyp_tem
%        array_val: [1xNoHYPS double]    ------> hypothesis confidence


[no_hyp,~] = size(hyp_center);   % Number of hypothesis
struct_bb.array_bb = zeros(2,no_hyp);
count_correct_hyp = 0;  % count for correct hypothesis 
struct_bb.cnt = 0;
for hyp_ind = 1:no_hyp
    hyp_cen_a = hyp_center(hyp_ind,1) - SPA_OFFSHOOT;
    hyp_cen_b = hyp_center(hyp_ind,1) + SPA_OFFSHOOT;
    hyp_cen_c = hyp_center(hyp_ind,2) - SPA_OFFSHOOT;
    hyp_cen_d = hyp_center(hyp_ind,2) + SPA_OFFSHOOT;
    hyp_tem_a = hyp_tem(hyp_ind,3) - TMP_OFFSHOOT;
    hyp_tem_b = hyp_tem(hyp_ind,3) + TMP_OFFSHOOT;
    hyp_tem_c = hyp_tem(hyp_ind,4) - TMP_OFFSHOOT;
    hyp_tem_d = hyp_tem(hyp_ind,4) + TMP_OFFSHOOT;
    flag = (hough_array(1,:)>hyp_cen_a & hough_array(1,:)<hyp_cen_b & hough_array(2,:)>hyp_cen_c...
    & hough_array(2,:)<hyp_cen_d);  
    flag_tem = (hough_array(3,:)>hyp_tem_a & hough_array(3,:)<hyp_tem_b & hough_array(4,:)>hyp_tem_c ...
        & hough_array(4,:)<hyp_tem_d);
    comb_flag = flag & flag_tem;
    index_bb_ptr = sum(comb_flag);  
    sum_bb_accum = sum(bb(:,comb_flag),2);
    if(index_bb_ptr>0)
        count_correct_hyp = count_correct_hyp + 1;
        struct_bb.array_bb(:,count_correct_hyp) = floor(sum_bb_accum/index_bb_ptr);
        struct_bb.array_center(:,count_correct_hyp) = hyp_center(hyp_ind,:)';
        struct_bb.array_tem(:,count_correct_hyp) = hyp_tem(hyp_ind,3:4)';
        struct_bb.array_val(:,count_correct_hyp) = hyp_tem(hyp_ind,5);
    end
end
struct_bb.cnt = count_correct_hyp;