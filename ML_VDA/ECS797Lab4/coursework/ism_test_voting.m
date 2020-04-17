function TP_FP_mat = ism_test_voting(struct_fea, THRESH, DISP_FLAG,TEM_MEANSHIFT_WIDTH,...
                                     LOW_THRESH, NO_OF_POINTS, DataStructureVotemap, Dictionary, a, b)
% TP_FP_mat              : is matrix with dimensions 2 x NO_OF_HYPOTHESIS
%                        : where first element of each column indicates whether the
%                        : generated hypothesis is correct or not i.e the
%                        : ratio of intersection to union of bounding box is
%                        : > 0.5. second element indicates the conficence of the detected hypothesis
 
% struct_fea             : descriptors(Data structure) for a single image sequence
% THRESH                 : used to set the threshold for the distance between the codeword and the discriptor 
% TEM_MEANSHIFT_WIDTH    : Meanshift kernel width. This is a parameter that controls the way that local maxima are found at the voting maps.
%                        : you can decrease this value to increase the number of generated hypothesis
% DISP_FLAG              : Display the generated hypothesis if true
% NO_OF_POINTS           : threshold for the number points in a hypothesis (To reject the hypothesis with no.of points < NO_OF_POINTS)
% LOW_THRESH             : threshold for the confidence of a hypothesis (To reject the hypothesis with confidence < LOW_THRESH)
% Dictionary             : Dictionary for each class
% DataStructureVotemap   : Data structure used for storing Voting maps
%a ,b                    : a and b are the lower and upper limit for the selection of the threshold (for the distance between the codeword and the discriptor)
%                        : you can change the value 1.8 <= a <= 2.4 to change the number of descriptors matched to the codeword

   
  
MIN_FRAMES = 1;          % threshold for the minimum number frames in a generated hypothesis (To reject the hypothesis with no.of frames < MIN_FRAMES)
SPA_OFFSHOOT = 20;       % spatial distance used to estimate the bounding box of the hypothesis
TMP_OFFSHOOT = 5;        % temporal distance used to estimate the bounding box of the hypothesis
patches = normalize_norm(struct_fea.features);  %normalizing i/p descriptors
location = struct_fea.location;    % spatial location of the descriptors in the sequence
n = size(struct_fea.label,2);
spa_scale = struct_fea.spa_sca;     % spatial scale at which STIP was detected 
tem_scale = struct_fea.tem_sca;     % temporal scale at which STIP was detected
frame_num = struct_fea.frame_num;   % frame number at which STIP was detected
hei_wid = [120;160];        % resolution of the video

eucl_mat = compute_dist(patches',Dictionary');  % VIJAY Remove this line 
%---------------------------------------------------------------------------------------------
% Write a code to compute eucl_mat, the Eucleadian distance between the codewords and the descriptors
% The output matrix eucl_mat should have dimensions Dsize x NoSTIPs where Dsize is the dictionary size
% and NoSTIPs is the number of STIPs in the test sequence

%----------------------------------------------------------------------------------------------


avg_distance = sum(sum(eucl_mat))/numel(eucl_mat); 
dist_thresh1 = THRESH*(avg_distance);
flag_mat = eucl_mat<dist_thresh1;

% To ensure that we have sufficient number of descriptors matched to the
% codewords (in order to have a good hough voting) % if the number
% of matching is less then increase the threshold to increase the number 
% of matching descriptors
while(1)
  if(((sum(sum(flag_mat))/n)<=b) && ((sum(sum(flag_mat))/n)>=a))
    break;
  end
  
  if((sum(sum(flag_mat))/n)>=b)
    THRESH = THRESH - 0.005;
  else
    THRESH = THRESH + 0.005;
  end
  dist_thresh1 = THRESH*(avg_distance);
  flag_mat = eucl_mat<dist_thresh1;
end

if(DISP_FLAG)
    disp('-----------------------------------------------------');
    fprintf('Matches per feature = %f\n',sum(sum(flag_mat))/n);
    disp('-----------------------------------------------------');
end

hough_array = houghvoting(patches,location,spa_scale,...
    tem_scale,frame_num,flag_mat,DataStructureVotemap);   
bb_array = hough_array(6:7,:);
hough_array = hough_array(1:5,:);
[hyp_st_fr, count] = generate_hypothesis(hough_array,TEM_MEANSHIFT_WIDTH,...
    LOW_THRESH,NO_OF_POINTS,hei_wid);  % select hypothesis for start frame
if(count==0)
   TP_FP_mat = [];
    return;
end

no_st_fr = size(hyp_st_fr,1);
hyp_tem = zeros(no_st_fr,5);
count_hyp_tem = 0;
for i = 1:no_st_fr
    if((hyp_st_fr(i,4) - hyp_st_fr(i,3)) > MIN_FRAMES)
        count_hyp_tem = count_hyp_tem + 1;
        hyp_tem(count_hyp_tem,:) = hyp_st_fr(i,:);
    end
end
hyp_tem = hyp_tem(1:count_hyp_tem,:);
hyp_spa = hyp_tem(1:count_hyp_tem,1:2);
struct_bb = bb_estimation(bb_array,hough_array,hyp_spa,hyp_tem,SPA_OFFSHOOT,TMP_OFFSHOOT);
if(DISP_FLAG)
    disp('-----------------------------------------------------');
    fprintf('Groundtruth:\n-----------------------------------------------------\n');
    fprintf('Spatial center  : %d     %d\nBounding box    : %d      %d\nVideo length    : %d      %d\n',...
        struct_fea.spa_center(1,1),struct_fea.spa_center(2,1),struct_fea.hei_wid_bb(1,1),...
        struct_fea.hei_wid_bb(2,1),struct_fea.st_end_frame(1,1),struct_fea.st_end_frame(2,1));
    disp('-----------------------------------------------------');
    fprintf('Hypothesis:\n-----------------------------------------------------\n');
    for iii = 1:struct_bb.cnt
        fprintf('Spatial center  : %d     %d\nBounding box    : %d      %d\n',...
            int32(struct_bb.array_center(1,iii)),int32(struct_bb.array_center(2,iii)),...
            int32(struct_bb.array_bb(1,iii)),int32(struct_bb.array_bb(2,iii)));
        disp('-----------------------------------------------------');
    end
    for iii = 1:struct_bb.cnt
        fprintf('Video length    : %d      %d\n',int32(struct_bb.array_tem(1,iii)),...
            int32(struct_bb.array_tem(2,iii)));
    end
end
count_tot = 0;
for i = 1:struct_bb.cnt
    count_tot = count_tot + 1;
    struct_spa_tem.array_spa(:,count_tot) = struct_bb.array_bb(:,i);
    struct_spa_tem.array_center(:,count_tot) = struct_bb.array_center(:,i);
    struct_spa_tem.array_frame(:,count_tot) = struct_bb.array_tem(:,i);
    struct_spa_tem.array_val(:,count_tot) = struct_bb.array_val(:,i);
end

gt_struct_spa_tem.array_spa = struct_fea.hei_wid_bb(:,1);
gt_struct_spa_tem.array_center = struct_fea.spa_center(:,1);
gt_struct_spa_tem.array_frame = struct_fea.st_end_frame(:,1);
TP_FP_mat = zeros(2,count_tot);
for i = 1:count_tot
    TP = verification(gt_struct_spa_tem,struct_spa_tem,i);
    TP_FP_mat(:,i) = [TP;struct_spa_tem.array_val(:,i)];
end
