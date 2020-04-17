function hough_array = houghvoting(patches,position,spa_scale,tem_scale,frame_num,flag_mat,struct_cb)
% hough_array           : is a matrix with 7 rows, each column indicating the
%                       : predicted(votes) spatial location, predicted (voted) start and end frames,
%                       : the votes, bounding box values(scale compensated). Example: Let a descriptor matched with
%                       : the codeword (with one offset) cast votes at the spatial location [x,y],
%                       : temporal location [s,e] (predictions for start and end frames), bounding box values [b1 b2] (stored during training)
%                       : with value(weight) 'v', then corresponding column in the
%                       : matrix hough_array will be [x y s e v b1 b2]'.
% patches               : i/p descriptors
% position              : spatial location of the detected STIP                   
% spa_scale             : spatial scale                      
% tem_scale             : temporal scale
% frame_num             : frame number at which STIP was detected
% flag_mat              : refer to ism_test_voting.m

[X, Y, S, E, V, B1, B2] = deal([]); % init  
% dict_size = size(flag_mat, 1); 
for p=1:size(patches,2)
    for i=1:size(flag_mat, 1)
        sum_act_cc = sum(flag_mat(i,:)); 
        if sum_act_cc ~= 0
            % if interest point exists
            num_of_edges = struct_cb.offset(i).tot_cnt; % patches
            act_pos_s = position(:, flag_mat(i,:)); % for X and Y, spatial 
            act_frame_t = frame_num(:, flag_mat(i,:)); % start and end, temporal       
            V_prob = [];
            for j=1:num_of_edges
                % upadte the weights
                X = [X act_pos_s(1,:) - spa_scale(1, flag_mat(i,:))  * struct_cb.offset(i).spa_offset(1, j)];
                Y = [Y act_pos_s(2,:) - spa_scale(1, flag_mat(i,:))  * struct_cb.offset(i).spa_offset(2, j)];       
                S = [S act_frame_t - tem_scale(1, flag_mat(i,:))  * struct_cb.offset(i).st_end_offset(1, j)];
                E = [E act_frame_t - tem_scale(1, flag_mat(i,:))  * struct_cb.offset(i).st_end_offset(2, j)];            
                B1 = [B1 struct_cb.offset(i).hei_wid_bb(1, j) * spa_scale(1, flag_mat(i,:)) ];
                B2 = [B2 struct_cb.offset(i).hei_wid_bb(2, j) * spa_scale(1, flag_mat(i,:)) ];
                V_prob = [V_prob repmat(1/(num_of_edges), sum_act_cc, 1)']; % voting       
            end
            V = [V V_prob];
        end
    end
end
hough_array = [X; Y; S; E; V; B1; B2];
