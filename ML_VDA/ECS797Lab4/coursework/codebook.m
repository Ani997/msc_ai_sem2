function struct_cb = codebook(mystruct,D,params)
cb_size = size(D,2);
spa_offset = mystruct.spa_offset;   % Spatial offset % offset = [x y] - [x_c y_c]
spa_scale = mystruct.spa_scale;       
tem_scale = mystruct.tem_scale;    
st_end_offset = mystruct.st_end_offset;  
hei_wid_bb = mystruct.hei_wid_bb;
struct_cb = [];
flag_mat = params.flag_mat; 
tot_patches = sum(flag_mat,2);     
for i = 1:cb_size
    struct_cb.offset(i).tot_cnt = tot_patches(i);   % Count of the total number of patches(both +ve and -ve) for a codeword
    if(tot_patches(i)>0)
        struct_cb.offset(i).spa_offset = spa_offset(:,flag_mat(i,:))./[spa_scale(:,flag_mat(i,:));spa_scale(:,flag_mat(i,:))]; 
        struct_cb.offset(i).st_end_offset = st_end_offset(:,flag_mat(i,:))./[tem_scale(:,flag_mat(i,:));tem_scale(:,flag_mat(i,:))];
        struct_cb.offset(i).spa_scale = spa_scale(:,flag_mat(i,:));
        struct_cb.offset(i).temp_scale = tem_scale(:,flag_mat(i,:));
        struct_cb.offset(i).hei_wid_bb = hei_wid_bb(:,flag_mat(i,:))./[spa_scale(:,flag_mat(i,:));spa_scale(:,flag_mat(i,:))];
    end
end 
