function count_TP = verification(gt_struct_spa_tem,struct_spa_tem,index)
% A hypothesis is considered true if the ratio of intersection to union of the hypothesis
% and groundtruth is > 0.5. 
% count_TP = 1 if the the ratio of intersection to union of the hypothesis and groundtruth is > 0.5
count_TP = 0;
i = index;
gt_vol_hei = floor([gt_struct_spa_tem.array_center(1,1) - gt_struct_spa_tem.array_spa(1,1);...
    gt_struct_spa_tem.array_center(1,1) + gt_struct_spa_tem.array_spa(1,1)]);
gt_vol_wid = floor([gt_struct_spa_tem.array_center(2,1) - gt_struct_spa_tem.array_spa(2,1);...
    gt_struct_spa_tem.array_center(2,1) + gt_struct_spa_tem.array_spa(2,1)]);
gt_vol_frame = floor(gt_struct_spa_tem.array_frame);

hyp_vol_hei = floor([struct_spa_tem.array_center(1,i) - struct_spa_tem.array_spa(1,i);...
    struct_spa_tem.array_center(1,i) + struct_spa_tem.array_spa(1,i)]);
hyp_vol_wid = floor([struct_spa_tem.array_center(2,i) - struct_spa_tem.array_spa(2,i);...
    struct_spa_tem.array_center(2,i) + struct_spa_tem.array_spa(2,i)]);
hyp_vol_frame = floor(struct_spa_tem.array_frame(:,i));

if(gt_vol_hei(1,1) <= hyp_vol_hei(1,1))
    if(gt_vol_hei(1,1) < 1)
        temp = -gt_vol_hei(1,1) + 1;
        gt_vol_hei = gt_vol_hei + temp*ones(2,1);
        hyp_vol_hei(:,1) = hyp_vol_hei(:,1) + temp*ones(2,1);
    else
        temp = gt_vol_hei(1,1) - 1;
        gt_vol_hei = gt_vol_hei - temp*ones(2,1);
        hyp_vol_hei(:,1) = hyp_vol_hei(:,1) - temp*ones(2,1);
    end
else
    if(hyp_vol_hei(1,1) < 1)
        temp = -hyp_vol_hei(1,1) + 1;
        gt_vol_hei = gt_vol_hei + temp*ones(2,1);
        hyp_vol_hei(:,1) = hyp_vol_hei(:,1) + temp*ones(2,1);
    else
        temp = hyp_vol_hei(1,1) - 1;
        gt_vol_hei = gt_vol_hei - temp*ones(2,1);
        hyp_vol_hei(:,1) = hyp_vol_hei(:,1) - temp*ones(2,1);
    end
end

if(gt_vol_wid(1,1) <= hyp_vol_wid(1,1))
    if(gt_vol_wid(1,1) < 1)
        temp = -gt_vol_wid(1,1) + 1;
        gt_vol_wid = gt_vol_wid + temp*ones(2,1);
        hyp_vol_wid(:,1) = hyp_vol_wid(:,1) + temp*ones(2,1);
    else
        temp = gt_vol_wid(1,1) - 1;
        gt_vol_wid = gt_vol_wid - temp*ones(2,1);
        hyp_vol_wid(:,1) = hyp_vol_wid(:,1) - temp*ones(2,1);
    end
else
    if(hyp_vol_wid(1,1) < 1)
        temp = -hyp_vol_wid(1,1) + 1;
        gt_vol_wid = gt_vol_wid + temp*ones(2,1);
        hyp_vol_wid(:,1) = hyp_vol_wid(:,1) + temp*ones(2,1);
    else
        temp = hyp_vol_wid(1,1) - 1;
        gt_vol_wid = gt_vol_wid - temp*ones(2,1);
        hyp_vol_wid(:,1) = hyp_vol_wid(:,1) - temp*ones(2,1);
    end
end

if(gt_vol_frame(1,1) <= hyp_vol_frame(1,1))
    temp = gt_vol_frame(1,1) - 1;
    gt_vol_frame = gt_vol_frame - temp*ones(2,1);
    hyp_vol_frame(:,1) = hyp_vol_frame(:,1) - temp*ones(2,1);
else
    if(hyp_vol_frame(1,1) < 1)
        temp = -hyp_vol_frame(1,1) + 1;
        gt_vol_frame = gt_vol_frame + temp*ones(2,1);
        hyp_vol_frame(:,1) = hyp_vol_frame(:,1) + temp*ones(2,1);
    else
        temp = hyp_vol_frame(1,1) - 1;
        gt_vol_frame = gt_vol_frame - temp*ones(2,1);
        hyp_vol_frame(:,1) = hyp_vol_frame(:,1) - temp*ones(2,1);
    end
end

if(gt_vol_hei(2,1) >= hyp_vol_hei(2,1))
    hei_end = gt_vol_hei(2,1);
else
    hei_end = hyp_vol_hei(2,1);
end

if(gt_vol_wid(2,1) >= hyp_vol_wid(2,1))
    wid_end = gt_vol_wid(2,1);
else
    wid_end = hyp_vol_wid(2,1);
end

if(gt_vol_frame(2,1) >= hyp_vol_frame(2,1))
    tem_end = gt_vol_frame(2,1);
else
    tem_end = hyp_vol_frame(2,1);
end


ori_vol = zeros(hei_end,wid_end,tem_end);
gt_vol = ori_vol;
hyp_vol = ori_vol;
gt_vol(gt_vol_hei(1,1):gt_vol_hei(2,1),gt_vol_wid(1,1):gt_vol_wid(2,1),...
    gt_vol_frame(1,1):gt_vol_frame(2,1)) = 1;
ori_vol = ori_vol + gt_vol;
hyp_vol(hyp_vol_hei(1,1):hyp_vol_hei(2,1),hyp_vol_wid(1,1):hyp_vol_wid(2,1),...
    hyp_vol_frame(1,1):hyp_vol_frame(2,1)) = 1;
ori_vol = ori_vol + hyp_vol;
% ratio = sum(s1(ori_vol==2))/(sum(s1(ori_vol==2))+sum(s1(ori_vol==1)));
ratio = sum(sum(sum((ori_vol==2))))/(sum(sum(sum((ori_vol==2))))+sum(sum(sum((ori_vol==1)))))

% disp(ratio);
if(ratio > (1/2))
    count_TP = count_TP + 1;
end