function recall_prec_curve()
load('struct_TP_FP');
for i = 1:3 
    cl_sz = size(struct_TP_FP.class(i).seq,2);   
    arr_tp_fp = [];
    for j = 1:cl_sz
        temp_ind_pos = ((struct_TP_FP.class(i).seq(j).array(1,:)==1) & (struct_TP_FP.class(i).seq(j).array(3,:)==i));
        cumsum_temp_ind_pos = cumsum(temp_ind_pos);
        temp_array = 1:length(cumsum_temp_ind_pos);
        temp_ind = temp_array(cumsum_temp_ind_pos==1);
        if(~isempty(temp_ind))
            arr_tp_fp = [arr_tp_fp [struct_TP_FP.class(i).seq(j).array(2,temp_ind(1));1;0]];
        end
        temp_ind_neg = (struct_TP_FP.class(i).seq(j).array(3,:)~=i) + ((struct_TP_FP.class(i).seq(j).array(1,:)==0) & (struct_TP_FP.class(i).seq(j).array(3,:)==i));
        log_temp_ind_neg = temp_ind_neg>0;
        if(~isempty(sum(log_temp_ind_neg)))
            arr_tp_fp = [arr_tp_fp [struct_TP_FP.class(i).seq(j).array(2,log_temp_ind_neg);zeros(1,sum(log_temp_ind_neg));ones(1,sum(log_temp_ind_neg))]];
        end
    end
    confidence = arr_tp_fp(1,:);
    tp = arr_tp_fp(2,:);
    fp = arr_tp_fp(3,:);
    [~,si]=sort(-confidence);
    tp = tp(si);
    fp = fp(si);
    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/cl_sz;
    prec=tp./(fp+tp);
    TP_FP_mat.array(i).rec = rec;
    TP_FP_mat.array(i).prec = prec;
end
for i = 1:3
    subplot(1,3,i);
    plot(TP_FP_mat.array(i).rec,TP_FP_mat.array(i).prec,'-c','LineWidth',2,'Marker','diamond');
    axis([0 1 0 1]);
    set(gca,'YTick',[0:0.2:1]);
    set(gca,'YTickLabel',char('0','0.2','0.4','0.6','0.8','1'));
    set(gca,'XTick',[0:0.2:1]);
    set(gca,'XTickLabel',char('0','0.2','0.4','0.6','0.8','1'));
    grid on;
    xlabel 'recall'
    ylabel 'precision'
end
f2 = figure(1);
set(f2,'Position',[10 300 1900 530]);