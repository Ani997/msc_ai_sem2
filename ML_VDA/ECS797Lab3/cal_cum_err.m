function [cum_err] = cal_cum_err(ypre, ytest)
    sorted_ae = sort(abs(ypre-ytest));
    cum_err = zeros(ceil(sorted_ae(end)), 1);
    cum_idx = 1;
    for ae_idx = 1:size(sorted_ae, 1)
        while sorted_ae(ae_idx) > cum_idx
            cum_idx = cum_idx + 1;
            cum_err(cum_idx) = cum_err(cum_idx) + cum_err(cum_idx-1);
        end
        cum_err(cum_idx) = cum_err(cum_idx) + 1;
    end
    cum_err = cum_err/size(ytest, 1);
end