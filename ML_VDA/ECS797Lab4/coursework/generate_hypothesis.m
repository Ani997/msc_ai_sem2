function [generated_hyp,count] = generate_hypothesis(hough_array,WIDTH,LOW_THRESH,NO_OF_POINTS,hei_wid)
% generated_hyp          : generated hypothesis
% hough_array            : hough_array (refer ism_test_voting.m)
% WIDTH                  : refer ism_test_voting.m
% LOW_THRESH             : refer ism_test_voting.m
% NO_OF_POINTS           : refer ism_test_voting.m
% hei_wid                : refer ism_test_voting.m
[m,~] = size(hough_array); 
array = hough_array(1:m-1,:); 
[norm_array,range, min_array, no_elem] = normalize_max(array');
[hyp_means,clust_siz,score] = MeanShiftCluster_modified([norm_array';hough_array(m,:)],WIDTH);
sz_mean = size(hyp_means,2);
if(sz_mean>0)
    hyp_means = floor((hyp_means'.* repmat(range,sz_mean,1)) + repmat(min_array,sz_mean,1));
end
generated_hyp = zeros(sz_mean,m);
count = 0;
for i = 1:sz_mean
    temp_cnt = clust_siz(i);
    if(temp_cnt>NO_OF_POINTS && hyp_means(i,1)>1 && hyp_means(i,2)>1 && hyp_means(i,1) < (hei_wid(1,1)-1) && hyp_means(i,2) < (hei_wid(2,1)-1))
        if(score(1,i)>LOW_THRESH)
            count = count + 1;
            generated_hyp(count,:) = [hyp_means(i,:) score(1,i)];
        end
    end
end
generated_hyp = generated_hyp(1:count,:);