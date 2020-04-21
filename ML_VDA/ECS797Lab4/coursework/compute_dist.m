function d = compute_dist(b,a)

aa=sum(a.*a,2); bb=sum(b.*b,2); ab=a*b'; 
d = sqrt(abs(repmat(aa,[1 size(bb,1)]) + repmat(bb',[size(aa,1) 1]) - 2*ab));
