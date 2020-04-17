function mat = normalize_norm(mat)
[m, n] = size(mat);
for i = 1:n
    mat(:,i) = mat(:,i)./sqrt(mat(:,i)'*mat(:,i));
end 