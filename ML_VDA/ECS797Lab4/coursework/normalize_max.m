function [ret_array, range, min_array, n] = normalize_max(array)
[m, n] = size(array);
max_array = max(array);
min_array = min(array);
array = array - repmat(min_array,m,1);
range = max_array - min_array;
temp_array = range==0;
range = range + 0.000000000001*temp_array;
ret_array = array./repmat(range,m,1);
 