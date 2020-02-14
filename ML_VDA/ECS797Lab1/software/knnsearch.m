function [idx,D]=knnsearch(varargin)


% KNNSEARCH   Linear k-nearest neighbor (KNN) search
% IDX = knnsearch(Q,R,K) searches the reference data set R (n x d array
% representing n points in a d-dimensional space) to find the k-nearest
% neighbors of each query point represented by eahc row of Q (m x d array).
% The results are stored in the (m x K) index array, IDX. 
%
% IDX = knnsearch(Q,R) takes the default value K=1.
%
% IDX = knnsearch(Q) or IDX = knnsearch(Q,[],K) does the search for R = Q.
%
% Rationality
% Linear KNN search is the simplest appraoch of KNN. The search is based on
% calculation of all distances. Therefore, it is normally believed only
% suitable for small data sets. However, other advanced approaches, such as
% kd-tree and delaunary become inefficient when d is large comparing to the
% number of data points. On the other hand, the linear search in MATLAB is
% relatively insensitive to d due to the vectorization. In  this code, the 
% efficiency of linear search is further improved by using the JIT
% aceeleration of MATLAB. Numerical example shows that its performance is
% comparable with kd-tree algorithm in mex.

% Check inputs
[Q,R,K,method] = parseinputs(varargin{:});

% Check outputs
error(nargoutchk(0,2,nargout));
assert(method==1 || method ==2 || method ==3);

[N,M] = size(Q);
L=size(R,1);
idx = zeros(N,K);

%----------------------------for loop implementaion------------------------


switch method 
    case 1
       for k1 = 1:N
          for k2 = 1:L
           dis(k1,k2) = EuclideanDistance(Q(k1,:),R(k2,:));
          end
        end 
   if K == 1
        [mv,idx] = min(dis,[],2);
   else 
       [sdist,indall] = sort(dis,2);% Sort array elements in each row in ascending order
       idx = indall(:,1:K);
   end  
    case 2
        for k1 = 1:N
            for k2 = 1:L
        	disk(k1,k2) = histogram_intersection(Q(k1,:),R(k2,:));% one-to-one distance 
            end 
        end
    if K == 1
        [mv,idx] = min(disk,[],2);
    else 
        [sdist,indall] = sort(disk,2);
        idx = indall(:,1:K);
    end 
    otherwise 
        fprintf('Please set method as 1 or 2! \n');
end 


%--------------------------for-loop implementation-------------------------


function d=histogram_intersection(a,b)
  
  p=size(a,2); % dimension of samples
  
  assert(p == size(b,2)); % equal dimensions
  assert(size(a,1) == 1); % a needs to be a single sample
  assert(size(b,1) == 1); % b needs to be a single sample
	
  d=zeros(m,1); % initialize output array


  d = 0;  
  % -------------- write your own code here ---------------
  d = 1;
  
  for i = 1:a
    d = d - min(b(1,index),c(1,index));      
  end
  % -------------- write your own code here ---------------

 end 


function [Q,R,K,method] = parseinputs(varargin)
% Check input and output
error(nargchk(1,4,nargin));

Q=varargin{1};

if nargin<2
    R=Q;
    fident = true;
else
    fident = false;
    R=varargin{2};
end

if isempty(R)
    fident = true;
    R=Q;
end

if nargin<3
    K=1;
else
    K=varargin{3};
end
if nargin<4 
    method = 1;
else 
    method = varargin{4};
end 
end 

end 
