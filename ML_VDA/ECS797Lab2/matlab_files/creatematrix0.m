addpath /homes/clt2/Matlab/;
Images = loadImagesInDirectory ( '/homes/clt2/Matlab/training-set/23x28/');
[MeansA, Space, EigenvaluesA, Space2, Eigenvalues2]= buildSpace (Images);
x=4;
y=5;

%Image=imread('/homes/clt2/Matlab/os40.0010.jpg');
Image2=imread('/homes/clt2/Matlab/training-set/23x28/os40.0001.rasN23');

MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = MeansA (1,k+1);
 
end

%CovarianceMatrix = zeros(644, 644);


figure;
subplot (x, y, 1);
%imshow(Image2);

%subplot (x, y, 2);
imshow(MeanImage);
%readras(os37.0001);
%imshow (os37.0001, rasN23);

