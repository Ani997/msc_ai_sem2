function [ ] = visualize_patches(index_train,index_test,wordid,n)
%visualize some patches are assigned to the same codeword

trainid = find(index_train==wordid);
testid  = find(index_test == wordid);
if (n> length(trainid) || n > length(testid))
    display('no enough pathtches in this cluster');  
else 
image_patches = cell(n*2,1);
id1 = trainid(1:n);
id2 = testid(1:n);
load('data/global/dictionary');
load ('data/local/001/sift_features.mat');
load ('data/global/image_names');
addpath image
x = features.x;
y = features.y;
figure,

for k = 1:n
    idd = id1(k);
    imageId = floor(idd/900)+1;% since the image is resized we know each one contains 900 patches 
    I = imread(strcat('image/',image_names{imageId}));
    patchID = rem(idd,900);
    xc = x(patchID)-7.5;
    yc = y(patchID)-7.5;
    Icrp = rgb2gray(imcrop(I,[xc,yc,15,15])); 
  %  subplot(3,n,k+n),subimage(Icrp);
    image_patches{k} = Icrp;
    axis tight;
    axis off;
%    label('random patches from training images');
    
end 

for k = 1:n
    idd = id2(k);
    imageId = floor(idd/900)+301;% since the image is resized we know each one contains 900 patches 
    I = imread(strcat('image/',image_names{imageId}));
    patchID = rem(idd,900);
    xc = x(patchID)-7.5;
    yc = y(patchID)-7.5;
    Icrp = rgb2gray(imcrop(I,[xc,yc,15,15]));
   % subplot(3,n,k+2*n),subimage(Icrp);
    image_patches{n+k} = Icrp;
    axis tight;
    axis off;
  %  label('random patches from test images')
    
end
n_c = floor(2*n/10)+1;
subplot(n_c,10,1:10),bar(C(wordid,:)),title('codeword histogram');
axis tight;
for k = 1:2*n
    h=subplot(n_c,10,k+10),subimage(image_patches{k});
    set(h,'XTick',[],'YTick',[]);
    axis off;
end 
end

end

