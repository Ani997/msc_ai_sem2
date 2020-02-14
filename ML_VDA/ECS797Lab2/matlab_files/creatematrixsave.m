addpath /homes/clt2/Matlab/;
Imagestrain = loadImagesInDirectory ( '/homes/clt2/Matlab/training-set/23x28/');
[Imagestest, ident] = loadTestImagesInDirectory ( '/homes/clt2/Matlab/testing-set/23x28/');
[Means, Space, Eigenvalues, Space2, Eigenvalues2,CovarianceMatrix]= buildSpace (Imagestrain);
x=4;
y=5;

%Image=imread('/homes/clt2/Matlab/os40.0010.jpg');
%Image2=imread('/homes/clt2/Matlab/training-set/23x28/os40.0001.rasN23');
%Image2name='/homes/clt2/Matlab/training-set/23x28/os40.0001.rasN23';

MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);
 
end


figure;
subplot (1, 1, 1);
imshow(MeanImage);
title('Mean Image');



%readras(os37.0001);
%imshow (os37.0001, rasN23);

figure;
Eigenface = uint8 (zeros(28, 23));
for i = 1:20
    for k = 0:643
   Eigenface( mod (k,28)+1, floor(k/28)+1 ) = (Space (i,k+1)+0.2)*(255/0.4);
    end
 subplot (x,y,i);
 imshow(Eigenface);
 title([ num2str(i),'th Eigenface']);
end

%Imagesmodel  =Imagestrain(1:5:196,:);

%approximateImage(Image2name, Means, Space, 90);

Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);
%Locationsmodel=projectImages (Imagesmodel, Means, Space);
Threshold =15;

TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));

for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70,
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,
figure;

x=5;
y=2;
for i=1:5,
    
      Image = uint8 (zeros(28, 23));
      for k = 0:643
     Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
      end,
   subplot (x,y,2*i-1);
    imshow (Image);
    title('Image tested');
    
    Imagerec = uint8 (zeros(28, 23));
      for k = 0:643
     Imagerec( mod (k,28)+1, floor(k/28)+1 ) = Imagestrain ((Indices(i,1)),k+1);
      end,
     subplot (x,y,2*i);
imshow (Imagerec);
title(['Image recognised with ', num2str(Threshold), ' eigenfaces:',num2str((Indices(i,1))) ]);
end,

%recognition rate compared to the number of test images:
number_of_test_images=zeros(1,40);
personn_recognised=zeros(1,40);
recognitionrate=zeros(1,5);
for  i=1:70,
    number_of_test_images(1,ident(1,i))= number_of_test_images(1,ident(1,i))+1;
end,

i=1;
while (i<71),
    id=ident(1,i);   
    distmin=Values(id,1);
        indicemin=Indices(id,1);
    while (i<71)&&(ident(1,i)==id), 
        if (Values(i,1)<distmin),
            distmin=Values(i,1);
        indicemin=Indices(i,1);
        end,
        i=i+1;
    end,
    personn_recognised(1,id)=indicemin;
    
end,
    
    

%figure;
%for i=1:70,

%      Image = uint8 (zeros(28, 23));
 %     for k = 0:643
 %    Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
 %     end,
 %  subplot (7,10,i);
 %   imshow (Image);
%end,
