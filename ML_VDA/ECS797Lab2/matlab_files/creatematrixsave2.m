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
% Display of the 20 frist eigenfaces :
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



%Display of first 6 recognitions:
figure;
x=6;
y=2;
for i=1:6,
    
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
for  i=1:70,
    number_of_test_images(1,ident(1,i))= number_of_test_images(1,ident(1,i))+1;
end,
personn_recognised=zeros(1,40);
recognitionrate=zeros(1,5);
number_per_number=zeros(1,5);


i=1;
while (i<70),
    id=ident(1,i);   
    distmin=Values(id,1);
        indicemin=Indices(id,1);
    while (i<70)&&(ident(1,i)==id), 
        if (Values(i,1)<distmin),
            distmin=Values(i,1);
        indicemin=Indices(i,1);
        end,
        i=i+1;
    
    end,
    personn_recognised(1,id)=indicemin;
    number_per_number(number_of_test_images(1,id))=number_per_number(number_of_test_images(1,id))+1;
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;
        
    end,
end,

for  i=1:5,
   recognitionrate(1,i)=recognitionrate(1,i)/number_per_number(1,i);
end,
   figure;
    plot (recognitionrate(1,:));
    
    
%effect of threshold:   
averageRR=zeros(1,20);
for t=1:20,
  Threshold =t;  
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

personn_recognised=zeros(1,40);
recognitionrate=zeros(1,5);
number_per_number=zeros(1,5);

i=1;
while (i<70),
    id=ident(1,i);   
    distmin=Values(id,1);
        indicemin=Indices(id,1);
    while (i<70)&&(ident(1,i)==id), 
        if (Values(i,1)<distmin),
            distmin=Values(i,1);
        indicemin=Indices(i,1);
        end,
        i=i+1;
    
    end,
    personn_recognised(1,id)=indicemin;
    number_per_number(number_of_test_images(1,id))=number_per_number(number_of_test_images(1,id))+1;
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;
        
    end,
end,

for  i=1:5,
   recognitionrate(1,i)=recognitionrate(1,i)/number_per_number(1,i);
end,
averageRR(1,t)=mean(recognitionrate(1,:));
end,
figure;
plot(averageRR(1,:));


%effect of K:
averageRR=zeros(1,20);
  Threshold =15;  
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


person=zeros(70,200);
person(:,:)=floor((Indices(:,:)-1)/5)+1;

for K=3:3s
personn_recognised=zeros(1,70);
recognitionrate=0;
number_per_number=zeros(1,5);
number_of_occurance=zeros(70,K);

for i=1:70;
    max=0;
  
    for j=1:K,
        for k=j:K,
            if (person(i,k)==person(i,j))
                number_of_occurance(i,j)=number_of_occurance(i,j)+1;
            end,
        end,
        if (number_of_occurance(i,j)>max)
            max=number_of_occurance(i,j);
            jmax=j;
        end,
    end,
    personn_recognised(1,i)=person(i,jmax);
  
 if (ident(1,i)==personn_recognised(1,i))
     recognitionrate=recognitionrate+1;
 end,

averageRR(1,K)=recognitionrate/70;
end,
end,
figure;
plot(averageRR(1,:));



%figure;
%for i=1:70,

%      Image = uint8 (zeros(28, 23));
 %     for k = 0:643
 %    Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
 %     end,
 %  subplot (7,10,i);
 %   imshow (Image);
%end,
