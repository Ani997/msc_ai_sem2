%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% coursework: face recognition with eigenfaces


% need to replace with your own path
% addpath software;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loading of the images: You need to replace the directory 
clear;
Imagestrain = loadImagesInDirectory ( 'images/training-set/23x28/');
[Imagestest, Identity] = loadTestImagesInDirectory ( 'images/testing-set/23x28/');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of the mean, the eigenvalues, amd the eigenfaces stored in the
%facespace:
ImagestrainSizes = size(Imagestrain);
Means = floor(mean(Imagestrain));
CenteredVectors = (Imagestrain - repmat(Means, ImagestrainSizes(1), 1));

CovarianceMatrix = cov(CenteredVectors);

[U, S, V] = svd(CenteredVectors);
Space = V(: , 1 : ImagestrainSizes(1))';
Eigenvalues = diag(S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display of the mean image:
MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);
 
end
figure;
subplot (1, 1, 1);
imshow(MeanImage);
title('Mean Image');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display of the 20 first eigenfaces : Write your code here

EigenFace = zeros(1, 644); % create a flat col matrix with 0s
NoOfEigenFaces = 20;
EFMatrix = S*V'; % using SVD V compute EF matrix


%normalization eigenfaces for better visualisation
EFMatrix = 255 *(EFMatrix - min(EFMatrix(:))) ./ (max(EFMatrix(:)) - min(EFMatrix(:)));


for k = 1:NoOfEigenFaces
   EigenFace = EFMatrix(k,:);
   EigenFace = reshape(EigenFace, [28,23]);
   subplot (4,5,k);
   imshow(uint8(EigenFace));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Projection of the two sets of images onto the face space:
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);

Threshold =20;

TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));
%Distances contains for each test image, the distance to every train image.

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display of first 6 recognition results, image per image:
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% recognition rate compared to the number of test images: Write your code here to compute the recognition rate using top 20 eigenfaces.

% RR is for 20 indices of eignefaces.
% if traon indices are not equal to test Identity the RR = 0

RR = zeros(1,length(Imagestest(:,1)));
for i = 1: length(Imagestest(:,1))
    if ceil(Indices(i,1)/5) == Identity(i)
        RR(i) = 1;
    else 
        RR(i) = 0;
    end
end
% The total recognition rate for the whole test set that has 70 images
TotalRR = sum(RR)/70 *100;
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% effect of threshold (i.e. number of eigenfaces):   
averageRR=zeros(1,20);
for t=1:40,
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
number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here
for i=1:70,
number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,

recognised_person=zeros(1,40);
recognitionrate=zeros(1,5);
number_per_number=zeros(1,5);


i=1;
while (i<70),
    id=Identity(1,i);   
    distmin=Values(id,1);
        indicemin=Indices(id,1);
    while (i<70)&&(Identity(1,i)==id), 
        if (Values(i,1)<distmin),
            distmin=Values(i,1);
        indicemin=Indices(i,1);
        end,
        i=i+1;
    
    end,
    recognised_person(1,id)=indicemin;
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
title('Recognition rate against the number of eigenfaces used');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% effect of K: You need to evaluate the effect of K in KNN and plot the recognition rate against K. Use 20 eigenfaces here.

train_lab = [];
for i = 1:40
    train_lab = horzcat(train_lab, repmat(i,1,5));
end
%% 
% can set the value of K (nearest neighbours)
K=1:200;
rec_rate = zeros(1,200);

% the identity of the test image was found using KNN
% The obtained identity was evaluated.
% The recognition rate is calculated
for k = 1:200
    knn = fitcknn(Imagestrain, train_lab, 'NumNeighbors', k);
    knn_prediction = predict(knn, Imagestest);
    knn_rec_rate = zeros(1, length(Imagestest(:,1)));
    
    for i = 1:length(Imagestest(:,1))
        if ceil(Indices(i,1)/5) == knn_prediction(i)
            knn_rec_rate(i) = 1;
        else
            knn_rec_rate(i) = 0;
        end
    end
    
    rec_rate(k) = (sum(knn_rec_rate)/70)*100;
end
figure
plot(K, rec_rate);
xlabel('K'); ylabel('Recognition rate')


%% k
averageRR=zeros(1,200);
  Threshold =20;  
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

for K=1:200
recognised_person_=zeros(1,70);
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
    recognised_person(1,i)=person(i,jmax);
  
 if (Identity(1,i)==recognised_person(1,i))
     recognitionrate=recognitionrate+1;
 end,

averageRR(1,K)=recognitionrate/70;
end,
end,

figure;
plot(averageRR(1,:));
title('Recognition rate against the number of nearest neighbours(threshold=20)');



