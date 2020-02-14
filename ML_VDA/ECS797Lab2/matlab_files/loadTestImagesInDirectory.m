% loads images present in the directory in matrix Images.
% Identity contains the number identifying each person


function [Images,Identity] = loadTestImagesInDirectory(Path);


ListingOffset = 2;
Sizes = [28, 23];


FileList = dir(Path);
FileListSizes = size(FileList);
Identity=zeros(1, FileListSizes(1) - ListingOffset);

for i = 1:(FileListSizes(1) - ListingOffset)
    FileName = (strcat(Path, FileList(i + ListingOffset).name));
    ShortFileName=FileList(i + ListingOffset).name;
  
    Images(i, :) = double(reshape(imread(FileName), 1, Sizes(1)*Sizes(2)));
    Identity(1,i)=str2double(ShortFileName(3:4));
end,


