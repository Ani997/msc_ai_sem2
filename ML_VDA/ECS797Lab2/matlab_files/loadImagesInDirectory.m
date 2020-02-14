% loads images present in the directory in matrix Images.

function Images = loadImagesInDirectory(Path);


ListingOffset = 2;
Sizes = [28, 23];

FileList = dir(Path);
FileListSizes = size(FileList);

for i = 1:(FileListSizes(1) - ListingOffset)
    FileName = (strcat(Path, FileList(i + ListingOffset).name));
    Images(i, :) = double(reshape(imread(FileName), 1, Sizes(1)*Sizes(2)));
  
end,