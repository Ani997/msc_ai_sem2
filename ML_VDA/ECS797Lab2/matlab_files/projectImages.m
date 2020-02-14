
% projects vectors 'Vectors' into the space 'Space'.


function Locations = projectImages(Vectors, Means, Space)

VectorsSizes = size(Vectors);
V= repmat(Means, VectorsSizes(1), 1);
Locations = (Vectors -V)*Space' ;