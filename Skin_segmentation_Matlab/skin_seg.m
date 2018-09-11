I = imread('./Data/3.jpg');
img=rgb2ycbcr(I);

for i=1:size(img,1)
    for j= 1:size(img,2)
        cb = img(i,j,2);
        cr = img(i,j,3);
        if(~(cr > 132 && cr < 173 && cb > 100 && cb < 126))
            img(i,j,1)=0;
            img(i,j,2)=128;
            img(i,j,3)=128;
        end
    end
end
img=ycbcr2rgb(img);
figure,imshow(img);
title('Image converted to YCbCr');
m_img = rgb2gray(img);
m_img(find(m_img<=50))=0;
binary_skin = m_img;
m_img = imfill(m_img,'holes');
se = strel('disk',10); %create structuring element
m_img = imerode(m_img,se);
se = strel('disk',4);
m_img = imdilate(m_img,se);
m_img = immultiply(m_img,binary_skin);
figure,imshow(m_img);
title('After Morphological processing');
CC = bwconncomp(m_img, 4);
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
avg = mean([S.Area]) ;
BW2 = ismember(L, find([S.Area] >= avg));
figure;
imshow(BW2);
title('Rejecting Small area');
CC = bwconncomp(BW2, 4);
L = labelmatrix(CC);
stats = regionprops(CC,'EulerNumber');
holes = stats;
stats = find([stats.EulerNumber] <= 0 & [stats.EulerNumber] >= -75);
BW2 = ismember(L, stats);
figure;
imshow(BW2);
title('Rejection of area with no eyes,etc');
CC = bwconncomp(BW2, 4);
L = labelmatrix(CC);
stats2 = regionprops(CC,'Eccentricity');
ecc = stats2;
stats2 = find([stats2.Eccentricity] <= 0.97);
BW2 = ismember(L, stats2);
figure;
imshow(BW2);
title('Rejection of non oval areas');
P=bwlabel(BW2,4);
%BB=regionprops(P,'Boundingbox');
%BB1=struct2cell(BB);
%BB2=cell2mat(BB1);
%[s1 s2]=size(BB2);
%mx=0;
%for k=3:4:s2-1
%    p=BB2(1,k)*BB2(1,k+1);
%    if p>mx && (BB2(1,k)/BB2(1,k+1))<1.8
%        mx=p;
%        j=k;
%end
st = regionprops(P,'Boundingbox');
figure,imshow(I);
hold on;
for k = 1 : length(st)
  thisBB = st(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end
title('Final Image');