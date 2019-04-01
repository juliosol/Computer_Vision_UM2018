%
% W18 EECS 504 HW1p1 Homography Estimation
%

close all;

% Load image
inpath1 = 'football1.jpg';
inpath2 = 'football2.jpg';

im1 = imread(inpath1);
im2 = imread(inpath2);

% Display the yellow line in the first image
figure;
imshow(im1); title('football image 1');
hold on;
u=[1210,1701];v=[126,939];  % marker 33
%u=[942,1294];v=[138,939];
plot(u,v,'y','LineWidth',2);
hold off;

%------------------------------------------------
% FILL BLANK HERE
% Specify the number of pairs of points you need.
n = 4;
%------------------------------------------------

% Get n correspondences
baseName = regexp(inpath1,'^\D+','match','once');
pointsPath = sprintf('%s_points%i.mat',baseName,n);
if exist(pointsPath,'file') 
   % Load saved points
   load(pointsPath);
else
   % Get correspondences
   [XY1, XY2] = getCorrespondences(im1,im2,n);
   save(pointsPath,'XY1','XY2');
end

%------------------------------------------------
% FILL YOUR CODE HERE
% Your code should estimate the homography and draw the  
% corresponding yellow line in the second image.

%We first form matrix A:

A = zeros(3*n,9);

for i = 1:n
    %disp(i)
    k = 1 + 3*(i-1);
    %disp(k)
    for j = 1:9
        %disp(j)
        if mod(j,3) < 3 && mod(j,3) ~= 0
            %disp(XY1(i,mod(j,3)))
            A(k,j) = XY1(i,mod(j,3));
        end
        if mod(j,3) == 0
            A(k,j) = 1;
            k = k+1;
        end
    end
end

% Now, we form the vector Y

Y = zeros(3*n,1);

for i = 1:n
    k = 1 + 3*(i-1);
    %disp(i);
    for j = 1:3
        if mod(j,3) < 3 && mod(j,3) ~= 0
            Y(k,1) = XY2(i,mod(j,3));
            k = k + 1;
        end
        if mod(j,3) == 0
            Y(k,1) = 1;
            k = k+1; 
        end
    end
end

%Computing vector X (vector where every entry is one of the entries of
%transformation matrix

Ainv = inv(A'*A);
H = Ainv * A' * Y;

%The last step is to make the entries in H form the transformation matrix.

Hmatrix = vec2mat(H,3);

%Now we take the points u and v from the original yellow line and apply the
%transformation H to them.

point1 = [u(1), v(1)];
point2 = [u(2),v(2)];

point1_transform = Hmatrix * [point1,1]';
point2_transform = Hmatrix * [point2,1]';

u_transform = [point1_transform(1),point2_transform(1)];
v_transform =[point1_transform(2),point2_transform(2)];

% Now, we display the yellow line in the second image
figure;
imshow(im2); title('football image 2');
hold on;
%u=[1210,1701];v=[126,939];  % marker 33
%u=[942,1294];v=[138,939];

plot(u_transform,v_transform,'y','LineWidth',2);
hold off;

%------------------------------------------------
