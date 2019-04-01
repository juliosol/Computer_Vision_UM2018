function C = harris(dx,dy,Whalfsize)

% function C = harris(dx,dy,Whalfsize)
%
%     EECS 504;
%     Jason Corso
%
%   I is the image (GRAY, DOUBLE)
%   or
%   dx is the horizontal gradient image
%   dy is the vertical gradient image
%
%   If you call it with the Image I, then you need set parameter dy to []
%
%   Whalfsize is the half size of the window.  Wsize = 2*Whalfsize+1
%
%   Corner strength is taken as min(eig) and not the det(T)/trace(T) as in
%   the original harris method.  Just for simplicity
%
%   Output
%   C is an image (same size as dx and dy) where every pixel contains the
%   corner strength.  


if (isempty(dy))
   im = dx;
   dy = conv2(im,fspecial('sobel'),'same');
   dx = conv2(im,fspecial('sobel')','same'); 
end


%%% YOU NEED TO FILL THE CODE BELLOW
% Corner strength is to be taken as min(eig) and not the det(T)/trace(T) as in
%  the original harris method.

Wsize = 2*Whalfsize + 1;

% This is the matrix of zeroes that we ned to fill out.

sz = size(dx);

M = sz(1);

N = sz(2);

C = zeros(M,N);

Struct_Matrix = zeros(2,2);

% Padding NaN to dx and dy

zeros_row = zeros([N,Whalfsize]);
%size(NaN_row);
zeros_vector = zeros([M + 2*Whalfsize,Whalfsize]);
%size(NaN_vector);

dx_zeros = [];
dx_zeros_2 = [];

dx_zeros = [zeros_row.'; dx; zeros_row.'];
size(dx_zeros);
dx_zeros_2 = [zeros_vector, dx_zeros, zeros_vector];
size(dx_zeros_2);

dy_zeros = [];
dy_zeros_2 = [];

dy_zeros = [zeros_row.'; dy; zeros_row.'];
dy_zeros_2 = [zeros_vector, dy_zeros, zeros_vector ];


size_NaN = size(dy_zeros_2);
M_NaN = size_NaN(1);
N_NaN = size_NaN(2);


% In the following lines we build the Structure tensor and take the
% eigenvalue of the matrix.

for i = (Whalfsize+1): M - Whalfsize
    for j = (Whalfsize + 1): N - Whalfsize
        for k = 1:3
            if k == 1
                I_xsum = 0;
                for p = -Whalfsize:Whalfsize
                    for q = -Whalfsize:Whalfsize
                        I_xsum = I_xsum + dx(i+p,j+q)*dx(i+p,j+q);
                    end
                end
                Struct_Matrix(1,1) = I_xsum;
            elseif k == 2
                I_xysum = 0;
                for p = -Whalfsize:Whalfsize
                    for q = -Whalfsize:Whalfsize
                        I_xysum = I_xysum + dx(i+p,j+q)*dy(i+p,j+q);
                    end
                end 
                Struct_Matrix(1,2) = I_xysum;
                Struct_Matrix(2,1) = I_xysum;
            elseif k == 3
                I_ysum = 0;
                for p = -Whalfsize:Whalfsize
                    for q = -Whalfsize:Whalfsize
                        I_ysum = I_ysum + dy(i+p,j+q)*dy(i+p,j+q);
                    end
                end 
                Struct_Matrix(2,2) = I_ysum;
            end
        end
        %Struct_Matrix;
        minimum_eigenvalue = min(eigs(Struct_Matrix));
        C(i,j) = minimum_eigenvalue;                 
    end
end

C;
end


%%% YOU NEED TO STOP HERE
