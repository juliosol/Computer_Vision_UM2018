function E_ = filterBlobs(im,L,E,sa,DoGtau)
%
%  W18 EECS 504 HW2p3 Blob detection
%
% the original scale space blob detector returns all local extrema in scale
% space, which tends to return many pixels that do not seem to be blobs.
%
% This function filters out extrema that had too weak a response in the 
%  DoG as well as non-blob-like regions.
%
% im is the grayscale, double image in the 0:1 range
% L is the DoG scale space
% E is the Extrema N x 3 matrix  (each row is X,Y,level)
% sa is the sigma vector (roughly 3*sigma is a size in image)
% DoGtau (optional) is the filter on the extrema response

if nargin<5
    DoGtau = 0.1;
end

% number of extrema inputted
n = size(E,1);

[r,c,b] = size(im);

if (b ~= 1)
    fprintf('please supply a grayscale image, double, in range 0:1\n');
    E_ = [];
    return
end

E_filter = [];

%%%% YOU NEED TO FILL IN THE CODE BELOW 
%%%% BE SURE TO FILTER OUT BOTH ON THE RESPONSE OF THE DOG AND THE
%%%% LOCAL IMAGE REGION BLOB-NESS

for i = 1:n
    vector = E(i,:);
    Pos_x = vector(1);
    Pos_y = vector(2);
    Pos_z = vector(3);
    %i
    %[Pos_y,Pos_x,Pos_z]
    %L(Pos_y,Pos_x,Pos_z)
    if abs(L(Pos_y,Pos_x,Pos_z)) > DoGtau
        E_filter = [E_filter;[Pos_x,Pos_y,Pos_z]];
    end
end


%%% Now we try to apply Harris operator and filter even more the maximum
%%% values we found

sz_max = size(E_filter);
sz_max_vector = sz_max(1);

bw_im = im;
size_im = size(im);
M = size_im(1);
N = size_im(2);

E_ = [];

for i = 1:sz_max_vector
    vector = E_filter(i,:);
    point_x = vector(2);
    point_y = vector(1);
    layer = vector(3);
    radius_layer = floor(2.5*sa(layer));
    
    zeros_row = zeros([N,radius_layer]);
    zeros_vector = zeros([M + 2*radius_layer,radius_layer]);
    
    im_zeros = [];
    im_zeros_2 = [];
    im_zeros = [zeros_row.'; im; zeros_row.'];
    im_zeros_2 = [zeros_vector, im_zeros, zeros_vector];    
    
    window = im_zeros_2(point_x + radius_layer - radius_layer:point_x + radius_layer +radius_layer,point_y+radius_layer - radius_layer:point_y + radius_layer +radius_layer);
    window_dx = conv2(window,fspecial('sobel'),'same');
    window_dy = conv2(window,fspecial('sobel')','same');
    Whalfsize = radius_layer;
%    Struct_Matrix = [];
    for k = 1:3
            if k == 1
                I_xsum = 0;
                for p = -Whalfsize:Whalfsize
                    for q = -Whalfsize:Whalfsize
                        I_xsum = I_xsum + window_dx(Whalfsize +1 + p,Whalfsize +1 +q)*window_dx(Whalfsize +1+p,Whalfsize + 1 +q);
                    end
                end
                Struct_Matrix(1,1) = I_xsum;
            elseif k == 2
                I_xysum = 0;
                for p = -Whalfsize:Whalfsize
                    for q = -Whalfsize:Whalfsize
                        I_xysum = I_xysum + window_dx(Whalfsize + 1 +p,Whalfsize + 1 +q)*window_dy(Whalfsize + 1 + p,Whalfsize + 1 +q);
                    end
                end 
                Struct_Matrix(1,2) = I_xysum;
                Struct_Matrix(2,1) = I_xysum;
            elseif k == 3
                I_ysum = 0;
                for p = -Whalfsize:Whalfsize
                    for q = -Whalfsize:Whalfsize
                        I_ysum = I_ysum + window_dy(Whalfsize + 1 + p,Whalfsize + 1 +q)*window_dy(Whalfsize + 1 +p,Whalfsize + 1 +q);
                    end
                end 
                Struct_Matrix(2,2) = I_ysum;
            end
        end
        %Struct_Matrix;
        %i
        eigenvalues = eig(Struct_Matrix);
        if min(eigenvalues) > 80
            E_ = [E_; [point_y,point_x,layer]];
        end
end

                         
    

%%%% YOU NEED TO STOP HERE
