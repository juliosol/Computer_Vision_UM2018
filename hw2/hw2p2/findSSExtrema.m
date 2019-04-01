function [E] = findSSExtrema(L)
%
%  W18 EECS 504 HW2p3 Blob detection
%
% Given the scale space layers image L (DoG Scale Space)
%  find local extrema and report them.
%
% E is an N by 3 matrix for N extrema, X, Y, Layer columns
%
% Extrema points are locally minimal or maximal in a 3x3x3 window in space and scale.
%
% You may not use the imregionalmax function in Matlab directly, but instead
% need to implement yourself (to gain the experience of working with scale spaces of images).

%%%%  YOU NEED TO FILL IN THE CODE BELOW 

sz = size(L(:,:,1));
sz_levels = size(L);

E = [];

M = sz(1);
N = sz(2);
levels = sz_levels(3);

zeros_vector = NaN([M + 2,1]);
zeros_row = NaN([N,1]);

%Appending columns of zeros to every matrix in every level

L_zeros = [];
L_zeros_2 = [];

for i = 1:levels
    L_zeros(:,:,i) = [zeros_row.'; L(:,:,i); zeros_row.'];
    L_zeros_2(:,:,i) = [zeros_vector, L_zeros(:,:,i), zeros_vector ];
end

NaN_layer = NaN([M+2,N+2]);

L_padded = cat(3,NaN_layer, L_zeros_2, NaN_layer);

size_padded = size(L_padded);
M_padded = size_padded(1);
N_padded = size_padded(2);

levels_padded = size_padded(3);

%L_padded(:,:,1)

%L_padded(:,:,9)

for k = 2:levels_padded - 1
    for j = 2:N_padded-1
        for i = 2:M_padded-1
            % Computing lower layer submatrix and finding max and min
            lower_submatrix = L_padded(i-1:i+1,j-1:j+1,k-1);
            max_lower_submatrix = max(max(lower_submatrix));
            min_lower_submatrix = min(min(lower_submatrix));
            
            % Computing upper layer submatrix
            upper_submatrix = L_padded(i-1:i+1,j-1:j+1,k+1);
            max_upper_submatrix = max(max(upper_submatrix));
            min_upper_submatrix = min(min(upper_submatrix));
            
            % COmparing middle number with neighbors in middle parameter
            middle_submatrix = L_padded(i-1:i+1,j-1:j+1,k);
            middle = middle_submatrix(2,2);
            middle_submatrix(2,2) = NaN;
            max_middle_submatrix = max(max(middle_submatrix));
            min_middle_submatrix = min(min(middle_submatrix));
            
            %max([middle,max_middle_submatrix,max_lower_submatrix,max_upper_submatrix])
            %min([middle, min_middle_submatrix, min_lower_submatrix,min_upper_submatrix])
            %min([middle, min_middle_submatrix, min_lower_submatrix,min_upper_submatrix]) == middle
            %max([middle,max_middle_submatrix,max_lower_submatrix,max_upper_submatrix]) == middle
            
            if min([middle, min_middle_submatrix, min_lower_submatrix,min_upper_submatrix]) == middle || max([middle,max_middle_submatrix,max_lower_submatrix,max_upper_submatrix]) == middle
                max_position = [j-1,i-1,k-1];
                E = [E;max_position];
            end
        end
    end
end

                
end            

%for i = 1:M
%    for j = 1:N
%        max_position_layer = [];
%        for k = 1:3
%            temp_mat = L_zeros_2(i:i+2,j:j+2,k);
%            [temp_max,j_max] = max(max(temp_mat));
%            if any(temp_mat(:,j_max)) == 0
%                i_max = 1;
%            else
%                i_max = find(temp_mat(:,j_max) == temp_max);
%                if isvector(i_max) == 1
%                    i_max = i_max(1);
%                end
%            end
%            max_position_layer = [max_position_layer; [temp_max, i_max,j_max]];
%        end
%        [max_ss,max_pos_ss] = max(max_position_layer(:,1));
%        E = [E; max_position_layer(max_pos_ss,:)];
%    end
%end






%%%%  YOU NEED TO STOP HERE
