function dx_image = derivative_x(I)

sz = size(I);

M = sz(1);
N = sz(2);

dxI = zeros([M,N]);

% Derivative with respect to x:

for i = 1:M
    for j = 2:N
        dxI(i,j) = I(i,j) - I(i,j-1);
    end
end

dx_image = dxI;
    
    