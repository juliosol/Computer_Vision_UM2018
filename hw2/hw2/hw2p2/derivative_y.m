function dy_image = derivative_y(I)

sz = size(I);

M = sz(1);
N = sz(2);

dyI = zeros([M,N]);

% Derivative with respect to x:

for j = 1:N
    for i = 2:M
        dyI(i,j) = I(i,j) - I(i-1,j);
    end
end

dy_image = dyI;