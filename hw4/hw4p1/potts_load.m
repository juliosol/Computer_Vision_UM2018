% W18 EECS 504 HW4p1 Mumford-Shah Piecewise Constant
% Siyuan Chen

load 'potts_data.mat'

E1 = potts(I1);
E2 = potts(I2);

figure(1); imagesc(I1); title(sprintf("I1, E = %d", E1));
figure(2); imagesc(I2); title(sprintf("I2, E = %d", E2));