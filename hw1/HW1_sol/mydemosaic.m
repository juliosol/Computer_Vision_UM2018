function I = mydemosaic(I_gray)

%
% W18 EECS 504 HW1p4 Bayer Demosaicking
%
% function mydemosaic recovers the original color image (M*N*3)
% from the Bayer encoded image I_gray (M*N).                                                                                                                                                                                                                                                                                                                                                                                                                                                            
%

%---------------------------------
% IMPLEMENT THE FUNCTION HERE

% Generating Bayern pattern filter

%R = [0.25, 0.5, 0.25 : 0.5, 1, 0.5: 0.25, 0.5, 0.25 ];
%B = [ 0.25, 0.5, 0.5; 0.5,; ];
%G = [ ; ; ];

% Generating the Bayer encoded image.
%[I,I_grey] = bayer_filter(im1);
double_I_gray = im2double(I_gray);
[M,N] = size(double_I_gray);
T = zeros(M,N,3);
%T(:,:,1) Red layer
%T(:,:,2) Green layer
%T(:,:,3) Blue layer

for i = 2:M-1
    for j = 2:N-1
        if mod(i,2) == 0  && mod(j,2) == 0 % Green pixels on even row
            T(i,j,1) = (double_I_gray(i,j-1) + double_I_gray(i,j+1))/2; %Red part
            T(i,j,2) = (double_I_gray(i,j)); % Green part
            T(i,j,3) = (double_I_gray(i+1,j) + double_I_gray(i-1,j))/2; % Blue Part
        elseif mod(i,2) == 1 && mod(j,2) == 1 %Green layer on odd row
            T(i,j,1) = (double_I_gray(i+1,j) + double_I_gray(i-1,j))/2; %Red part
            T(i,j,2) = double_I_gray(i,j); %Green part
            T(i,j,3) = (double_I_gray(i,j-1) + double_I_gray(i,j+1))/2; %Blue part
        elseif mod(i,2) == 1 && mod(j,2) == 0 % First selecting the blue pixels
            T(i,j,1) = (double_I_gray(i-1,j+1) + double_I_gray(i-1,j-1) + double_I_gray(i+1,j-1) + double_I_gray(i+1,j+1))/4; %Red layer
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i+1,j) + double_I_gray(i,j-1) + double_I_gray(i,j+1))/4; %Green Layer
            T(i,j,3) = double_I_gray(i,j);%Blue layer
        elseif mod(i,2) == 0 &&  mod(j,2)== 1 % Select red pixels
            T(i,j,1) = double_I_gray(i,j); %Red Layer
            T(i,j,2) = (double_I_gray(i,j-1) + double_I_gray(i,j+1) + double_I_gray(i-1,j) + double_I_gray(i+1,j))/4; %Green Layer
            T(i,j,3) =  (double_I_gray(i-1,j-1) + double_I_gray(i-1,j+1)+ double_I_gray(i+1,j-1) + double_I_gray(i+1,j+1))/4; %Blue part
        end
    end
end

% Now we correct for the borders. We divided the borders into 4 corners and
% 4 edges. 

%------------------------------------------------------------------

% Checking corners:

% UL corner:
for i = [1,M]
    for j = [1,N]
        if i == 1 && j == 1 %UL corner (green corner)
            T(i,j,1) = double_I_gray(i+1,j); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i,j+1); % Blue part

            %Two possibilities for UR corner
            %1) for even N (blue corner)
        elseif i == 1 && j == N && mod(N,2) == 0
            T(i,j,1) = double_I_gray(i+1,j-1); % Red part
            T(i,j,2) = (double_I_gray(i,j-1) + double_I_gray(i+1,j))/2; % Green part
            T(i,j,3) = double_I_gray(i,j); % Blue part
            %2) for odd N (green corner)
        elseif i == 1 && j == N && mod(N,2) == 1
            T(i,j,1) = double_I_gray(i+1,j); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i,j-1); % Blue part
    
            %DL corner, two possibilities:
            %1) for odd M (green corner)
        elseif j == 1 && i == M && mod(M,2) == 1
            T(i,j,1) = double_I_gray(i-1,j); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i,j+1); % Blue part
            %2) for even M (red corner)
        elseif j == 1 && i == M && mod(M,2) == 0 
            T(i,j,1) = double_I_gray(i,j); % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i,j+1))/2; % Green part
            T(i,j,3) = double_I_gray(i-1,j+1); % Blue part

            %DR corner, four possibilities:
            %M odd and N odd (green corner)
        elseif j == N && i == M && mod(N,2) == 1 && mod(M,2) == 1  
            T(i,j,1) = double_I_gray(i-1,j); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i,j-1); % Blue part
            %M even and N even (green corner)   
        elseif j == N && i == M && mod(N,2) == 0 && mod(M,2) == 0 
            T(i,j,1) = double_I_gray(i,j-1); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i-1,j); % Blue part
            % M even and N odd (red corner)
        elseif j == N && i == M && mod(N,2) == 1 && mod(M,2) == 0 
            T(i,j,1) = double_I_gray(i,j); % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i,j-1))/2; % Green part
            T(i,j,3) = double_I_gray(i-1,j-1); % Blue part
            % M odd and N even (blue corner)
        elseif j == N && i == M && mod(N,2) == 0 && mod(M,2) == 1 
            T(i,j,1) = double_I_gray(i-1,j-1); % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i,j-1))/2; % Green part
            T(i,j,3) = double_I_gray(i,j); % Blue part
        end
    end
end

%----------------------------------------

%Edges of the picture

%Upper and lower edge

%Upper edge
for i = [1,M]
    for j = 2:N-1
        if mod(j,2) == 0 && i == 1 % Blue upper pixels
            T(i,j,1) = (double_I_gray(i+1,j-1) + double_I_gray(i+1,j+1))/2; % Red part
            T(i,j,2) = (double_I_gray(i,j-1) + double_I_gray(i+1,j) + double_I_gray(i,j+1))/3; % Green part
            T(i,j,3) = double_I_gray(i,j); % Blue part
        elseif mod(j,2) == 1 && i == 1% Green upper pixels
            T(i,j,1) = (double_I_gray(i+1,j)); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = (double_I_gray(i,j-1) + double_I_gray(i,j+1))/2; % Blue part
        elseif i == M && mod(j,2) == 0 && mod(M,2) == 1 % Blue lower pixels and M odd
            T(i,j,1) = (double_I_gray(i-1,j-1) + double_I_gray(i-1,j+1))/2; % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i,j-1) + double_I_gray(i,j+1))/3; % Green part
            T(i,j,3) = double_I_gray(i,j); % Blue part
        elseif i == M && mod(j,2) == 1 && mod(M,2) == 1 % Green lower pixels and M odd
            T(i,j,1) = (double_I_gray(i-1,j)); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = (double_I_gray(i,j+1) + double_I_gray(i,j-1))/2; % Blue part
        elseif i == M && mod(j,2) == 0 && mod(M,2) == 0 %Lower part green pixels and even M
            T(i,j,1) = (double_I_gray(i,j-1) + double_I_gray(i,j+1))/2; % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = (double_I_gray(i-1,j)); % Blue part 
        elseif i == M && mod(j,2) == 1 && mod(M,2) == 0 % Lower part red  pixels and even M
            T(i,j,1) = double_I_gray(i,j); % Red part
            T(i,j,2) = (double_I_gray(i,j-1) + double_I_gray(i,j+1) + double_I_gray(i-1,j))/3; % Green part
            T(i,j,3) = (double_I_gray(i-1,j-1) + double_I_gray(i-1,j+1))/2; % Blue part 
        end
    end
end

%Left and right edge

for j = [1,N]
    for i = 2:M-1
        if j == 1 && mod(i,2) == 0 %Left edge red pixel
            T(i,j,1) = double_I_gray(i,j); % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i+1,j) + double_I_gray(i,j+1))/3; % Green part
            T(i,j,3) = (double_I_gray(i-1,j+1) + double_I_gray(i+1,j+1))/2; % Blue part
        elseif j == 1 && mod(i,2) == 1 %Left edge green pixel
            T(i,j,1) = (double_I_gray(i+1,j) + double_I_gray(i-1,j))/2; % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i,j+1); % Blue part
        elseif j == N && mod(i,2) == 0 && mod(N,2) == 1 %Right edge, red pixel with N odd
            T(i,j,1) = double_I_gray(i,j); % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i+1,j) + doubl_I_gray(i,j-1))/3; % Green part
            T(i,j,3) = (double_I_gray(i-1,j-1) + double_I_gray(i+1,j-1))/2; % Blue part
        elseif j == N && mod(i,2) == 1 && mod(N,2) == 1 % Right edge, green pixel with N odd
            T(i,j,1) = (double_I_gray(i+1,j) + double_I_gray(i-1,j))/2; % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = double_I_gray(i,j-1); % Blue part
        elseif j == N && mod(i,2) == 0 && mod(N,2) == 0 %Right edge, green pixel with N even
            T(i,j,1) = double_I_gray(i,j-1); % Red part
            T(i,j,2) = double_I_gray(i,j); % Green part
            T(i,j,3) = (double_I_gray(i-1,j) + double_I_gray(i+1,j))/2; % Blue part
        elseif j == N && mod(i,2) == 1 && mod(N,2) == 0 %Right edge, blue pixel with N even
            T(i,j,1) = (double_I_gray(i-1,j-1) + double_I_gray(i+1,j-1))/2; % Red part
            T(i,j,2) = (double_I_gray(i-1,j) + double_I_gray(i+1,j) + double_I_gray(i,j-1))/3; % Green part
            T(i,j,3) = double_I_gray(i,j); % Blue part
        end
    end
end

T;
figure; imshow(T);
end