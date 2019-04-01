function [test_images_one_digit,test_images_two_digit,test_images_three_digit,test_images_four_digit,test_images_five_digit,test_labels] = SVHN_images(digitStruct,no_samples)

% Function that reads the content of the folders with SVHN images and prepares 
% to be able to apply the funcion class_vector, which makes classifications
% of the numbers.

if nargin < 2
    no_samples = 500;
end

image_size = 28;
%test_images_one_digit = zeros(784,no_samples);
test_images_one_digit = [];
test_images_two_digit = [];
test_images_three_digit = [];
test_images_four_digit = [];
test_images_five_digit = [];
test_labels = [];
num_test = 1;

% This chunk of code is for reading the numbers given in the images

for i = 1:no_samples
    %im = double(imread(['test_SVHN/' num2str(i) '.png']))/255;
    im = im2double(imread(['test_SVHN/' num2str(i) '.png']));
    num_test = num_test + 1;
    for j = 1:length(digitStruct(i).bbox)
        [height, width] = size(im);
        image_size = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        %image_size(1)
        %image_size(2)
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, image_size(1));
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, image_size(2));
        new_image = im(aa:bb,cc:dd,:);
        new_image_gray = rgb2gray(new_image);
        n_img_gray_reshape = imresize(new_image_gray,[28,28]);
        img_gray_to_class = [n_img_gray_reshape(:)];
        if length(digitStruct(i).bbox) == 1
            test_images_one_digit = [test_images_one_digit, [i; img_gray_to_class]];
        end
        if length(digitStruct(i).bbox) == 2
            test_images_two_digit = [test_images_two_digit, [i; img_gray_to_class]];
        end
        if length(digitStruct(i).bbox) == 3
            test_images_three_digit = [test_images_three_digit, [i; img_gray_to_class]];
        end
        if length(digitStruct(i).bbox) == 4
            test_images_four_digit = [test_images_four_digit,[i;  img_gray_to_class]];
        end
        if length(digitStruct(i).bbox) == 5
            test_images_five_digit = [test_images_five_digit, [i; img_gray_to_class]];
        end
    end 
    
    
    if length(digitStruct(i).bbox) == 1
        label = digitStruct(i).bbox(j).label;
        if label == 10
            label = 0;
        end
        test_labels = [test_labels; label];
    end
    
    if length(digitStruct(i).bbox) == 2
        label = 0;
        for j = 2:-1:1
            label_pre = digitStruct(i).bbox(3-j).label;
            if label_pre == 10
                label_pre = 0;
            end
            label = label_pre*10^(j-1) + label ;
        end
        test_labels = [test_labels; label];
    end
    
    if length(digitStruct(i).bbox) == 3
        label = 0;
        for j = 3:-1:1
            label_pre = digitStruct(i).bbox(4-j).label;
            if label_pre == 10
                label_pre = 0;
            end
            label = label_pre*10^(j-1) + label;
        end
        test_labels = [test_labels; label];
    end
    if length(digitStruct(i).bbox) == 4
        label = 0;
        for j = 4:-1:1
            label_pre = digitStruct(i).bbox(5-j).label;
            if label_pre == 10
                label_pre = 0;
            end
            label = label_pre*10^(j-1) + label ;
        end
        test_labels = [test_labels; label];
    end
    if length(digitStruct(i).bbox) == 5
        label = 0;
        for j = 5:-1:1
            label_pre = digitStruct(i).bbox(6-j).label;
            if label_pre == 10
                label_pre = 0;
            end
            label = label_pre*10^(j-1) + label ;
        end
        test_labels = [test_labels; label];
    end
end
%    size(test_images_five_digit(1,:))
    
end

