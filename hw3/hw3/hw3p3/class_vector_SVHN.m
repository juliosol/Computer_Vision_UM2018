function [avg_error_prediction_SVHN,avg_acuracy_prediction_SVHN, prediction_vector_SVHN] = class_vector_SVHN(train_images_input,train_labels_input,k,epsilon_k,no_samples)

run('test_SVHN/see_bboxes.m');
train_data = loadMNISTImages(train_images_input);
train_labels = loadMNISTLabels(train_labels_input);

[eigenvectors,eigenvalues, average_face] = eigennumbers_basis(train_images_input,train_labels_input,k);

no_samples = 500;

[test_images_one_digit,test_images_two_digit,test_images_three_digit,test_images_four_digit,test_images_five_digit,test_labels] = SVHN_images(digitStruct,no_samples);

eigenvectors_size = size(eigenvectors);
no_eigenvectors = eigenvectors_size(2);
size_matrix_train = size(train_data);
qty_numbers_train = size_matrix_train(2);
size_number_vector = size_matrix_train(1);

%size_matrix_test = size(test_data);
qty_numbers_test = no_samples;
%size_number_vector_test = size_matrix_test(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This part is used for computing the face classes for each of the digit we
% want to predict.

% Vector of eigenfaces classes of size k 

range_numbers = linspace(0,k,k+1);
eigenfaces_classes_weight_vector = zeros(k,k);
possible_labels = linspace(0,k,k+1);
total = 0;

for i = 1 : k 
    current_sum_vector = zeros(k,1);
    current_label = possible_labels(i);
    count_number_current_labels = 0;
    for j = 1:qty_numbers_train
        if train_labels(j) == current_label
            temporal_pattern_vector = zeros(k,1);
            current_vector = train_data(:,j);
            %imshow(reshape(current_vector,[28,28]));
            for l = 1:k
                temporal_pattern_vector(l) = dot(eigenvectors(:,l),current_vector - average_face);
            end
            size(temporal_pattern_vector);
            current_sum_vector = current_sum_vector + temporal_pattern_vector;
            count_number_current_labels = count_number_current_labels + 1;
        end
    end
    total = total + count_number_current_labels;
    eigenfaces_class_weight_vector = current_sum_vector./count_number_current_labels;
    eigenfaces_classes_weight_vector(:,i) = eigenfaces_class_weight_vector;
end
%eigenfaces_classes_weight_vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This part is for doing the actual prediction
% We first compute the projection of the new test image onto the test space
% and then we measure the distance from the corresponding eigenface
% representation for the given class.

prediction_vector_SVHN = zeros(qty_numbers_test,1);
error = 0;

% This part of the code actually makes the predictions. We are checking
% whether the vector of weights computed for the current image is close to
% one of the face classes we computed above and whether it is close to face
% space. 

for i = 1:qty_numbers_test
    
    % This chunk of code is for getting the current number we want to
    % predict from the number of SVHN. Since these numbers can be of more
    % than 1 digit, we have a big division of work.
    len1d = length(test_images_one_digit(1,:));
    len2d = length(test_images_two_digit(1,:));
    len3d = length(test_images_three_digit(1,:));
    size(test_images_four_digit)
    if size(test_images_four_digit) == 0
        len4d = 0;
    else
        len4d = length(test_images_four_digit(1,:));
    end
    if size(test_images_five_digit)== 0 
        len5d = 0;
    else
        len5d = length(test_images_five_digit(1,:));
    end
    max_len = max(max(max(max(len1d,len2d),len3d),len4d),len5d);
    
     
    for m = 1: max_len
        if m < len1d && test_images_one_digit(1,m) == i 
            current_image = test_images_one_digit(2:785,m);
            test_images_one_digit(2:785,m) = zeros(784,1);
            break
        elseif m < len2d && test_images_two_digit(1,m) == i 
            current_image = test_images_two_digit(2:785,m:m+1);
            test_images_two_digit(2:785,m:m+1) = zeros(784,2);
            break
        elseif m < len3d && test_images_three_digit(1,m) == i
            current_image = test_images_three_digit(2:785,m:m+2);
            test_images_three_digit(2:785,m:m+2) = zeros(784,3);
            break
        elseif m < len4d && test_images_four_digit(1,m) == i
            current_image = test_images_four_digit(2:785,m:m+3);
            test_images_four_digit(2:785,m:m+3) = zeros(784,4);
            break
       end
    end
        
    size_number = size(current_image);
    no_digits = size_number(2);
    digits_predicted = zeros(no_digits,1);
            
    for l = 1: no_digits
        imshow(reshape(current_image(:,l),[28,28]))
        current_number_test = current_image(:,l);
        eigenface_weights_test = zeros(1,k);
    
        % This part is for computing the projection of the normalized current 
        % number image into the fce space using the eigenfaces
        %k
        for j = 1:k
            component_j  = dot(eigenvectors(:,j),im2double(current_number_test) - average_face);
            eigenface_weights_test(1,j) = component_j;
        end
        
       
        % This part is for computing the distance from the projection of the
        % given image weight vector to each of the face classes for each digit
        % and determining which one is hte smallest one possible. 
    
        min_k_face_class = 0;
   
        for s = 1: k
            distance_eigenface = norm(eigenface_weights_test' - eigenfaces_classes_weight_vector(:,s));
            if s == 1
                epsilon_k = distance_eigenface;
            end
            if distance_eigenface < epsilon_k
                min_k_face_class = s-1;
                min_distance_face_class = distance_eigenface;
                epsilon_k = distance_eigenface;
            end
        end
        digits_predicted(l,1) = min_k_face_class;
        min_k_face_class;
    end
    predicted_house_number = 0;
    for s = 1:no_digits
        predicted_house_number = digits_predicted(s,1)*10^(no_digits-s) + predicted_house_number;
    end
    prediction_vector_SVHN(i,1) = predicted_house_number;
end  

difference = (test_labels == prediction_vector_SVHN);

counts_differences = hist(difference,numel(unique(difference)));

no_wrong_predictions = counts_differences(1);
correct_predictions = counts_differences(2);

avg_error_prediction_SVHN = no_wrong_predictions/qty_numbers_test;
avg_acuracy_prediction_SVHN = correct_predictions/qty_numbers_test;

end