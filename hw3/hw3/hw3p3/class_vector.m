function [avg_error_prediction,avg_acuracy_prediction, prediction_vector] = class_vector_MNIST(train_images_input,train_labels_input,test_images_input, test_labels_input,k,epsilon_k,epsilon)

train_data = loadMNISTImages(train_images_input);
train_labels = loadMNISTLabels(train_labels_input);
test_data_total = loadMNISTImages(test_images_input);
test_labels_total = loadMNISTLabels(test_labels_input);

%train_data = train_data_total(:,1:500);
%train_labels = train_labels_total(1:500);
test_data = test_data_total(:,1:500);
test_labels = test_labels_total(1:500);

[eigenvectors,eigenvalues, average_face] = eigennumbers_basis(train_images_input,train_labels_input,k);


%imshow(reshape(average_face,[28,28]));
%for i = 1:k
%    imagesc(reshape(eigenvectors(:,i),[28,28]));
%end

eigenvectors_size = size(eigenvectors);
no_eigenvectors = eigenvectors_size(2);
size_matrix = size(train_data);
qty_numbers = size_matrix(2);
size_number_vector = size_matrix(1);

size_matrix_test = size(test_data);
qty_numbers_test = size_matrix_test(2);
size_number_vector_test = size_matrix_test(1);

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
    for j = 1:qty_numbers
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
    %size(eigenfaces_class_weight_vector)
    %eigenface_class_image = eigenfaces_class_weight_vector' * eigenvectors'
    %imagesc(reshape(eigenface_class_image,[28,28]))
    eigenfaces_classes_weight_vector(:,i) = eigenfaces_class_weight_vector;
end

%for l = 1:k
%    imshow(reshape(eigenfaces_classes_weight_vector(:,l)' * eigenvectors', [28,28]))
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This part is for doing the actual prediction
% We first compute the projection of the new test image onto the test space
% and then we measure the distance from the corresponding eigenface
% representation for the given class.

prediction_vector = zeros(qty_numbers_test,1);
error = 0;

% This part of the code actually makes the predictions. We are checking
% whether the vector of weights computed for the current image is close to
% one of the face classes we computed above and whether it is close to face
% space. 

for i = 1:qty_numbers_test
    current_number_test = test_data(:,i);
    %imshow(reshape(current_number_test,[28,28]))
    eigenface_weights_test = zeros(1,k);
    
    % This part is for computing the projection of the normalized current 
    % number image into the fce space using the eigenfaces
    
    for j = 1:k
        component_j  = dot(eigenvectors(:,j),current_number_test - average_face);
        eigenface_weights_test(1,j) = component_j;
    end
    
    %test_labels(i)
    %check_im = eigenface_weights_test * eigenvectors';
    %imshow(reshape(check_im,[28,28]))
    
    % This part is for computing the distance from the projection of the
    % given image weight vector to each of the face classes for each digit
    % and determining which one is hte smallest one possible. 
    
    %min_distance_face_class = 0;
    min_k_face_class = 0;
   
    for l = 1: k
        %l-1
        %eigenface_weights_test
        %eigenfaces_classes_weight_vector(:,l)
        %eigenface_weights_test' - eigenfaces_classes_weight_vector(:,l)
        distance_eigenface = norm(eigenface_weights_test' - eigenfaces_classes_weight_vector(:,l));
        if l == 1
            epsilon_k = distance_eigenface;
        end
        if distance_eigenface < epsilon_k
            min_k_face_class = l-1;
            min_distance_face_class = distance_eigenface;
            epsilon_k = distance_eigenface;
         end
    end
    prediction_vector(i,1) = min_k_face_class;
    %min_k_face_class;
    
    % This chunk of code checks the closedness to facespace.
    
    facespace = 0;
    for l = 1:k
       aa = eigenface_weights_test(l) * eigenvectors(:,l);
       facespace = aa + facespace;
    end  
    mean_adjusted_img = current_number_test - average_face;
    
    
end  

difference = (test_labels == prediction_vector);

counts_differences = hist(difference,numel(unique(difference)));

no_wrong_predictions = counts_differences(1);
correct_predictions = counts_differences(2);

avg_error_prediction = no_wrong_predictions/qty_numbers_test;
avg_acuracy_prediction = correct_predictions/qty_numbers_test;

end