function [eigenvectors,eigenvalues, average_face] = eigennumbers_basis(training_data_input,training_labels_input,k)

% This is a function that takes the training input data, training labels
% and some natural number k and returns the k largest eigenvalues and their
% corresponding eigenvectors. 

train_data = loadMNISTImages(training_data_input);
train_labels = loadMNISTLabels(training_labels_input);

size_train_data = size(train_data);
len_train_pixels = size_train_data(1);
len_train_images = size_train_data(2);

average_number = mean(train_data,2);

matrix_numbers_diff_average = bsxfun(@minus,train_data,average_number);

numbers_covariance_matrix = matrix_numbers_diff_average*(matrix_numbers_diff_average)';

[eigenvectors_pre_order,eigenvalues_pre_order] = eig(numbers_covariance_matrix);

eValues = diag(eigenvalues_pre_order);

[~ , indices] = sort(eValues,'descend');

sortedEigenvectors = eigenvectors_pre_order(:,indices);
sortedEigenvalues = eValues(indices);

k_eigenvectors_mat = sortedEigenvectors(:,1:k);
k_eigenvalues_vector = sortedEigenvalues(1:k);

eigenvalues = k_eigenvalues_vector;
eigenvectors = k_eigenvectors_mat;
average_face = average_number;

%d = size(eigenvectors);