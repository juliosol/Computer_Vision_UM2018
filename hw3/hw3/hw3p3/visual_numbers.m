function numbers = visual_numbers(training_data_input,training_labels_input,k)

% This script takes the basis of eigennumbers and makes the necessary
% transformations to be able to visualize the basis elements.

[eigenvectors,eigenvalues] = eigennumbers_basis(training_data_input,training_labels_input,k);

size_eigenvectors_mat = size(eigenvectors)
num_numbers = size_eigenvectors_mat(2);

numbers = [];

for i = 1:num_numbers
    numbers(:,:,i) = reshape(eigenvectors(:,i),[28,28]);
end