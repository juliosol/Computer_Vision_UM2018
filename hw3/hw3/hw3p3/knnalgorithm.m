function [Error_rate] = knnalgorithm(test_data,test_labels,k)

train_data = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_data = loadMNISTImages(test_data);
test_labels = loadMNISTLabels(test_labels);

len_train = length(train_labels);
len_test = length(test_labels);

data_train_big = [];

for i = 1 : len_train
    data_train_big = [data_train_big, [train_labels(i);train_data(:,i)]];
end

data_train_big(:,1);
%print(length(data_train_big(:,1)))
test_predictions = [];

for j = 1 : len_test
    distances = [];
    for i = 1 : len_train
        len_number = data_train_big(:,i);
        data_train_current = data_train_big(:,i);
        %data_train_current
        length(test_data(:,j))
        data_train_number = data_train_current(2:len_number);
        length(data_train_number)        
        distance_i = norm(test_data(:,j) - data_train_number);
        distances = [distances;[distance_i,i]];
    end
    sorted_distances = sortrows(distances);
    sorted_k_positions = sorted_distances(1:k,2);
    number_k_vectors = length(sorted_k_positions);
    
    possible_prediction = [];
    
    for i = 1 : number_k_vectors
        possible_prediction = [possible_prediction; data_train(1,i)];
    end
    
    prediction_i = mode(possible_prediction);
    test_predictions = [test_predictions;prediction_i];        
end

counter_errors = 0;
length_predictions = length(test_predictions);

for i = 1:length_predictions
    if test_predictions(i) ~= test_labels
        counter_errors = counter_errors + 1;
    end
end

Error_rate = counter_errors/len_labels;    


