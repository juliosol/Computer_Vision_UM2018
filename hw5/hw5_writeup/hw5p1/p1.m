% W18 EECS 504 HW5p1
% Implement a simple 2-layer NN.
% Fill the missing part in this program. Blanks are marked with "FILL THE BLANK".

% to reproduce the same result
rng(0);

% load data
if ~exist('data', 'var')
    data = loadMNISTImages('data/train-images.idx3-ubyte');
    labels = loadMNISTLabels('data/train-labels.idx1-ubyte') + 1;
    tdata = loadMNISTImages('data/t10k-images.idx3-ubyte');
    tlabels = loadMNISTLabels('data/t10k-labels.idx1-ubyte') + 1;
    
    rp = randperm(size(data, 2));
    data = data(:, rp(1:10000));
    labels = labels(rp(1:10000));
    
    trp = randperm(size(tdata, 2));
    tdata = tdata(:, trp(1:1000));
    tlabels = tlabels(trp(1:1000));
end

% network setting
[dim, num_data] = size(data);
[~, num_tdata] = size(tdata);
num_epoch = 100;
batch_size = 500;
num_hidden_1 = 50;
num_output = 10;

% basic setting
% xaxier's method
r1 = sqrt(6) / sqrt(dim + num_hidden_1);
w1 = rand(num_hidden_1, dim)*2*r1 - r1;
b1 = zeros(num_hidden_1, 1);

r2 = sqrt(6) / sqrt(num_hidden_1 + num_output);
w2 = rand(num_output, num_hidden_1)*2*r2 - r2;
b2 = zeros(num_output, 1);

lr_w = 2; % learning rate for w term
lr_b = 2; % learning rate for b tew1rm
gam = 0.5;

% loss/acc history
loss_list = zeros(num_epoch, 1);
acc_list = zeros(num_epoch, 1);
tloss_list = zeros(num_epoch, 1);
tacc_list = zeros(num_epoch, 1);

for i = 1 : num_epoch
    rp = randperm(num_data);
    
    for j = 1 : batch_size : (num_data - batch_size + 1)
        % random batch
        x = data(:, rp(j:j+batch_size-1));
        l = labels(rp(j:j+batch_size-1));
        
        %%%%%%%%%%%%%%
        % feed forward
        %%%%%%%%%%%%%%
        
        %%%% FILL THE BLANK
        % get the output of the first layer with sigmoid activation function
        % the output of the second layer should only be activated with exp()
        f1 = sigmoid(w1 * x);   % 50 x 500 matrix
        f2 = exp(w2 * f1);   % 10 x 500 matrix

        % get the score
        % each row vector should indicate the probability for each digit
        size_f2 = size(f2);
        s = zeros(size(f2));
        for p = 1 : size_f2(2)
            s(:,p) = f2(:,p)./sum(f2(:,p));
        end
        
        %s = ;   % 10 x 500 matrix
        %%%%
        
        % compute the error
        I = sub2ind(size(s), l', 1:batch_size);
        p = zeros(size(s));
        p(I) = 1; % assign 1 only on where 'labels' indicate
        err2 = (s - p)/batch_size;
        
        %%%%%%%%%%%%%%%%%%
        % back propagation
        %%%%%%%%%%%%%%%%%%
        
        grad2_w = err2 * f1';
        grad2_b = sum(err2, 2);
        c1 = w2' * err2;
        err1 = c1 .* f1 .* (1-f1);
        
        grad1_w = err1 * x';
        grad1_b = sum(err1, 2);
        
        %%%% FILL THE BLANK
        % update the weights.
        % each weight and bias should be updated
        % with its gradient and lr_w and lr_b
        
        w2 = w2 - lr_w * grad2_w;
        b2 = b2 - lr_b * grad2_b;
        w1 = w1 - lr_w * grad1_w;
        b1 = b1 - lr_b * grad1_b;
        
        %%%%
    end
    
    % decay
    if ~mod(i, 40)
        lr_w = lr_w * gam;
        lr_b = lr_b * gam;
    end
    
    %%%% FILL THE BLANK
    % record the training loss and accuracy
    loss_list(i) = -sum(log(s(I)))/batch_size;
    [~, pred_labels] = max(s);
    acc_list(i) = sum(pred_labels == l') / length(pred_labels)

    % get the test output
    tf1 = sigmoid(w1 * tdata);
    tf2 = exp(w2 * tf1);
    size_tf2 = size(tf2);
    ts = zeros(size(tf2));
    for q = 1 : size_tf2(2)
        ts(:,q) = tf2(:,q)./sum(tf2(:,q));
    end
    %ts = ;
    %%%%
    
    % get the test result and record it
    I = sub2ind(size(ts), tlabels', 1:num_tdata);
    tloss_list(i) = -sum(log(ts(I)))/num_tdata;
    [~, tpred_labels] = max(ts);
    
    %%%% FILL THE BLANK
    % record the test accuracy
    tacc_list(i) = sum(tpred_labels == tlabels')/length(tpred_labels)
    %%%%
end

plot(1:num_epoch, loss_list, 'b-'); hold on;
plot(1:num_epoch, tloss_list, 'r-');
xlabel('Iteration'); ylabel('loss');
legend('training', 'test'); figure;
plot(1:num_epoch, acc_list, 'b-'); hold on;
plot(1:num_epoch, tacc_list, 'r-');
xlabel('Iteration'); ylabel('accuracy');
legend('training', 'test');
