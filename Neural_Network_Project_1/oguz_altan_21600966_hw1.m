function oguz_altan_21600966_hw1(question)
clc
close all

switch question
    case '1'
        disp('1')
        %% question 1 code goes here
        disp('The answer of this question is analytical and can be found on the report.');
        
    case '2'
        disp('2')
        %% question 2 code goes here
        
        %hidden layer (AND gate) input weights and bias
        w1 = [2 0 2 2];
        w2 = [0 -1 1 2];
        w3 = [-2 2 -1 0];
        w4 = [-1 2 0 -1];
        theta = [4.5 2.25 1.75 1.25];
        W_hidden = [w1;w2;w3;w4];
        
        disp('hidden layer input weights matrix is: ');
        disp(W_hidden);
        disp('hidden layer bias weight vector is: ');
        disp(theta);
        
        %output layer (OR gate) input weigths and bias
        W_or = [1 1 1 1];
        theta_or = 0.75;
        
        disp('output layer input weights vector is: ')
        disp(W_or);
        disp('output layer bias weight is: ');
        disp(theta_or);
        
        bina_input = decimalToBinaryVector(0:15); %creates matrix of binary equivalents of 0 to 15
        disp('Matrix of binary equivalents of decimal 0 to 15: ');
        disp(bina_input);
        
        %calculates the network output: hidden layer takes weights and theta and uses
        %weighted sum minus bias and puts the result in activation
        %function. Output layer makes similar calculation using OR layer
        %weights, which are outputs of hidden layer and again puts the result into activation function. The
        %result is nn_out
        nn_out =  unitStep(W_or*unitStep(W_hidden*bina_input'-theta')-theta_or);
        disp('The output of the network taking inputs in binary from 0 to 15 is the vector: ')
        disp(nn_out)
        
        %generate 400 input samples by concatenating 25 samples from each input vector
        %and adds gaussian noise with std of 0.2
        std = 0.2;
        new_X = repmat(bina_input',1,25);
        noise = std*randn(4,400);
        X_noise = new_X + noise;
        disp('New X matrix with Gaussian noise:');
        disp(X_noise);
        
        %not robust network
        %calculates network output using the same way that is explained for
        %nn_out
        nn_out_tt =  unitStep(W_or*unitStep(W_hidden*new_X-theta')-theta_or);
        nn_out_noised =  unitStep(W_or*unitStep(W_hidden*X_noise-theta')-theta_or);
        notrobustcorrect = sum(nn_out_tt == nn_out_noised)/4; %compares the original vs noised network
        disp('Correctness percentage of not robust network is: ')
        disp(notrobustcorrect);
        
        %robust network
        robust_hidden_theta = [5 2.5 1.5 1.5];
        robust_theta_or = 0.5;
        
        nn_out_tt_robust =  unitStep(W_or*unitStep(W_hidden*new_X-robust_hidden_theta')-robust_theta_or);
        nn_out_noised_robust =  unitStep(W_or*unitStep(W_hidden*X_noise-robust_hidden_theta')-robust_theta_or);
        
        %compares the original vs noised network and divides the number of times
        %that they are equal by 4 to find the correctness percentage
        robustcorrect = sum(nn_out_tt_robust == nn_out_noised_robust)/4;
        
        disp('Correctness percentage of robust network is: ')
        disp(robustcorrect);
        
    case '3'
        disp('3')
        %% question 3 code goes here
        load('assign1_data1.mat')
        
        %part a
        %find random indexes for each class
        rand_vector = [];
        for i = 1:26
            rand_vector = [rand_vector randi([(i-1)*200 i*200])];
        end
        rand_vector;
        
        %print sample images for each class
        for i = 1:26
            figure;
            image(trainims(:,:,rand_vector(i)));
        end
        figure;
        image(trainims(:,:,randi([5000 5200])))
        
        %construct correlation matrix for each pair of images
        bina_input = [];
        
        for i = 1:26
            matt = trainims(:,:,rand_vector(i));
            col_vec = matt(:); %reshapes the image matrix into a column vector
            bina_input = [bina_input col_vec];
        end
        
        bina_input = double(bina_input);
        corr_matrix = corrcoef(bina_input);
        disp('The correlation coefficient matrix: ');
        disp(corr_matrix);
        
        imagesc(corr_matrix);
        title('Correlation Coefficient Matrix');
        
        %using one hot encoding for labels to make them appropriate
        %to use in error calculation
        onehot = zeros(26,5200);
        for i = 1:5200
            onehot(trainlbls(i),i) = 1;
        end
        
        %constructing weights and learning rate (nu) matrices
        std = 0.1;
        W = 0.1*randn(785,26); %Gaussian noise weights
        nu_opt = 0.1; %learning rate
        W_low = W;
        W_high = W;
        loss_vec = [];
        
        %10000 iterations and updates using gradient descent
        for i = 1:10000
            rand_image_index = randi([1 5200]); %randomly selects an index in [1 5200]
            trainims_in_matrix = trainims(:,:,rand_image_index); %creates image matrix from trainims dataset
            x = double(trainims_in_matrix(:)); %casting uint8 to double
            x = [x ;-1]; %adding bias to the weights matrix
            x = x/255; %normalization
            v = W'*x ; %weighted sum
            y = sigmoid(v); %applying activation function to v
            der_act = sigmoid(v) .* (1 - sigmoid(v)); %derivative of the sigmoid function
            error = (onehot(:,rand_image_index)-y); %difference between the desired and realized output of the network
            grad = x * (error .* der_act)'; %gradient for gradient descent
            W = W + nu_opt * grad; %updating weights matrix
            loss = 1/2 * error' * error; %Mean Square Error
            loss_vec = [loss_vec loss]; %storing loss for each iteration into a vector
        end
        
        figure;
        for k = 1:26
            subplot(6,5,k),imshow(reshape(W(2:785,k),[28 28]), [min(W(2:785,k)) max(W(2:785,k))]);
        end
        
        %testing the trained network using test images
        W_test = W;
        correct = 0;
        output_tot = [];
        max_value = [];
        
        for i = 1:1300
            rand_image_index = i; %randomly selects an index in [1 5200]
            testims_in_matrix = testims(:,:,rand_image_index);
            x_test = double(testims_in_matrix(:)); %casting uint8 to double
            x_test = [x_test;-1]; %adding bias
            x_test = x_test/255;
            output_test = sigmoid(W_test.'*x_test);
            [max_value, label_index] = max(output_test);
            if (testlbls(i) == label_index)
                correct = correct+1;
            end
        end
        fprintf('The correctness for optimal learning rate nu 0.1 is = %f\n',correct/1300*100);
        
        %%%%%%%%%%%%%%%%%% lower learning rate nu
        nu_low = 0.001;
        loss_vec_low = [];
        for i = 1:10000
            rand_image_index = randi([1 5200]); %randomly selects an index in [1 5200]
            trainims_in_matrix = trainims(:,:,rand_image_index); %creates image matrix from trainims dataset
            x = double(trainims_in_matrix(:)); %casting uint8 to double
            x = [x ;-1]; %adding bias to the weights matrix
            x = x/255; %normalization
            v = W_low'*x ; %weighted sum
            y = sigmoid(v); %applying activation function to v
            der_act = sigmoid(v) .* (1 - sigmoid(v)); %derivative of the sigmoid function
            error = (onehot(:,rand_image_index)-y); %difference between the desired and realized output of the network
            grad = x * (error .* der_act)'; %gradient for gradient descent
            W_low = W_low + nu_low * grad; %updating weights matrix
            loss_low = 1/2 * error' * error; %Mean Squarred Error
            loss_vec_low = [loss_vec_low loss_low]; %storing loss for each iteration into a vector
        end
        
        W_test_low = W_low;
        correct_low = 0;
        output_tot_low = [];
        max_value_low = [];
        for i = 1:1300
            rand_image_index = i;
            testims_in_matrix = testims(:,:,rand_image_index);
            x_test = double(testims_in_matrix(:));
            x_test = [x_test;-1];
            x_test = x_test/255;
            output_test = sigmoid(W_test_low.'*x_test);
            [max_value_low, label_index] = max(output_test);
            if (testlbls(i) == label_index)
                correct_low = correct_low+1;
            end
        end
        fprintf('The correctness for low learning rate nu 0.001 is = %f\n',correct_low/1300*100);
        
        
        %%%%%%%%%%%%%%%%%% higher learning rate nu
        nu_high = 0.9;
        loss_vec_high = [];
        for i = 1:10000
            rand_image_index = randi([1 5200]); %randomly selects an index in [1 5200]
            trainims_in_matrix = trainims(:,:,rand_image_index); %creates image matrix from trainims dataset
            x = double(trainims_in_matrix(:)); %casting uint8 to double
            x = [x ;-1]; %adding bias to the weights matrix
            x = x/255; %normalization
            v = W_high'*x ; %weighted sum
            y = sigmoid(v); %applying activation function to v
            der_act = sigmoid(v) .* (1 - sigmoid(v)); %derivative of the sigmoid function
            error = (onehot(:,rand_image_index)-y); %difference between the desired and realized output of the network
            grad = x * (error .* der_act)'; %gradient for gradient descent
            W_high = W_high + nu_high * grad; %updating weights matrix
            loss_high = 1/2 * error' * error; %Mean Squarred Error
            loss_vec_high = [loss_vec_high loss_high]; %storing loss for each iteration into a vector
        end
        
        W_test_high= W_high;
        correct_high = 0;
        output_tot_high = [];
        max_value_high = [];
        
        for i = 1:1300
            rand_image_index = i;
            testims_in_matrix = testims(:,:,rand_image_index);
            x_test = double(testims_in_matrix(:));
            x_test = [x_test;-1];
            x_test = x_test/255;
            output_test = sigmoid(W_test_high.'*x_test);
            [max_value_high, label_index] = max(output_test);
            if (testlbls(i) == label_index)
                correct_high = correct_high+1;
            end
        end
        fprintf('The correctness for high learning rate nu 0.9 is = %f\n',correct_high/1300*100);
        
        %plotting loss functions for three different learning rates on the
        %same figure to compare easily
        iter = [1:10000];
        figure;
        plot(iter,loss_vec);
        hold on;
        plot(iter,loss_vec_low);
        plot(iter,loss_vec_high);
        title('Plot of Loss Function');
        legend('optimal learning rate 0.1', 'low learning rate 0.01', 'high learning rate 0.9');
        xlabel('Iteration');
        ylabel('Loss Function J');
        
end
end

function outStep = unitStep(t)
outStep = t >= 0;
end

function sigmoidOut = sigmoid(v)
sigmoidOut = 1./(1+exp(-v));
end