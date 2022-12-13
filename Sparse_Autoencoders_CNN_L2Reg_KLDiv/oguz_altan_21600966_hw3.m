function oguz_altan_21600966_hw3(question)
clc
close all

switch question
    case '1'
        disp('1')
        %% question 1 code goes here
        clear;
        load assign3_data1;
        
        %part a
        gray_data = 0.2126*data(:,:,1,:) + 0.7152*data(:,:,2,:) + 0.0722*data(:,:,3,:);
        flat_gray_data = reshape(gray_data,[256, 10240]);
        remove_pix = flat_gray_data - mean(flat_gray_data);
        
        std_mean = std(remove_pix(:)); %finding std across all pixel in the data
        meanpix = max(min(remove_pix, 3*std_mean), - 3*std_mean) / 3*std_mean;
        normalized_ims = (meanpix + 1) * 0.4 + 0.1;
        norm_in_gray = reshape(normalized_ims,[16,16,10240]); %reshape to matrix 16*16*10240
        rand_patch = randperm(10240);
        
        figure;
        for i = 1:200
            subplot(10,20,i);
            imshow(data(:,:,:,rand_patch(i)));
        end
        
        figure;
        for i = 1:200
            subplot(10,20,i);
            imshow(norm_in_gray(:,:,rand_patch(i)))
        end
        
        %part b
        num_images = 10240; %number of images in the dataset
        params.L_in = 256; %input is 256 pixel image vector
        
        params.L_hid = 64; %by architecture, number of neurons for output and input are same
        params.lambda = 5e-4;
        
        params.rho = 0.01;
        params.beta = 1.5;
        
        options = optimset('MaxIter',250);
        
        W_interval = sqrt(6/(params.L_in + params.L_hid));
        W_hid = - W_interval + (2 * W_interval) .* rand(params.L_in, params.L_hid);
        W_out = - W_interval + (2 * W_interval) .* rand(params.L_hid, params.L_in);
        b_hid =  rand(1,params.L_hid)*2*W_interval-W_interval;
        b_out =  rand(1,params.L_in)*2*W_interval-W_interval;
        
        We = [W_hid(:) ; W_out(:) ; b_hid(:) ; b_out(:)];
        
        costFunction = @(We) aeCost(We,normalized_ims,params);
        [we_opt, cost, ep] = fmincg(costFunction,We,options);
        
        W1 = reshape(we_opt (1:params.L_hid*params.L_in), params.L_in, params.L_hid);
        show_w(W1);
end
end

function [J, Jgrad] = aeCost(We,data,params)

[~,num_images] = size(data);

%packing up weight and bias matrix
W_hid = reshape(We(1 : params.L_in*params.L_hid),params.L_in,params.L_hid);
W_out = reshape(We(params.L_in*params.L_hid+1 : params.L_in*params.L_hid*2),params.L_hid,params.L_in);
b_hid = reshape(We(params.L_in*params.L_hid*2+1 : params.L_in*params.L_hid*2 + params.L_hid),1,params.L_hid);
b_out = reshape(We(params.L_in*params.L_hid*2 + params.L_hid + 1 : size(We)),1,params.L_in);

%feedforwarding network
hid_act = sigmoid(W_hid'*data - b_hid');
out_nn = sigmoid(W_out'*hid_act - b_out');
rho_hat = mean(hid_act,2);

%calculating errors separated into three terms in the cost function
mse = (1/(2*num_images)).*sum(sum((data-out_nn).^2,2));
regul = (params.lambda/2)*(sum(W_hid.^2,'all') + sum(W_out.^2,'all'));
KL_div = params.beta*sum((params.rho*log2(params.rho./rho_hat) + (1-params.rho)*log2((1-params.rho)./(1-rho_hat))));
J = mse + regul + KL_div; %calculating total cost
Jgrad_W_hidden = (-(1/num_images)*(((W_out*((data-out_nn).*((1-out_nn).*out_nn))).*((1-hid_act).*hid_act))*data')+ ...
    params.lambda*W_hid' + params.beta*(1/log(2)).*(-params.rho./rho_hat + (1-params.rho)./(1-rho_hat)).*(1/num_images).*(((1-hid_act).*hid_act)*data'))';
Jgrad_W_out = (-(1/num_images)*(data-out_nn).*((1-out_nn).*out_nn)*hid_act' + params.lambda*W_out')';
Jgrad_b_hid = (-(1/num_images)*((W_out*((data-out_nn).*((1-out_nn).*out_nn))).*((1-hid_act).*hid_act)*(-1*ones(1,num_images))') + ...
    params.beta*(1/log(2)).*(-params.rho./rho_hat + (1-params.rho)./(1-rho_hat)).*(1/num_images).*(((1-hid_act).*hid_act)*(-1*ones(1,num_images))'))';
Jgrad_b_out = (-(1/num_images)*(data-out_nn).*((1-out_nn).*out_nn)*(-1*ones(1,num_images))')';

Jgrad = [Jgrad_W_hidden(:) ; Jgrad_W_out(:) ; Jgrad_b_hid(:) ; Jgrad_b_out(:)]; %rolling the matrices to a single vector
end

%this function takes the weights, adjust the contrast and dimensions of the
%images that the weights represent in the corresponding layer neurons.
function [h, array] = show_w (we)

we = we - mean(we(:));
[r,c] = size(we);
weight_height = sqrt(r);
div = divisors(c);
[~, no_of_div] = size(div);
im_height = div(round(no_of_div/2));
im_width = c/im_height;
patch = zeros(im_height*(weight_height), im_height*(weight_height));
weight_counter = 1;

for i = 1 : im_height
    for j = 1 : im_width
        if weight_counter > c
            continue;
        end
        patch((j-1)*(weight_height)+(1:weight_height), (i-1)*(weight_height)+(1:weight_height)) = reshape(we(:, weight_counter), weight_height,weight_height) / max(abs(we(:, weight_counter)));
        weight_counter = weight_counter + 1;
    end
end

figure;
we = imagesc(patch,[-1 1]);
colormap(gray);
axis image off
drawnow;

end

function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function. Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives. The Polack-
% Ribiere flavour of conjugate gradients is used to compute search directions,
% and a line search using quadratic and cubic polynomial approximations and the
% Wolfe-Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes. Additionally a bunch of checks are made to
% make sure that exploration is taking place and that extrapolation will not
% be unboundedly large. The "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The function returns when either its length is up, or if no further
% progress can be made (ie, we are at a minimum, or so close that due to
% numerical problems, we cannot get any closer). If the function terminates
% within a few iterations, it could be an indication that the function value
% and derivatives are not consistent (ie, there may be a bug in the
% implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%
% See also: checkgrad
%
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
%
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
%
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%
% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
    argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
    i = i + (length>0);                                      % count iterations?!
    
    X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
    X = X + z1*s;                                             % begin line search
    [f2 df2] = eval(argstr);
    i = i + (length<0);                                          % count epochs?!
    d2 = df2'*s;
    f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
    if length>0, M = MAX; else M = min(MAX, -length-i); end
    success = 0; limit = -1;                     % initialize quanteties
    while 1
        while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0)
            limit = z1;                                         % tighten the bracket
            if f2 > f1
                z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
            else
                A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
                B = 3*(f3-f2)-z3*(d3+2*d2);
                z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
            end
            if isnan(z2) | isinf(z2)
                z2 = z3/2;                  % if we had a numerical problem then bisect
            end
            z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
            z1 = z1 + z2;                                           % update the step
            X = X + z2*s;
            [f2 df2] = eval(argstr);
            M = M - 1; i = i + (length<0);                           % count epochs?!
            d2 = df2'*s;
            z3 = z3-z2;                    % z3 is now relative to the location of z2
        end
        if f2 > f1+z1*RHO*d1 | d2 > -SIG*d1
            break;                                                % this is a failure
        elseif d2 > SIG*d1
            success = 1; break;                                             % success
        elseif M == 0
            break;                                                          % failure
        end
        A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
        if ~isreal(z2) | isnan(z2) | isinf(z2) | z2 < 0   % num prob or wrong sign?
            if limit < -0.5                               % if we have no upper limit
                z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
            else
                z2 = (limit-z1)/2;                                   % otherwise bisect
            end
        elseif (limit > -0.5) & (z2+z1 > limit)          % extraplation beyond max?
            z2 = (limit-z1)/2;                                               % bisect
        elseif (limit < -0.5) & (z2+z1 > z1*EXT)       % extrapolation beyond limit
            z2 = z1*(EXT-1.0);                           % set to extrapolation limit
        elseif z2 < -z3*INT
            z2 = -z3*INT;
        elseif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT))   % too close to limit?
            z2 = (limit-z1)*(1.0-INT);
        end
        f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
        z1 = z1 + z2; X = X + z2*s;                      % update current estimates
        [f2 df2] = eval(argstr);
        M = M - 1; i = i + (length<0);                             % count epochs?!
        d2 = df2'*s;
    end                                                      % end of line search
    
    if success                                         % if line search succeeded
        f1 = f2; fX = [fX' f1]';
        %     fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
        s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
        tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
        d2 = df1'*s;
        if d2 > 0                                      % new slope must be negative
            s = -df1;                              % otherwise use steepest direction
            d2 = -s'*s;
        end
        z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
        d1 = d2;
        ls_failed = 0;                              % this line search did not fail
    else
        X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
        if ls_failed | i > abs(length)          % line search failed twice in a row
            break;                             % or we ran out of time, so we give up
        end
        tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
        s = -df1;                                                    % try steepest
        d1 = -s'*s;
        z1 = 1/(1-d1);
        ls_failed = 1;                                    % this line search failed
    end
    %   if exist('OCTAVE_VERSION')
    %     fflush(stdout);
    %   end
    disp("Iteration " + i);
end
%plot(1:length(fX),fX);
% fprintf('\n');
end

function sig = sigmoid(x)
sig = 1 ./ (1+exp(-x));
end
