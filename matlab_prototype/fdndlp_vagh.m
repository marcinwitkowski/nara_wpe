function [y, dk, f_vec, t_vec, ck] = fdndlp_vagh(x, cfgs, varargin)
% ============================================================================= 
% This program is an implementation of Variance-Normalizied Delayed Linear
% Prediction in time-frequency domain, which is aimed at speech
% dereverberation, known as weighted prediction error (WPE) method.
% 
% Main parameters:
% mic_num                  the number of channels
% K                        the number of subbands
% F                        over-sampling rate
% N                        decimation factor
% D                        subband preditction delay
% Lc                       subband prediction order 
% epsilon                  lower bound of normalizaton factor
% pp					   parameter of rho calculation as d^(2-pp), set pp=0 to ogtain classic WPE
% eta					   parameter for spectral smoothing, set eta=0 to disable smoothing
% fs                       sampling rate
%
% References:
% [1] Nakatani T, Yoshioka T, Kinoshita K, et al. Speech Dereverberation 
%     Based on Variance-Normalized Delayed Linear Prediction[J]. IEEE 
%     Transactions on Audio Speech & Language Processing, 2010, 18(7):1717-1731.
%
% ============================================================================= 
% Created by Teng Xiang at 2017-10-14, access: https://github.com/helianvine/fdndlp (access 2019-06-12)
% ============================================================================= 

cfgs2var 

if exist('varargin', 'var')
    for ii = 1 : 2 : length(varargin)
        eval([varargin{ii}, '= varargin{ii+1};'])
    end
end

sig_channels = size(x, 2);
if sig_channels > num_mic
   x = x(:,1:num_mic);
   fprintf('Only the first %d channels of input data are used\n\n', num_mic)
elseif sig_channels < num_mic
    error('The channels of input does not match the channel setting');        
end

if strcmp(method, 'wpe')
    cfgs.pp = 0;
	wpe_func = @(xk) cgg_wpe(xk, cfgs);
elseif strcmp(method, 'cgg')
	wpe_func = @(xk) cgg_wpe(xk, cfgs);
elseif strcmp(method, 'admm')
    wpe_func = @(xk) admm_wpe(xk, cfgs);
elseif strcmp(method, 'none')
    y = x;
    [dk, f_vec, t_vec] = stft_kkmw(x,winsize,K,winshift,fs);
    ck = NaN;
    return
else
	error('Undefined WPE. Set dist_method to "wpe", "cgg" or "admm"');
end
[xk, f_vec, t_vec] = stft_kkmw(x,winsize,K,winshift,fs);
[dk, ck] = wpe_func(permute(xk,[2,1,3]));
y = istft_kkmw(dk.', winsize,K,winshift);
end

function [dk, ck] = cgg_wpe(xk, cfgs)
    cfgs2var

    if isscalar(D)
        D = repmat(D,1,K/2+1);
    end
    if isscalar(Lc)
        Lc = repmat(Lc,1,K/2+1);
    end
    
    smooth_func = validate_eta(cfgs);

    N = size(xk, 1);
    dk = zeros(N, K, num_out);
    
    for k = 1 : K/2 + 1
        xk_tmp = zeros(N+Lc(k), num_mic);
        xk_tmp(Lc(k)+1:end,:) = squeeze(xk(:,k,:));
        x_buf = xk_tmp(Lc(k)+1:end,1:num_out);
        X_D = zeros(N,num_mic * Lc(k));
        for ii = 1 : N-D(k)
            x_D = xk_tmp(ii+Lc(k):-1:ii+1,:).';
            X_D(ii+D(k),:) = x_D(:).'; 
        end 

       % initialise
        dk(:,k,:) = x_buf(:,1:num_out);

        sigma_d2 = abs(x_buf(:,1:num_out)).^(2-pp);
        sigma_d2 = smooth_func(sigma_d2);

        i = 1;
        dd = Inf;
        d = x_buf(:,1:num_out);
        while norm(abs(dd))>1e-3 && i < max_iterations +1      
            D_sigma_d2 = diag(1 ./ sigma_d2);
            Phi = X_D'*D_sigma_d2*X_D;
            p = X_D'*D_sigma_d2*x_buf;

            c = pinv(Phi)*p;
            %c = lsqminnorm(Phi,p);
            d_old = d;
            d = x_buf - X_D*c; 
            dd = d_old - d;

            sigma_d2 = max(mean(squeeze(abs(d).^(2-pp)),2), epsilon);
            sigma_d2 = smooth_func(sigma_d2);
            
            i = i+1;
        end
        ck(:,k) = c;
        dk(:,k,:) = d;
    end
    dk(:,K/2+2:end,:) = conj(dk(:,K/2:-1:2,:));
end

function [dk , ck] = admm_wpe(xk, cfgs)
    % implements Narrowband Generalized dereverberation framework desribed in 
    % Jukic, Ante, et al. "A general framework for incorporating time�frequency
    % domain sparsity in multichannel speech dereverberation." 
    % Journal of the Audio Engineering Society 65.1/2 (2017): 17-30.
    cfgs2var 

    if isscalar(D)
        D = repmat(D,1,K/2+1);
    end
    if isscalar(Lc)
        Lc = repmat(Lc,1,K/2+1);
    end
    
    Gmin = admm_cfgs.Gmin;
    vareps = admm_cfgs.vareps;
    rho = admm_cfgs.rho ;
    gamma = admm_cfgs.gamma;
    num_ADM_iter = admm_cfgs.num_ADM_iter;

    N = size(xk, 1);
    dk = zeros(N, K, num_out);

    for k = 1 : K/2 + 1
        xk_tmp = zeros(N+Lc(k), num_mic);
        xk_tmp(Lc(k)+1:end,:) = squeeze(xk(:,k,:));
        x_buf = xk_tmp(Lc(k)+1:end,1:num_out);
        X_D = zeros(N,num_mic * Lc(k));
        for ii = 1 : N-D(k)
            x_D = xk_tmp(ii+Lc(k):-1:ii+1,:).';
            X_D(ii+D(k),:) = x_D(:).'; 
        end  

        % initialise dk and ck with single original WPE iteration (not
        % commented in the article)
        dk(:,k,:) = x_buf(:,1:num_out);
        c = zeros(Lc(k)*num_mic,1);
        dd = Inf;  %differencials of d
        i = 1;    
        mu = zeros(N,1)+1i*zeros(N,1);   
        XXX = pinv(X_D'*X_D)*X_D';
        gl2 = XXX*x_buf;  %iteration independent term
        d = dk(:,k,:);
        while  i < num_ADM_iter + 1  && max(abs(dd)./abs(d))>1e-3          
            d = x_buf - X_D*c - mu; % should be commented according to the article
            d_old = d;
            
            if cfgs.eta > 0
                sigma_d2 = nb_smooth(abs(d).^2, cfgs.eta)+vareps;
                w = 1./(sigma_d2.^(0.5)); % should be w = vareps ./(...) according to the article
            elseif cfgs.eta == 0
                sigma_d2 = abs(d) + vareps;
                w = 1./(sigma_d2); % should be w = vareps ./(...) according to the article
            end  
            h = d; % should be h = x_buf - X_D*c - mu; according to the article
            sp1 = 1 - (w./rho)./(abs(h)+eps);            
            d = max(sp1, Gmin) .* h;
            gi = XXX * (d + mu);        
            c = gl2 - gi;
            dd = d_old - d;
            mu = mu + gamma * (d + X_D*c - x_buf);
            i = i+1;
        end  
        ck(:,k) = c;
        dk(:,k,:) =  d; %x_buf - X_D*ck -mu;
    end
    dk(:,K/2+2:end,:) = conj(dk(:,K/2:-1:2,:));
end

function smooth_func = validate_eta(cfgs) 
    if cfgs.eta>=1 && floor(cfgs.eta)==cfgs.eta
        smooth_func = @(x) nb_smooth(x, cfgs.eta);
    elseif cfgs.eta<1 && cfgs.eta >0
        smooth_func = @(x) simple_smooth(x,cfgs.eta);
    elseif cfgs.eta==0
        smooth_func = @(x) x;
    else
         error('Wrong eta setting. Set eta to:\n%s\n%s\n%s',...
             'value {0}          - no smoothing',...
             'range (0;1)        - simple smoothing eta*x(n-1) + (1-eta)*x(n)',...
             'values {1,2,3,...} - neighborhood smoothing');
    end
end

function x = simple_smooth(x, eta)
    %apllies smoothing x'(n) = eta*x(n-1) + (1-eta)*x(n)
    x = eta*[0;x(1:end-1)] + (1-eta)*x;
end

function x = nb_smooth(x, nb)
    %apllies smoothing x'(n) = 1/(2*nb+1)*(x(n-nb)+ ... + x(n) + ... + x(n+nb))
    N = length(x);
    x_temp = [zeros(nb,1); x ; zeros(nb,1)];
    Nnb = 2*nb+1;
    Nd = zeros(N, Nnb);
    for in = 1:N
        Nd(in,:)=x_temp(in:in+Nnb-1).';
    end
    weights = ones(Nnb,1)*(1/Nnb);
    x = Nd*weights;
end
