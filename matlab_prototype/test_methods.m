clearvars; close all;
% Load multichannel audio
Nmic = 4;
for i=1:Nmic
    [x(:,i),fs] = audioread(['../data/AMI_WSJ20-Array1-' num2str(i) '_T10c0201.wav']);
end

cfgs = struct();
cfgs.fs = fs;                           % sampling rate  
cfgs.method = 'wpe';                    % Dereverberation method method ('wpe' (default), 'cgg', 'admm' or 'none')
cfgs.num_mic = Nmic;                    % number of microphones   
cfgs.num_out = 1;
cfgs.K = 512;                            % the number of FFT bins (subbands)
cfgs.winsize = 320;                      % analysis window size in samples
cfgs.winshift = 160;                     % analysis window shift in samples
cfgs.epsilon = 1e-10;                    % lower bound of desired signal spectral variance
cfgs.D = 3;                              % subband preditction delay >0 (1 - classic no delay LP)                     
cfgs.Lc = 20;                            % subband prediction order 
cfgs.eta = 0;                            % parameters for smoothing(0 (default) - original, (0-1) simple smoothing, {1,2,..N} - neighborhood smoothing
cfgs.max_iterations = 3;                 % number of iteration of EM algorithm

tic; y1 = fdndlp_vagh(x,cfgs); toc;

cfgs.method = 'cgg';
cfgs.pp = 0.5;                           % parameter of CGG WPE as d^(2-pp), set pp=0 to obtain classic WPE

tic; y2 = fdndlp_vagh(x,cfgs); toc;

cfgs.method = 'admm';
cfgs.admm_cfgs.Gmin = 0.04;%0.01;
cfgs.admm_cfgs.vareps = 1e-10;%1e-8;
cfgs.admm_cfgs.rho = 1;%0.001;
cfgs.admm_cfgs.gamma = 1;%1.6;
cfgs.admm_cfgs.num_ADM_iter = 20;%40

tic; y3 = fdndlp_vagh(x,cfgs); toc;

%% Plot STFTs
y_ref = x(1:length(y1),1);
[Y,f_vec, t_vec] = stft_kkmw([y_ref, y1 y2 y3], cfgs.winsize, cfgs.K, cfgs.winshift, fs);
titles = {'Input Signal', 'WPE', 'CGG', 'ADMM'};
for i = 1:size(Y,3)
    plot_normalised_spectrum(squeeze(Y(:,:,i)).', i, f_vec, t_vec); title(titles(i));
end

