clearvars; close all;

Nmic = 4;

% Load multichannel audio
for i=1:Nmic
    [x1(:,i),fs] = audioread(['../data/AMI_WSJ20-Array1-' num2str(i) '_T10c0201.wav']);
end

% Simulate multichannel audio
x2_clean = audioread('../data/DR3_FEME0_SX335.WAV'); % from TIMIT
load('../data/sim_4ch_ir.mat','h');
x2 = fftfilt(h.entire(:,1:Nmic), x2_clean);
x2_dir = fftfilt(h.direct(:,1:Nmic), x2_clean);

% Choose signal
x = x2;

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

%% Calculate metrics 
[pesq_ref, cd_ref, fwsegsnr_ref, srmr_ref] = calculate_metrics(x2(:,1), x2_dir(:,1),fs);
[pesq(1), cd(1), fwsegsnr(1), srmr(1)] = calculate_metrics(y1(:,1), x2_dir(:,1),fs);
[pesq(2), cd(2), fwsegsnr(2), srmr(2)] = calculate_metrics(y2(:,1), x2_dir(:,1),fs);
[pesq(3), cd(3), fwsegsnr(3), srmr(3)] = calculate_metrics(y3(:,1), x2_dir(:,1),fs);
