function [pesq, cd, fwsegsnr_val, srmr] = calculate_metrics(s,s_dir,fs)
    addpath(genpath('metrics'));
    
    % hyperparmaters for metrics
    param_fwsegsnr = struct('frame'  , 0.02, ...
                             'shift'  , 0.01, ...
                             'window' , @hanning, ...
                             'numband', 23);

    param_cd = struct('frame' , 0.02   , ...
                      'shift' , 0.01    , ...
                      'window', @hanning, ...
                      'order' , 24      , ...
                      'timdif', 0.0     , ...
                      'cmn'   , 'y');
                  
    % truncate longer signal to shorter one
    L = min(length(s), length(s_dir));
    s = s(1:L);
    s_dir = s_dir(1:L);
    
    % calc metrics
    pesq = pesq_mex(s_dir, s, fs);
    fwsegsnr_val = fwsegsnr(s, s_dir, fs, param_fwsegsnr);
    cd = cepsdist(s, s_dir, fs, param_cd);
    srmr = SRMR(s,fs);
end