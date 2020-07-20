
% STFT including zero padding (optional)
% x: multichannel signal (sign_len,nr_mics) [length in samples]
% M: frame length (samples)
% N: fft length (samples), zeropadding for N > M
% R: hop-size (samples)

function [X, f_vec, t] = stft_kkmw(x,M,N,R,fs)

% set defaults
if nargin < 5
    error('Input arguments missing');
end

% pre-allocation
%num_frames = ceil(size(x,1)/R);
num_frames = floor((size(x,1)-M)/R)+1;
N_out = N/2+1;
nr_mics = size(x,2);

xt = zeros(M,nr_mics,'double');
f_vec = (0:N/2) * fs/N;
t = (0:num_frames-1) * R/fs;
X = zeros(N_out,num_frames,nr_mics,'double');

% transform window
%win = repmat(sin(linspace(0,pi,M)).',[1 nr_mics]);
win = repmat(hamming(M),[1 nr_mics]);

% STFT
for kk = 0:num_frames-1
    %xt = [ xt(R+1:end,:); x(kk*R+1:(kk+1)*R,:) ];
    xt = x((kk*R+1):(kk*R+M), :);
    Xt = fft(xt .* win, N);
    X(:,kk+1,1:nr_mics) = Xt(1:N_out,:);
end
