
% ISTFT for zero padded signals (optional)
%
% M: frame length (samples)
% N: fft length (samples), zeropadding for N > M
% R: hop-size (samples)
%

function x =  istft_kkmw(X,M,N,R)

% set defaults
if nargin < 4
    error('Input arguments missing');
end

% restore the negative spectrum
X = [ X; conj(X(size(X,1)-1:-1:2,:,:)) ];

% pre-allocation
out = zeros(M,1,size(X,3));
x = zeros(size(X,2)*R,size(X,3));

% transform window
%win = repmat(sin(linspace(0,pi,M)).',[1 1 size(X,3)] );
win = repmat(tukeywin(M),[1 1 size(X,3)]);

% ISTFT
for kk = 1:size(X,2)
    xt = ifft(X(:,kk,:),N);
    out = [ out(R+1:end,1,:); zeros(R,1,size(X,3)) ] + win .* xt(1:M,:,:);
    x( (kk-1) * R+1 : kk * R, : ) = out(1:R,1,:);
end



