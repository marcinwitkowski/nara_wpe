function plot_normalised_spectrum(x, fignum, f_vec,t_vec)
if nargin<2
    figure('units','normalized','outerposition',[0 0 1 1]);
else
    h1 = figure(fignum);
    set(h1,'units','normalized','outerposition',[0 0 1 1])
end
X = abs(x);%(:,1:size(x,2)/2));
max_val = max(max(X));
XPow = 20*log10(X.'./max_val);
if nargin<3
    imagesc(XPow);
    xlabel('frame index')
    ylabel('Frequency bin')
else
    imagesc(t_vec,f_vec,XPow);
    xlabel('Time [s]')
    ylabel('Frequency [Hz]')
end
title(['Power Spectrum Normalisation factor: ' num2str(20*log10(max_val)) ' dB']);
caxis([-60, 0]);
ax = gca;
ax.YDir = 'normal';
colorbar()