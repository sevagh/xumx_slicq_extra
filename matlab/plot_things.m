init;

set(0, 'DefaultAxesFontSize', 34);
set(0,'DefaultFigureColormap', ltfat_inferno);
dpiVar = 100;
gcfPosition = [1 1 1920 1080];

[sig, fs] = gspi;

siglen = size(sig, 1);

duration = siglen/fs;

dT = 1/fs;
t = (0.0:dT:duration-dT)';

% figure;
% plot(t, sig);
% grid on;
% xlabel('time (s)');
% ylabel('amplitude');
% title('Glockenspiel waveform','FontWeight','Normal');
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_waveform.png","Resolution",dpiVar);

N = 2048;
y = fft(sig, N);                               % Compute DFT of x
m = abs(y);                               % Magnitude
y(m<1e-6) = 0;
p = unwrap(angle(y));                     % Phase

df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies

figure;
subplot(2,1,1)
plot(f,m)
title(sprintf('Magnitude DFT, %d points', N),'FontWeight','Normal');
grid on;
xlabel('frequency (Hz)');
ylabel('|DFT|');
set(gca, 'FontSize', 21); % smaller font on DFTs

subplot(2,1,2)
plot(f,p*180/pi)
title(sprintf('Phase DFT, %d points', N),'FontWeight','Normal');
grid on;
xlabel('frequency (Hz)');
ylabel('\angle DFT');

set(gcf, 'Position', gcfPosition);
set(gca, 'FontSize', 21); % smaller font on DFTs
exportgraphics(gcf,sprintf("../latex/images-gspi/glock_dft_%d.png", N),"Resolution",dpiVar);

N = siglen;
y = fft(sig, N);                               % Compute DFT of x
m = abs(y);                               % Magnitude
y(m<1e-6) = 0;
p = unwrap(angle(y));                     % Phase

df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies

figure;
subplot(2,1,1)
plot(f,m)
title(sprintf('Magnitude DFT, %d points', N),'FontWeight','Normal');
grid on;
xlabel('frequency (Hz)');
ylabel('|DFT|');
set(gca, 'FontSize', 21); % smaller font on DFTs

subplot(2,1,2)
plot(f,p*180/pi)
title(sprintf('Phase DFT, %d points', N),'FontWeight','Normal');
grid on;
xlabel('frequency (Hz)');
ylabel('\angle DFT');

set(gcf, 'Position', gcfPosition);
set(gca, 'FontSize', 21); % smaller font on DFTs
exportgraphics(gcf,sprintf("../latex/images-gspi/glock_dft_%d.png", N),"Resolution",dpiVar);

% smallWin = 128;
% midWin = 2048;
% bigWin = 16384;
% 
% figure;
% smallGauss=gausswin(smallWin);
% spectrogram(sig,smallGauss,smallWin/2,smallWin*2,fs,"yaxis");
% title("Glockenspiel, gausswin = 128",'FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,sprintf("../latex/images-gspi/glock_gauss_%d.png", smallWin),"Resolution",dpiVar);
% 
% figure;
% midGauss=gausswin(midWin);
% spectrogram(sig,midGauss,midWin/2,midWin*2,fs,"yaxis");
% title("Glockenspiel, gausswin = 2048",'FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,sprintf("../latex/images-gspi/glock_gauss_%d.png", midWin),"Resolution",dpiVar);
% 
% figure;
% bigGauss=gausswin(bigWin);
% spectrogram(sig,bigGauss,bigWin/2,bigWin*2,fs,"yaxis");
% title("Glockenspiel, gausswin = 16384",'FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,sprintf("../latex/images-gspi/glock_gauss_%d.png", bigWin),"Resolution",dpiVar);
% 
% figure;
% smallHamm=hamming(smallWin);
% spectrogram(sig,smallHamm,smallWin/2,smallWin*2,fs,"yaxis");
% title("Glockenspiel, hammwin = 128",'FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,sprintf("../latex/images-gspi/glock_hamm_%d.png", smallWin),"Resolution",dpiVar);
% 
% figure;
% midHamm=hamming(midWin);
% spectrogram(sig,midHamm,midWin/2,midWin*2,fs,"yaxis");
% title("Glockenspiel, hammwin = 2048",'FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,sprintf("../latex/images-gspi/glock_hamm_%d.png", midWin),"Resolution",dpiVar);
% 
% figure;
% bigHamm=hamming(bigWin);
% spectrogram(sig,bigHamm,bigWin/2,bigWin*2,fs,"yaxis");
% title("Glockenspiel, hammwin = 16384",'FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,sprintf("../latex/images-gspi/glock_hamm_%d.png", bigWin),"Resolution",dpiVar);

figure;
plot(hamming(2048)); hold on; plot(gausswin(2048)); legend('Hamming window', 'Gaussian window');
title('2048-point windows; Hamming vs. Gaussian','FontWeight','Normal');
xlim([-64 2048+64]);
grid on;

set(gcf, 'Position', gcfPosition);
exportgraphics(gcf,"../latex/images-tftheory/hamming_vs_gauss.png","Resolution",dpiVar);

% figure;
% spectrogram(sig,4096,1024,4096,fs,'yaxis');
% title('Magnitude STFT, window=4096','FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_stft_4096.png","Resolution",dpiVar);
% 
% figure;
% spectrogram(sig,1024,256,1024,fs,'yaxis');
% title('Magnitude STFT, window=1024','FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_stft_1024.png","Resolution",dpiVar);
% 
% figure;
% spectrogram(sig,256,64,256,fs,'yaxis');
% title('Magnitude STFT, window=256','FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_stft_256.png","Resolution",dpiVar);
% 
% figure;
% cqt(sig,'SamplingFrequency',fs,'BinsPerOctave',12);
% title('Magnitude CQT/CQ-NSGT, 12 bins-per-octave','FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_cqt12.png","Resolution",dpiVar);
% 
% figure;
% cqt(sig,'SamplingFrequency',fs,'BinsPerOctave',24);
% title('Magnitude CQT/CQ-NSGT, 24 bins-per-octave','FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_cqt24.png","Resolution",dpiVar);
% 
% figure;
% cqt(sig,'SamplingFrequency',fs,'BinsPerOctave',48);
% title('Magnitude CQT/CQ-NSGT, 48 bins-per-octave','FontWeight','Normal');
% 
% set(gcf, 'Position', gcfPosition);
% exportgraphics(gcf,"../latex/images-gspi/glock_cqt48.png","Resolution",dpiVar);

close all;
