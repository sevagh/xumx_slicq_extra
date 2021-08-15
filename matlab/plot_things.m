init;

set(0, 'DefaultAxesFontSize', 20);

[sig, fs] = gspi;

siglen = size(sig, 1);

duration = siglen/fs;

dT = 1/fs;
t = (0.0:dT:duration-dT)';

plot(t, sig);
grid on;
xlabel('time (s)');
ylabel('amplitude');
title('Glockenspiel waveform');

N = 2048;
y = fft(sig, N);                               % Compute DFT of x
m = abs(y);                               % Magnitude
y(m<1e-6) = 0;
p = unwrap(angle(y));                     % Phase

df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies

subplot(2,1,1)
plot(f,m)
title('Magnitude DFT');
xlabel('frequency (Hz)');
ylabel('|DFT|');

subplot(2,1,2)
plot(f,p*180/pi)
title('Phase DFT')
xlabel('frequency (Hz)');
ylabel('\angle DFT');

smallWin = 128;
midWin = 2048;
bigWin = 16384;

figure;
smallGauss=gausswin(smallWin);
spectrogram(sig,smallGauss,smallWin/2,smallWin*2,fs,"yaxis");
title("Glockenspiel, gausswin = 128");

figure;
midGauss=gausswin(midWin);
spectrogram(sig,midGauss,midWin/2,midWin*2,fs,"yaxis");
title("Glockenspiel, gausswin = 2048");

figure;
bigGauss=gausswin(bigWin);
spectrogram(sig,bigGauss,bigWin/2,bigWin*2,fs,"yaxis");
title("Glockenspiel, gausswin = 16384");

figure;
smallHamm=hamming(smallWin);
spectrogram(sig,smallHamm,smallWin/2,smallWin*2,fs,"yaxis");
title("Glockenspiel, hammwin = 128");

figure;
midHamm=hamming(midWin);
spectrogram(sig,midHamm,midWin/2,midWin*2,fs,"yaxis");
title("Glockenspiel, hammwin = 2048");

figure;
bigHamm=hamming(bigWin);
spectrogram(sig,bigHamm,bigWin/2,bigWin*2,fs,"yaxis");
title("Glockenspiel, hammwin = 16384");