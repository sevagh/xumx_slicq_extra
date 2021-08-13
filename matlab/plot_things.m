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

N = siglen;
y = fft(sig, N);                               % Compute DFT of x
m = abs(y);                               % Magnitude
y(m<1e-6) = 0;
p = unwrap(angle(y));                     % Phase

df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies

subplot(3,1,1)
plot(y, 'o')
title('Complex DFT')
xlabel('real');
ylabel('imaginary');

subplot(3,1,2)
plot(f,m)
title('Magnitude DFT');
xlabel('frequency (Hz)');
ylabel('|DFT|');

subplot(3,1,3)
plot(f,p*180/pi)
title('Phase DFT')
xlabel('frequency (Hz)');
ylabel('\angle DFT');
