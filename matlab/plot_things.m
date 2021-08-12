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

y = fft(sig);                               % Compute DFT of x
m = abs(y);                               % Magnitude
y(m<1e-6) = 0;
p = unwrap(angle(y));                     % Phase

f = (0:length(y)-1)*100/length(y);        % Frequency vector

subplot(2,1,1)
plot(f,m)
title('Magnitude')

subplot(2,1,2)
plot(f,p*180/pi)
title('Phase')
