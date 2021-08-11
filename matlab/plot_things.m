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