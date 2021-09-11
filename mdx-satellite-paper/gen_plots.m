[x, fs] = audioread('./static-assets/gspi.wav');

colormap inferno;

figure;
spectrogram(x,4096,1024,4096,fs,'yaxis');
