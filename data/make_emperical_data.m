S = embedded_process_data('e2_5050');
kt = 0.0255;
R = 36;
radius = 0.006875;
fs = 1000;

fc = 3;
[b,a] = butter(1,2*fc/fs);
theta = filtfilt(b,a,S.data.x);
x = deg2rad(theta)*radius;
fc = 20;
[b,a] = butter(1,2*fc/fs);
dx = 2*pi*fc*(x - filter(b,a,x));

%iq = 9924:97950;
iq = 9924:37950;
%
%iq = 9924:20924;

tau = S.data.u(iq);
p1 = psi2kPa(S.data.p_m1(iq));
p2 = psi2kPa(S.data.p_m2(iq));


F = tau/radius;
x = x(iq);
dx = dx(iq);

figure;
  plot(F);

figure; hold all;
  plot(p1);
  plot(p2);

figure;
  plot(x)

save data.mat F x dx p1 p2


