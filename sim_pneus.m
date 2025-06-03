T = 15;
dt = 1/1000;
fs = 1/dt;
tt = (0:(dt):(T - dt))';

%
% -- Desired Human Trajectories
%

% --- Smooth reach and back
Td = 5;
A = 0.125;
xd_ =@(t)  (t < 5).*0 + ...
      (t>=5 & t<10).*(A*sin(2*pi*(1/Td)*t + 3*pi/2) + A) + ...
      (t>=10).*0;
dxd_ =@(t) (t < 5).*0 + ...
      (t>=5 & t<10).*(A*(2*pi*(1/Td))*cos(2*pi*(1/Td)*t + 3*pi/2)) + ...
      (t>=10).*0;
xd =@(t) [xd_(t); dxd_(t)];

% --- sine tracking
%xd =@(t) [sin(2*pi*0.5*t); -cos(2*pi*0.5*t)];
%
%xd = chirp(tt,0.01,100,1);
%[b,a] = butter(1,2*fc/fs);
%dxd = 2*pi*fc*(xd - filter(b,a,xd));
%
%xd =@(t) [interp1(tt,xd,t); interp1(tt,dxd,t)];



% -- step
%xd = tt.*0;
%xd(tt > 5) = 1;
%fc = 0.2;
%[b,a] = butter(1,2*fc/fs);
%xd = filter(b,a,xd);
%dxd = 2*pi*fc*(xd - filter(b,a,xd));
%xd =@(t) [interp1(tt,xd,t); interp1(tt,dxd,t)];

%xd =@(t) [sin(2*pi*0.05*t); -cos(2*pi*0.05*t)];

% -- Human controller parameters
kp = 10;
kd = 10;
kh = 0.5;

Fh =@(t,x,y) kh*(y-x(1));
y =@(t,x,xd) [kp, kd]*(xd - x);
Fe =@(t,x) Fh(t,[x(1);x(2)],y(t,[x(1);x(2)],xd(t)));

%-----------------------------------------------------------------------------
model = make_model(@(t,x) Fe(t,[x(1);x(2)]), ...
                    'fc',0.5,...
                    'threshold',1e-9, ...
                    'deltaP',0.01,...
                    'trefract',0.05);
%model = make_model_old(xd);
x0 = [0; 0; 0.5; 0.5; 0; 0.5; 0.5; 0.5; 0.5; 0; 0];
[T,X] = odeHybrid(model,tt,x0,'debug','on');

plot_sim;
ann = annotation('textbox',[0 0 1 0.97]);
ann.String = 'Active Robot With EP-Reflexes';
ann.EdgeColor = 'none';
ann.HorizontalAlignment = 'center';

%-----------------------------------------------------------------------------
model = make_model(@(t,x) Fe(t,[x(1);x(2)]), ...
                  'threshold',inf);
x0 = [0; 0; 0.5; 0.5; 0; 0.5; 0.5; 0.5; 0.5; 0; 0];
[T,X] = odeHybrid(model,tt,x0);

plot_sim;
ann = annotation('textbox',[0 0 1 0.97]);
ann.String = 'Static Robot (inflated)';
ann.EdgeColor = 'none';
ann.HorizontalAlignment = 'center';

%-----------------------------------------------------------------------------
model = make_model(@(t,x) Fe(t,[x(1);x(2)]), ...
                  'threshold',inf);
x0 = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
[T,X] = odeHybrid(model,tt,x0);

plot_sim;
ann = annotation('textbox',[0 0 1 0.97]);
ann.String = 'Static Robot (deflated)';
ann.EdgeColor = 'none';
ann.HorizontalAlignment = 'center';

