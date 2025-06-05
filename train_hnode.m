clear classes;
% Time
T = 10;
dt = 0.1;
t = 0:dt:(T-dt);
nsteps = numel(t);

% Training Parameters
nSamples = 8;
nTrainingRuns = 5;
Epochs = 100;
LearningRate = 1e-3;

use_adjoint = false;

% Toy model
dx =@(t,x,u,p) [ ...
    x(2);
    1/p(1) * (u - p(2)*x(2) - (p(3)*x(2) + p(4)*x(1) + p(5)*x(1)^2))];

% Use step inputs
u =@(t, ts, a) a * (t >= ts);

%cmap = cbrewer('div','Spectral',nSamples);
%cmap(cmap > 1) = 1;
cmap = colormap(copper(nSamples));

p = [0.5, 0.5, -0.03, 0.1, 0.5]; % Parameters

X_train = cell(size(nSamples, 1), 1);
Y_train = cell(size(nSamples, 1), 1);

% Generate training set
tic
rn = zeros(nSamples, 1);
for i = 1:nSamples
    x0 = [0; 0];
    rn = rand();
    u_ = @(t) u(t, 1, rn);
    u_vec = arrayfun(u_,t);
    [t, x] = ode45(@(t, x) dx(t, x, u_(t), p), t, x0);

    X_train{i} = [x0; t(:); u_vec(:)];
    Y_train{i} = x(:);
    rn(i) = rn;
end
fprintf('Time to generate training set: \n');
toc

[~, idx] = sort(rn);
X_train = X_train(idx);
Y_train = Y_train(idx);

% Set up Neural Hybrid
% Everthing but the stiffness:
Dynamics =@(p,t,x,u) [ ...
  x(2);
  1/p(1) * (u - p(2)*x(2))];


Jacobian = @(p,t,x,u) deal( ...
    [0 1; 0 -p(2)/p(1)], ...                     % dfdx
    [ 0               0 ; ...                    % d/dp₁ d/dp₂
      p(2)/p(1)^2   -1/p(1) ] );                 % dfdpf

numParameters = 2;
stateSize = 2;
controlSize = 2;

if use_adjoint
  hnode_layer = HybridNeuralODE_adjoint(Dynamics, Jacobian, 2, 2, 1);
  fprintf('Using adjoint method for training.\n');
else
  hnode_layer = HybridNeuralODE(Dynamics, 2, 2, 1);
end

input_layer = featureInputLayer(stateSize + 2*numel(t), 'Name', 'input');
example_input = dlarray(X_train{1}, 'CB');
net = dlnetwork([input_layer; hnode_layer], example_input);

options = trainingOptions('adam', ...
    'InitialLearnRate', LearningRate, ...
    'MaxEpochs', Epochs, ...
    'MiniBatchSize', 1, ...
    'Verbose', true);
  %  'Plots','training-progress');

X_train_matrix = cat(2, X_train{:})';  % Each row is a sample
Y_train_matrix = cat(2, Y_train{:})';  % Each row is a target

tic

figure; hold all;
for i=1:nSamples
  x_target = reshape(Y_train{i},size(x));
  plot(t,x_target(:,1),'-','Color',cmap(i,:));
end

ps = gobjects(nTrainingRuns, numel(nSamples));
for k=1:nTrainingRuns

  % Decay Learning rate
  options.InitialLearnRate = options.InitialLearnRate*0.9;

  net = trainnet( ...
    X_train_matrix, Y_train_matrix, net, 'mse', options);

  if ~exist('./net', 'dir'), mkdir('./net'); end
  save(['./net/net-',num2str(k),'.mat'],'net');

  gcf; hold all;
  title(['Training run:', num2str(k)]);
  for i = 1:nSamples
      Y = predict(net, X_train{i}');
      x_model = reshape(Y, size(x));
      ps(k,i) = plot(t,x_model(:,1),'--','Color',cmap(i,:));
      if k > 1
        color_ = ps(k,i).Color;
        ps(k-1,i).Color = [color_, 0.1];
      end
  end

  if ~exist('./training_run', 'dir'), mkdir('./training_run'); end
  print('-dpng',['./training_run/',num2str(k),'.png']);
end
toc
