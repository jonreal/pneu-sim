classdef HybridNeuralODE < nnet.layer.Layer

  properties
    PartialDynamics
    PartialDyanmicsJacobian

    NumParameters
    StateSize
    ControlSize

    HiddenSize
    NumHiddenLayers

  end

  properties (Learnable)
    PartialDynamicsParameters
    ForceNet
  end

  methods

    function layer = HybridNeuralODE( ...
        partialDynamics, numParameters, stateSize, controlSize, opt ...
    )
      arguments
        partialDynamics
        numParameters
        stateSize
        controlSize
        opt.HiddenSize = 64
        opt.NumHiddenLayers = 2
        opt.Name = "HybridNeuralODE"
      end

      layer.Name = opt.Name;
      layer.Description = 'Hybrid Neural ODE Layer';
      layer.PartialDynamics = partialDynamics;
      layer.NumParameters = numParameters;
      layer.StateSize = stateSize;
      layer.ControlSize = controlSize;
      layer.HiddenSize = opt.HiddenSize;
      layer.NumHiddenLayers = opt.NumHiddenLayers;

      layer.ForceNet = [ ...
        featureInputLayer(stateSize, 'Name', 'input') ...
        fullyConnectedLayer(layer.HiddenSize, 'Name', 'hidden1') ...
        reluLayer('Name', 'relu1') ...
      ];
      for i = 2:opt.NumHiddenLayers
        layer.ForceNet = [ ...
          layer.ForceNet ...
          fullyConnectedLayer(layer.HiddenSize, 'Name', sprintf('hidden%d', i)) ...
          reluLayer('Name', sprintf('relu%d', i)) ...
        ];
      layer.ForceNet = [ ...
        layer.ForceNet ...
        fullyConnectedLayer(1, 'Name', 'output') ...
      ];
      end
      layer.ForceNet = dlnetwork(layer.ForceNet);

     end

    function layer = initialize(layer,layout)

      % Physics Parameters
      layer.PartialDynamicsParameters = abs( ...
          randn([layer.NumParameters, 1]) * 0.01) + 0.1;

      % Initialize ForceNet with example input
      exampleInput = dlarray(randn(layer.StateSize, 1), 'CB');
      layer.ForceNet = initialize(layer.ForceNet, exampleInput);

      % Scale down ForceNet parameters for small initialization
      for i = 1:height(layer.ForceNet.Learnables)
        layer.ForceNet.Learnables.Value{i} =  ...
          layer.ForceNet.Learnables.Value{i} * 0.001;
      end

    end

    function force = forceNetPredict(layer, x)
      if ~isa(x, 'dlarray')
        x = dlarray(x, 'CB');
      end
      force = predict(layer.ForceNet, x');
    end


    function Y = predict(layer, X)

      if isa(X, 'dlarray')
          X_vals = extractdata(X);
      else
          X_vals = X;
      end
      X_vals = X_vals;

      total_input_size = numel(X_vals);
      nsteps = (total_input_size - layer.StateSize) / (1 + layer.ControlSize);

      x0 = dlarray(X_vals(1:layer.StateSize));
      t = X_vals((1:nsteps) + layer.StateSize);
      u_flat = X_vals((1:nsteps*layer.ControlSize) + layer.StateSize + nsteps);
      u_ = reshape(u_flat, [layer.ControlSize, nsteps]);
      u = @(tq) interp1(t, u_', tq, 'linear', 'extrap')';

      % Handle dlnetwork validation case (when time vector is invalid)
      if t(1) == t(end) || any(diff(t) <= 0)
        Y = dlarray(zeros(nsteps * layer.StateSize, 1));
        return;
      end

      [t, y] = ode45(@(t,x) double(layer.PartialDynamics( ...
          log(1 + exp(extractdata(layer.PartialDynamicsParameters))), ...
          t, x, u(t)) ...
        + [zeros(layer.StateSize-1, 1); layer.forceNetPredict(x)]), ...
        double(t), double(x0));
      Y = dlarray(y(:));
    end

    function Y = forward(layer,X)
      Y = predict(layer, X);
    end
  end
end


