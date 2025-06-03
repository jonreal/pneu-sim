classdef HybridNeuralODE_adjoint < nnet.layer.Layer

  properties
    Dynamics
    Jacobian

    NumParameters
    StateSize
    ControlSize

    HiddenSize
    NumHiddenLayers
  end

  properties (Learnable)
    Parameters
    W1, b1  % First layer: StateSize -> HiddenSize  
    W2, b2  % Hidden layers (if NumHiddenLayers > 1)
    Wout, bout  % Output layer: HiddenSize -> 1
  end

  methods
    function layer = HybridNeuralODE_adjoint( ...
        dynamics, jacobian, numParameters, stateSize, controlSize, opt ...
    )
      arguments
        dynamics
        jacobian
        numParameters
        stateSize
        controlSize
        opt.HiddenSize = 64
        opt.NumHiddenLayers = 2
        opt.Name = ""
      end

      layer.Name = opt.Name;
      layer.Description = 'Hybrid Neural ODE Layer';
      layer.Dynamics = dynamics;
      layer.Jacobian = jacobian;
      layer.NumParameters = numParameters;
      layer.StateSize = stateSize;
      layer.ControlSize = controlSize;
      layer.HiddenSize = opt.HiddenSize;
      layer.NumHiddenLayers = opt.NumHiddenLayers;

     end

    function layer = initialize(layer,layout)

      % Physics Parameters
      layer.Parameters = abs( ...
          randn([layer.NumParameters, 1]) * 0.01) + 0.1;

      % Initialize ForceNet weights (very small initialization)
      layer.W1 = dlarray(randn(layer.HiddenSize, layer.StateSize) * 0.001);
      layer.b1 = dlarray(zeros(layer.HiddenSize, 1));

      % Hidden layer 
      layer.W2 = dlarray(randn(layer.HiddenSize, layer.HiddenSize) * 0.001);
      layer.b2 = dlarray(zeros(layer.HiddenSize, 1));

      % Output layer
      layer.Wout = dlarray(randn(1, layer.HiddenSize) * 0.001);
      layer.bout = dlarray(zeros(1, 1));

      % Debug output
      fprintf('Initialized HybridNeuralODE:\n');
      fprintf( ...
        '  Input layout size: [%s]\n', join(string(layout.Size), ', '));
      fprintf( ...
        '  State size: %d, Control size: %d\n', ...
        layer.StateSize, layer.ControlSize);
      fprintf( ...
        '  Physics params: %d (values: %s)\n', layer.NumParameters, ...
        mat2str(layer.Parameters', 3));
      fprintf( ...
        '  ForceNet: %d→%d→1 (state to force)\n', ...
        layer.StateSize, layer.HiddenSize);
    end

    function force = forceNetPredict(layer,x)
        if ~isa(x,'dlarray'), x = dlarray(x); end   % <<< added
        h1    = relu(layer.W1 * x + layer.b1);
        h2    = relu(layer.W2 * h1 + layer.b2);
        force = layer.Wout * h2 + layer.bout;      % dlarray output
    end

    function Y = predict(layer,X)

        wasDL = isa(X,'dlarray');          % remember type
        if wasDL, X = extractdata(X); end  % always work with double

        total   = numel(X);
        nsteps  = (total-layer.StateSize)/(1+layer.ControlSize);

        x0      = X(1:layer.StateSize);    % double
        tvec    = X(layer.StateSize+(1:nsteps));
        uvec    = X(layer.StateSize+nsteps+(1:nsteps*layer.ControlSize));
        U       = reshape(uvec,[layer.ControlSize nsteps]);
        u       = @(t) interp1(tvec,U.',t,'linear','extrap').';

        % guard for malformed time vector
        if tvec(1)==tvec(end) || any(diff(tvec)<=0)
            Ynum = zeros(layer.StateSize,nsteps,'like',x0);
        else
            pf   = log1p(exp(layer.Parameters));  % soft‑plus
            invM = 1/pf(1);
            rhs  = @(t,x) double(layer.Dynamics(pf,t,x,u(t)) + ...
                          [zeros(layer.StateSize-1,1); ...
                           invM * extractdata(layer.forceNetPredict(x))]);
            [~,Ymat] = ode45(rhs,double(tvec),double(x0));     % double output
            Ynum = Ymat(:);
        end

        % ---- return same type as input ----
        if wasDL
            Y = dlarray(Ynum);
        else
            Y = Ynum;
        end
    end

    function [Y, mem] = forward(layer,X)
      Y = predict(layer, X);
      mem = [];
    end

    function [dLdX,dLdParameters, ...
              dLdW1,dLdb1,dLdW2,dLdb2,dLdWout,dLdbout] = ...
              backward(layer,X,~,dLdY,~)

    % ------------------------------------------------------------------
    % unpack training batch (single sample column‑vector layout [CB])
    % ------------------------------------------------------------------
    if isa(X,'dlarray'), X = extractdata(X); end
    total     = numel(X);
    nsteps    = (total-layer.StateSize)/(1+layer.ControlSize);

    x0        = X(1:layer.StateSize);
    tvec      = X(layer.StateSize + (1:nsteps));
    uvec      = X(layer.StateSize+nsteps + (1:nsteps*layer.ControlSize));
    U         = reshape(uvec,[layer.ControlSize nsteps]);
    u_interp  = @(t) interp1(tvec,U.',t,'linear','extrap').';

    %pf        = double(layer.Parameters);               % 2×1 numeric

    % --- very top of backward (after extractdata) -----------------
    theta = double(layer.Parameters);          % raw learnables (θ)
    pf    = log1p(exp(theta));                 % same warp as predict
    sigm  = 1./(1+exp(-theta));                % ∂pf/∂θ = σ(θ)
    invM  = 1/pf(1);                            % 1/p₁

    W1 = double(layer.W1);  b1 = double(layer.b1);
    W2 = double(layer.W2);  b2 = double(layer.b2);
    Wo = double(layer.Wout); bo = double(layer.bout);

    % ------------------------------------------------------------------
    % forward trajectory (pure double) ---------------------------------
    % ------------------------------------------------------------------
    relu = @(x) max(0,x);
    rhs = @(t,x) double(layer.Dynamics(pf,t,x,u_interp(t)) + ...
                 [0; invM * (Wo * relu(W2*relu(W1*x+b1)+b2) + bo)]);  % append g
    [~,xtraj] = ode45(rhs,double(tvec),double(x0));
    xtraj     = interp1(tvec,xtraj,tvec,'linear')';            % 2×nsteps

    % ------------------------------------------------------------------
    % reshape dL/dY into [StateSize × nsteps], get λ(T)
    % ------------------------------------------------------------------
    dY     = reshape(double(dLdY),layer.StateSize,[]);
    lambdaT= dY(:,end);                                        % 2×1

    % total number of NN params
    nW1 = numel(W1); nb1 = numel(b1);
    nW2 = numel(W2); nb2 = numel(b2);
    nWo = numel(Wo); nbo = numel(bo);
    nPg = nW1+nb1+nW2+nb2+nWo+nbo;

    % ------------------------------------------------------------------
    % build initial adjoint state   z = [λ ; dLdp_f ; dLdp_g ]
    % ------------------------------------------------------------------
    z0 = [lambdaT ; zeros(layer.NumParameters,1) ; zeros(nPg,1)];

    % ------------------------------------------------------------------
    % adjoint ODE (all‑numeric) ----------------------------------------
    % ------------------------------------------------------------------
    % ---------- slice‑and‑reshape helper ----------------------------
    function [blk,newIdx] = take(vec,tmpl,idx)
        n       = numel(tmpl);
        blk     = reshape(vec(idx+1:idx+n),size(tmpl));
        newIdx  = idx + n;
    end
    function [g, dgdx_row, dgdtheta] = forwardAndJacobian(W1,b1,W2,b2,Wo,bo,x)
        % -------- forward pass -----------------------------------------
        z1 = W1*x + b1;           m1 = double(z1>0);   a1 = m1 .* z1;
        z2 = W2*a1 + b2;          m2 = double(z2>0);   a2 = m2 .* z2;
        g  = invM * (Wo * a2 + bo);                    % scaled output

        % -------- ∂g/∂x  (row 1×State) -------------------------------
        row1      = invM * (Wo .* m2');                % 1×Hidden
        row2      = row1 * W2;                         % 1×Hidden
        dgdx_row  = (row2 .* m1') * W1;                % 1×State

        % -------- parameter gradients --------------------------------
        dWo = invM * a2';                              dbo = invM;

        tmp  = invM * (m2 .* Wo');                     % Hidden×1
        dW2  = tmp * a1';                              db2 = tmp;

        tmp2 = W2' * tmp;                              % Hidden×1
        dW1  = (tmp2 .* m1) * x';                      db1 = tmp2 .* m1;

        dgdtheta = [dW1(:); db1(:); dW2(:); db2(:); dWo(:); dbo];
    end
    function dz = adjoint_ode(t,z)

        x  = interp1(tvec,xtraj',t,'linear')';     % 2×1
        ut = u_interp(t);
        lam  = z(1:layer.StateSize);

        % ------- analytic ∂f/∂x and ∂f/∂pf (2×2 each) -------
        dfdx = [ 0 , 1 ;
                 0 , -pf(2)/pf(1) ];

        % ------- NN forward + analytic Jacobians -------------
        [g,dgdx_row,dgdtheta] = forwardAndJacobian(W1,b1,W2,b2,Wo,bo,x);

        dfdpf = [ 0 , 0 ;
                  (pf(2)*x(2)-ut)/pf(1)^2 - g/pf(1) ,   -x(2)/pf(1) ];


        % λ̇  = -λᵗ(∂f/∂x + ∂g/∂x)ᵗ
        %dlam = -(dfdx + dgdx_row)' * lam;

        dfdx(2,:) = dfdx(2,:) + dgdx_row;     % add ∂g/∂x to the 2nd row
        dlam      = -dfdx' * lam;             % λ̇ = -λᵗ(∂f/∂x)ᵗ

        % accumulate physics‑param gradients
        dpf_dot = dfdpf' * lam;

        % only second state is affected by g ⇒ scale by λ₂
        dpg_dot = dgdtheta * lam(2);

        dz = -[dlam ; dpf_dot ; dpg_dot];
    end

    % integrate adjoint backward
    [~,Z] = ode45( ...
      @(t,z) double(adjoint_ode(t,z)),double(flip(tvec)),double(z0));
    Z     = flip(Z,1);

    % ------------------------------------------------------------------
    % extract gradients at t=0 -----------------------------------------
    % ------------------------------------------------------------------
    %dLdParameters = dlarray(Z(1, layer.StateSize+(1:layer.NumParameters))');

    dLdpf        = Z(1, layer.StateSize+(1:layer.NumParameters))';
    dLdParameters = cast( dLdpf .* sigm , 'like', layer.Parameters);
    %dLdParameters = cast( ...
    %  Z(1,layer.StateSize+(1:layer.NumParameters))','like', layer.Parameters);

    pg_vec = Z(1, layer.StateSize+layer.NumParameters+1 : end)';

    % unpack NN‑param gradient vector
    offset = 0;
    [dW1, offset]   = take(pg_vec,W1,offset);
    [db1, offset]   = take(pg_vec,b1,offset);
    [dW2, offset]   = take(pg_vec,W2,offset);
    [db2, offset]   = take(pg_vec,b2,offset);
    [dWo, offset]   = take(pg_vec,Wo,offset);
    [dbo, ~]        = take(pg_vec,bo,offset);

%    dLdW1   = dlarray(dW1);   dLdb1  = dlarray(db1);
%    dLdW2   = dlarray(dW2);   dLdb2  = dlarray(db2);
%    dLdWout = dlarray(dWo);   dLdbout= dlarray(dbo);

    dLdW1   = cast(dW1 , 'like', W1);   dLdb1   = cast(db1 , 'like', b1);
    dLdW2   = cast(dW2 , 'like', W2);   dLdb2   = cast(db2 , 'like', b2);
    dLdWout = cast(dWo , 'like', Wo);   dLdbout = cast(dbo , 'like', bo);

% ---- input‑gradient: keep dlarray if X was dlarray -------------------
    dLdXnum = zeros(size(X),'like',X);
    dLdXnum(1:layer.StateSize) = Z(1,1:layer.StateSize);

    wasDL = isa(X,'dlarray');          % put this before you overwrite X
    if wasDL
        dLdX = dlarray(dLdXnum);
    else
        dLdX = dLdXnum;
    end

    end  % backward

  end
end


