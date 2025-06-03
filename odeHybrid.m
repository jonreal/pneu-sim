function [T,X] = odeHybrid(model,tspan,x0,options)
  arguments
    model
    tspan
    x0
    options.debug (1,1) string = 'off'
  end

  t0 = 0;
  T = zeros(numel(tspan),1);
  X = zeros(numel(tspan),numel(x0));
  m = 1;


%  figure;
  fprintf('Integrating HDS .')
  while(1)
    if isempty(t0) || (t0 > tspan(end))
      break;
    end

    displayProgress(t0,tspan,options);
    displayDebug(model,m,options);

    [~,i0] = min(abs(tspan-t0));
    [edge,x,tout,t0,x0] = SimulateToNextEvent(model.modes(m),tspan(i0:end),x0);
    m = find(strcmp(edge,{model.modes.domain}));
    X(i0:(i0+numel(tout)-1),:) = x;
    T(i0:(i0+numel(tout)-1),:) = tout;

%    plot(T(1:i0+numel(tout)-1),X(1:i0+numel(tout)-1,:));
  %  pause(0.01);

  end
  fprintf('done.\n\n');
end


% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
function [edge,x,tspan,t0,x0] = SimulateToNextEvent(m,tt,x0);
  if size(tt) < 2
    x = x0;
    tspan = tt(end);
    edge = [];
    t0 = [];
    return
  end

  dt = tt(2)-tt(1);
  options = odeset('Events',m.guards, ...
                   'MaxStep',dt, ...
                   'RelTol',1e-9,'AbsTol',1e-12);
  [x0,xp0] = decic(m.flow,tt(1),x0,ones(size(x0)),zeros(size(x0)),[]);

  %[x0,xp0] = decic(m.flow,tt(1),x0,[0;0;1;0;0;1;1;1;1;0;0], ...
  %                  zeros(size(x0)),[]);

  % Check if initial coniditions are in the guard sets, if so, simulate foward
  % one time step and trigger the transition manually.
  [value,~,direction] = m.guards(tt(1),x0,xp0);
  ie = find((round(value,10).*direction >= 0) == 1);
  if ~isempty(ie)
    options.Events = [];
    [tspan,x] = ode15i(m.flow,[tt(1);tt(2)],x0,xp0,options);
    [tspan,iunique,~] = unique(tspan);
    x = x(iunique,:);
    if numel(ie) > 1
      v_ie = ie(end);
    else
      v_ie = ie(1);
    end
    x0 = m.jumpmaps{v_ie}(tspan(end),x(end,:));
    t0 = tspan(2);
    tspan = tspan(1);
    x = x(1,:);
    edge = m.edges{v_ie};
    return
  end

  % Simulate till next event
  [tspan,x,te,xe,ie] = ode15i(m.flow,tt,x0,xp0,options);

  if isempty(ie)
    edge = [];
    t0 = [];
    return;
  elseif tspan(end) >= tt(end)
    x = interp1(tspan,x,tt,'linear');
    tspan = tt(1:end);
    edge = [];
    t0 = [];
    return
  % Double event, use third edge
  elseif numel(ie) > 1
    ie = 3;
    te = te(end);
  end

  [~,i0] = min(abs(tspan-te));
  t0 = tt(i0+1);
  x0 = m.jumpmaps{ie}(te,x(end,:));
  edge = m.edges{ie};


  %% find tspan at ie
  %[~,i0] = min(abs(tspan-te));
  %tspan(i0:end) = [];
  %x(i0:end,:) = [];

  %% jump map
  %x0 = m.jumpmaps{ie}(t_(end),x_(end,:));
  %% find next time step closest to the event
  %[~,i0] = min(abs((tt>te).*tt - te));

  %% simulate foward onto the next grid timestep
  %[x0,xp0] = decic(m.flow,tspan(end), ...
  %                  x(end,:)',ones(size(x0)),zeros(size(x0)),[]);
  %i0 = min(i0,numel(tt)-1);
  %options.Events = [];

  %%[t_,x_] = ode15i(m.flow,[tspan(end);tt(i0+1)],x0,xp0,options);
  %%[t_,iunique,~] = unique(t_);
  %%x_ = x_(iunique,:);

  %%% jump map
  %%x0 = m.jumpmaps{ie}(t_(end),x_(end,:));

  %%% interp flow to match timesteps
  %%x = interp1([tspan; t_(2:end)],[x; x_(2:end,:)],tt(1:i0+1),'linear');
  %%x = x(1:end-1,:);
  %%tspan = tt(1:i0);
  %%t0 = tt(i0+1);

  %% next mode
  %edge = m.edges{ie};

end

% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
function displayProgress(t0,tspan,options)
  if (mod(floor(t0/tspan(end)*100),10) == 0) && strcmp(options.debug,'off')
    fprintf(repmat(['.'],1));
  end
end

% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
function displayDebug(model,m,options)
  if strcmp(options.debug,'on');
    fprintf('State = %s\n\n',model.modes(m).domain);
  end
end


