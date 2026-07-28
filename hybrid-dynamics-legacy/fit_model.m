load data/data.mat
fs = 1000;
t = [0:numel(F)-1]'.*(1/fs);

doCheck = 0;
doCheckCost = 0;
doOpt = 1;
doPatternSearch = 1;
doFmin = 1;

nloops = 10;
maxIter = 10;
minStep = 1e-30;

radius = 0.006875;
Itotal = 0.012;
mest = Itotal/radius;
nu =  0.01;
b = 0.096068;



p0 =  [0.0154
        10740
     0.015451
    0.0016449
    mest
    nu
    289.82
    b];


iopts_sets = {[6,7],[1,2,3,4],[1,2,3,4,5,6,7,8]};
lb = [1000 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 0.1]';
ub = [100000 1 1 1 10 1 1 100]';

if doCheck
  tic
  F =@(tq) interp1(t,F,tq);
  model = make_model(@(t,x) F(t), ...
                  'fc',0,...
                  'threshold',Inf, ...
                  'Pt',psi2kPa(120), ...
                  'L', 0.2, ...
                  'pk1',p0(1), ...
                  'pk2',p0(2), ...
                  'pl1',p0(3), ...
                  'pl2',p0(4), ...
                  'm',p0(5), ...
                  'nu',p0(6), ...
                  'C',p0(7), ...
                  'b',p0(8));

  x0 = [x(1); dx(1); p1(1); p2(1); 0; p1(1); p2(1); p1(1); p2(1); 0; 0];
  [T,X] = odeHybrid(model,t,x0);
  toc

figure;
  subplot(211);
    plot(x,'r'); hold all;
    plot(X(:,1),'k');

  subplot(212);
    plot(p1,'r'); hold all;
    plot(p2,'--r'); hold all;
    plot(X(:,3),'k');
    plot(X(:,4),'--k');

  return
end

if doCheckCost
  tic
  c = cost(F,x,dx,p1,p2,t,1,p0)
  toc
  return
end

% Optimization
if doOpt
  psoptions = optimoptions(@patternsearch, ...
                  'MaxIterations',maxIter, ...
                  'MeshTolerance', 1e-30, ...
                  'PlotFcn',{'psplotfuncount', 'psplotbestf'}, ...
                  'PollMethod','GPSPositiveBasis2N', ...
                  'Display','iter', ...
                  'MaxFunctionEvaluations', 1e3, ...
                  'UseCompletePoll',true);
  options = optimoptions('fmincon', ...
                  'algorithm','sqp', ...
                  'StepTolerance',minStep, ...
                  'MaxIterations',maxIter, ...
                  'Display','iter', ...
                  'MaxFunctionEvaluations', 1e3, ...
                  'PlotFcn','optimplotfval');

  for i=1:numel(iopts_sets)

    ihold = 1:numel(p0);
    ihold(iopts_sets{i}) = [];
    for k=1:numel(ihold)
      lb(ihold(k)) = p0(ihold(k));
      ub(ihold(k)) = p0(ihold(k));
    end

    if doPatternSearch
      p0 = patternsearch(@(p) cost(F,x,dx,p1,p2,t,0,p),p0, ...
                                  [],[],[],[],lb,ub,[],psoptions);
    end
    if doFmin
      problem = createOptimProblem( ...
          'fmincon', ...
          'objective',@(p) cost(F,x,dx,p1,p2,t,0,p), ...
          'x0',p0, ...
          'lb',lb,'ub',ub, ...
          'options',options);
      p0 = fmincon(problem)
    end
  end
  p0
end


function c = cost(F,x,dx,p1,p2,t,debug,p)
  p
  tic
  F =@(tq) interp1(t,F,tq);
  model = make_model(@(t,x) F(t), ...
                  'fc',0,...
                  'threshold',Inf, ...
                  'Pt',psi2kPa(120), ...
                  'L', 0.2, ...
                  'pk1',p(1), ...
                  'pk2',p(2), ...
                  'pl1',p(3), ...
                  'pl2',p(4), ...
                  'm',p(5), ...
                  'nu',p(6), ...
                  'C',p(7), ...
                  'b',p(8));

  x0 = [x(1); dx(1); p1(1); p2(1); 0; p1(1); p2(1); p1(1); p2(1); 0; 0];
  [T,X] = odeHybrid(model,t,x0);
  toc

  iq = 9638:numel(x);

  if debug
    figure;
      subplot(211);
        plot(x,'r'); hold all;
        plot(X(:,1),'k');
        ylim([-10e-4,10e-4])

      subplot(212);
        plot(p1,'r'); hold all;
        plot(p2,'--r'); hold all;
        plot(X(:,3),'k');
        plot(X(:,4),'--k');
    end

  c1 = norm(x(iq)-X(iq,1))./norm(X(iq,1))
  c2 = norm(p1(iq)-X(iq,3))./norm(X(iq,3))
  c3 = norm(p2(iq)-X(iq,4))./norm(X(iq,4))

  rsqr = 1 - sum([x(iq) - X(iq,1); ...
                  p1(iq) - X(iq,3); ...
                  p2(iq) - X(iq,4)].^2) ...
            /sum([x(iq) - mean(x(iq)); ...
                  p1(iq) - mean(p1(iq)); ...
                  p2(iq) - mean(p2(iq))].^2)

  %c = -rsqr
  c = c1 + c2 + c3
end
