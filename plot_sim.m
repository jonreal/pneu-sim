figure('Position',[1035 909 380 820]);

  ysp = 0.025;
  w = 0.8;
  h = (0.8 - 4*ysp)/5;
  xs = 0.15;
  ys = 0.9 - h;

  pos = [xs, ys, w, h];
  ax = axes('position',pos);
  xxd = cell2mat(arrayfun(xd,T,'UniformOutput',0)')';
  plot(T,xxd(:,1),'m'); hold all;
  plot(T,X(:,1),'k');
  err = xxd(:,1) - X(:,1);
  [v2,i] = max(abs(err));
  plot(T,err,'g');
  p1 = plot(T(i),err(i),'or');
  p1.MarkerFaceColor = 'r';
  ylabel('Position')
  l = legend('Desired','Actual','Error','location','NW');
  l.Color = 'none';
  l.Box = 'off';
  axis off;
  ax.YAxis.Visible = 'on';
  ylabel('Joint Position')
  ylim([floor(min([xxd(:,1);X(:,1);err])),ceil(max([xxd(:,1);X(:,1);err]))]);
  ann = annotation('textbox',pos);
  ann.String = { ['Peak error = ', num2str(v2,'%3.4f')], ...
                  ['RMS error = ', num2str(rms(err),'%3.4f')] };
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'right';
  ann.FontSize = 14;
  ys = ys - ysp - h;

  pos = [xs, ys, w, h];
  ax = axes('position',pos); hold all;
  Xd = cell2mat(arrayfun(xd,T,'UniformOutput',0)')';
  yhelp =@(t,x,dx,xd,dxd) y(t,[x;dx],[xd;dxd]);
  y_ = cell2mat(arrayfun(yhelp,T,X(:,1),X(:,2),Xd(:,1),Xd(:,2), ...
               'UniformOutput',0));
  plot(T,y_,'k');
  axis off;
  ax.YAxis.Visible = 'on';
  ylabel({'Human Input','(Equilbrium)'})
  ylim([floor(min(y_)),ceil(max(y_))]);
  ys = ys - ysp - h;

  Fh_ = arrayfun(Fh,T,X(:,1),y_);
  fc = 10;
  [b,a] = butter(1,2*fc/fs);
  dy_ = 2*pi*fc*(y_ - filter(b,a,y_));
  power = Fh_.*dy_;
  [v,i] = max(abs(power).*(T>0.25));
  pos = [xs, ys, w, h];
  ax = axes('position',pos); hold all;
  plot(T,power,'k');
  p1 = plot(T(i),power(i),'or');
  p1.MarkerFaceColor = 'r';
  axis off;
  ax.YAxis.Visible = 'on';
  ylabel('Human Power')
  ylim([floor(min(power*10))/10, ceil(max(power*10))/10]);
  ann = annotation('textbox',pos);
  ann.String = { ['Peak power = ', num2str(v,'%3.4f')], ...
                  ['RMS power = ', num2str(rms(power),'%3.4f')] };
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'right';
  ann.FontSize = 14;
  ys = ys - ysp - h;

  pos = [xs, ys, w, h];
  err = arrayfun(model.fe,X(:,3),X(:,8),X(:,4),X(:,9));
  a = arrayfun(model.ft,X(:,5),err);
  %a = arrayfun(model.f_trigger,X(:,5),X(:,3),X(:,4),X(:,8),X(:,9));
  ax = axes('position',pos); hold all;
  %plot(T, trigger/model.threshold,'r');
  plot(T, a,'k');
  plot(T,T.*0 + model.threshold,'--k');
  plot(T,T.*0 - model.threshold,'--k');
  axis off;
  ax.YAxis.Visible = 'on';
  ylabel('Activation')
  ylim([-2,2]);
  ys = ys - ysp - h;

  pos = [xs, ys, w, h];
  ax = axes('position',pos); hold all;
  ep = model.xeq(X(:,3),X(:,4));
  epd = model.xeq(X(:,6),X(:,7));
  plot(T,epd,'r');
  plot(T,ep,'k');
  ylim([floor(min(ep)),ceil(max(ep))]);
  xlim([tt(1),ceil(tt(end))]);
  axis off;
  ax.YAxis.Visible = 'on';
  ax.XAxis.Visible = 'on';
  ylabel('Equilibrium');
  xlabel('time (s)');

