syms x y;
clc; clear all; close all;
a0 = -0.643994119305536;
a1 = 0.193026088639877;
a2 = -0.588012674015623;
p = [0.4;0.5;0.6];
colors = ["#9999FF","#CCFFCC","#80B3FF"];
% figure(1);
figure('Position', [300, 300, 1000, 800]);

axis([-1 20 -10 25]) % 시각화 영역을 제한하여 직선 부분 제거;
set(gca, 'FontSize', 20);
grid on;
hold on;
% fimplicit(@(x,y) a1 + a0/x + a2/y, [0.01 30 0.01 30]);
for i = 1:size(p,1)
    odd_logit = log(p(i)/(1-p(i)));
    
    if odd_logit<a0
        X_plot = linspace(0,50,700);
    else
        X_plot = linspace((odd_logit-a0)/a1+0.01,50,700);
    end
    Y_plot = -a2.*X_plot./(a1.*X_plot+a0-odd_logit);
    if odd_logit<a0
        X_plot = [0, X_plot];
        Y_plot = [600, Y_plot];
    end
%     윗 그래프
%     fimplicit(@(x,y) a1*x*y+(a0+odd_logit)*y + a2*x, [-1 30 -30 30]);
    graph(i) = fill(X_plot, Y_plot, 'k');
    graph(i).FaceColor = colors(i);
    graph(i).EdgeColor = colors(i);
    if odd_logit > a0
        X_plot = linspace(0,(odd_logit-a0)/a1-0.01,700);
        Y_plot = -a2.*X_plot./(a1.*X_plot+a0-odd_logit);
        graph(i) = area(X_plot, Y_plot);
        graph(i).FaceColor = colors(i);
        graph(i).EdgeColor = colors(i);
        x_fill = [(odd_logit-a0)/a1 20 20 (odd_logit-a0)/a1];
        y_fill = [0 0 -25 -25];
        temp = fill(x_fill,y_fill, 'k');
        temp.FaceColor = colors(i);
        temp.EdgeColor = colors(i);
    else
        x_fill = [0 20 20 0];
        y_fill = [0 0 -25 -25];
        temp = fill(x_fill,y_fill, 'k');
        temp.FaceColor = colors(i);
        temp.EdgeColor = colors(i);
    end
end
line([-10 50],[0 0], 'Color', 'k', 'LineStyle', '-');
line([0 0],[60 -10], 'Color', 'k', 'LineStyle', '-');
for i = 1:size(p,1)
    odd_logit = log(p(i)/(1-p(i)));
    line([(odd_logit-a0)/a1 (odd_logit-a0)/a1],[60 -10], 'Color',[0 0 0 0.5], 'LineStyle', '--','LineWidth',0.3);
end
legend(graph,'p = 0.4','p = 0.5','p = 0.6','Fontsize',19);
xlabel('Follow gap [m]','fontsize', 19);
ylabel('TTC [s]','fontsize', 19);
