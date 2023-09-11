
clear all
clc

N = 2999; % simulation samples
h = 0.05; % sampling time (C 2 D)

%% Gradient Parameters
EG = 1;
EC = 1;

EG_SGD = 1;

%% parameters
Thetaf1 = [1 h]; %x1k + h*x2k
Thetaf2 = [1-0.2*h -4.905*h]; % (1-0.2*h)*x2k - 4.905*h*sin(x1k)
q = 3;
P = 3;

Num_Sim = 10;

%% For trapz integration
x1_trapz = -2:0.1:2;
x2_trapz = -0.5:0.01:0.5;
[X1, X2] = meshgrid(x1_trapz, x2_trapz);

%% Measurement noise
vb = [0.001, 0.01];
var_vb = [0, 0]; % To be computed during the simulation


for ivb=1:2

v1 = zeros(1, N);
v2 = zeros(1, N);


Thetahatf1 = zeros(q, N); % identified parameters in Thetaf1
Thetahatf2 = zeros(q, N); % identified parameters in Thetaf2



E_SCL = zeros(3, N+1); % 3 for three functions f1, f2, g2
E_SGD = zeros(3, N+1); % 3 for three functions f1, f2, g2

E_SCL_1 = zeros(Num_Sim, N+1); %f1
E_SCL_1_min = zeros(1, N+1);
E_SCL_1_max = zeros(1, N+1);
E_SCL_1_mean = zeros(1, N+1);

E_SCL_2 = zeros(Num_Sim, N+1); %f2
E_SCL_2_min = zeros(1, N+1);
E_SCL_2_max = zeros(1, N+1);
E_SCL_2_mean = zeros(1, N+1);

E_SCL_3 = zeros(Num_Sim, N+1); %g2
E_SCL_3_min = zeros(1, N+1);
E_SCL_3_max = zeros(1, N+1);
E_SCL_3_mean = zeros(1, N+1);

E_SGD_1 = zeros(Num_Sim, N+1);
E_SGD_1_min = zeros(1, N+1);
E_SGD_1_max = zeros(1, N+1);
E_SGD_1_mean = zeros(1, N+1);

E_SGD_2 = zeros(Num_Sim, N+1);
E_SGD_2_min = zeros(1, N+1);
E_SGD_2_max = zeros(1, N+1);
E_SGD_2_mean = zeros(1, N+1);

E_SGD_3 = zeros(Num_Sim, N+1);
E_SGD_3_min = zeros(1, N+1);
E_SGD_3_max = zeros(1, N+1);
E_SGD_3_mean = zeros(1, N+1);



%% Stacked storage for simulation results
X1_SCL = zeros(Num_Sim, N+1);
X2_SCL = zeros(Num_Sim, N+1);

X1_SCL_min = zeros(1, N+1);
X1_SCL_max = zeros(1, N+1);
X1_SCL_mean = zeros(1, N+1);

X2_SCL_min = zeros(1, N+1);
X2_SCL_max = zeros(1, N+1);
X2_SCL_mean = zeros(1, N+1);

Thetahatf1_SCL_1 = zeros(Num_Sim, N+1);
Thetahatf1_SCL_1_min = zeros(1, N+1);
Thetahatf1_SCL_1_max = zeros(1, N+1);
Thetahatf1_SCL_1_mean = zeros(1, N+1);

Thetahatf1_SCL_2 = zeros(Num_Sim, N+1);
Thetahatf1_SCL_2_min = zeros(1, N+1);
Thetahatf1_SCL_2_max = zeros(1, N+1);
Thetahatf1_SCL_2_mean = zeros(1, N+1);

Thetahatf2_SCL_1 = zeros(Num_Sim, N+1);
Thetahatf2_SCL_1_min = zeros(1, N+1);
Thetahatf2_SCL_1_max = zeros(1, N+1);
Thetahatf2_SCL_1_mean = zeros(1, N+1);

Thetahatf2_SCL_2 = zeros(Num_Sim, N+1);
Thetahatf2_SCL_2_min = zeros(1, N+1);
Thetahatf2_SCL_2_max = zeros(1, N+1);
Thetahatf2_SCL_2_mean = zeros(1, N+1);

Thetahatf2_SCL_3 = zeros(Num_Sim, N+1);
Thetahatf2_SCL_3_min = zeros(1, N+1);
Thetahatf2_SCL_3_max = zeros(1, N+1);
Thetahatf2_SCL_3_mean = zeros(1, N+1);

ThetaTilda_SCL_F1_norm = zeros(1, N+1);
ThetaTilda_SCL_F2_norm = zeros(1, N+1);
ThetaTilda_SCL_G2_norm = zeros(1, N+1);
ThetaTilda_SCL_F1 = zeros(Num_Sim, N+1);
ThetaTilda_SCL_F2 = zeros(Num_Sim, N+1);
ThetaTilda_SCL_G2 = zeros(Num_Sim, N+1);

ThetaTilda_SCL_F1_min = zeros(1, N+1);
ThetaTilda_SCL_F1_max = zeros(1, N+1);
ThetaTilda_SCL_F1_mean = zeros(1, N+1);

ThetaTilda_SCL_F2_min = zeros(1, N+1);
ThetaTilda_SCL_F2_max = zeros(1, N+1);
ThetaTilda_SCL_F2_mean = zeros(1, N+1);

ThetaTilda_SCL_G2_min = zeros(1, N+1);
ThetaTilda_SCL_G2_max = zeros(1, N+1);
ThetaTilda_SCL_G2_mean = zeros(1, N+1);

% SGD Statistics initialization
Thetahatf1_SGD_1 = zeros(Num_Sim, N+1);
Thetahatf1_SGD_1_min = zeros(1, N+1);
Thetahatf1_SGD_1_max = zeros(1, N+1);
Thetahatf1_SGD_1_mean = zeros(1, N+1);

Thetahatf1_SGD_2 = zeros(Num_Sim, N+1);
Thetahatf1_SGD_2_min = zeros(1, N+1);
Thetahatf1_SGD_2_max = zeros(1, N+1);
Thetahatf1_SGD_2_mean = zeros(1, N+1);

Thetahatf2_SGD_1 = zeros(Num_Sim, N+1);
Thetahatf2_SGD_1_min = zeros(1, N+1);
Thetahatf2_SGD_1_max = zeros(1, N+1);
Thetahatf2_SGD_1_mean = zeros(1, N+1);

Thetahatf2_SGD_2 = zeros(Num_Sim, N+1);
Thetahatf2_SGD_2_min = zeros(1, N+1);
Thetahatf2_SGD_2_max = zeros(1, N+1);
Thetahatf2_SGD_2_mean = zeros(1, N+1);

Thetahatf2_SGD_3 = zeros(Num_Sim, N+1);
Thetahatf2_SGD_3_min = zeros(1, N+1);
Thetahatf2_SGD_3_max = zeros(1, N+1);
Thetahatf2_SGD_3_mean = zeros(1, N+1);

for ins=1:Num_Sim

    % Reset parameters

    % SCL identified parameters
    Thetahatf1 = zeros(q, N);
    Thetahatf2 = zeros(q, N);
    
    % SGD identified parameters
    Thetahatf1_SGD = zeros(q, N);
    Thetahatf2_SGD = zeros(q, N);

    % initial condition
    x1 = zeros(1, N);
    x2 = zeros(1, N);
    u = zeros(1, N);
    x1(1) = 0.1;
    x2(1) = 0.1;

    yh1 = zeros(1, N);
    yh2 = zeros(1, N);

    yh1_SGD = zeros(1, N);
    yh2_SGD = zeros(1, N);

    e1_SGD = zeros(1, N); % yh1-y1
    e2_SGD = zeros(1, N); % yh2-y2
    
    y1 = zeros(1, N); %x1(k+1) + v1
    y2 = zeros(1, N); %x2(k+1) + v2
    
    e1 = zeros(1, N); % yh1-y1
    e2 = zeros(1, N); % yh2-y2

    phi = zeros(q, N);
    
    M = zeros(q, P); %stacked phi
    Yh1 = zeros(1, P); %stacked y1
    Yh2 = zeros(1, P); %stacked y2 
    mi = 1;

    for k=1:N
    u(k) = -0.8*x1(k)-0.8*x2(k) + 1*exp(-0.2*k)*(sin(0.1*x1(k))+.5*sin(0.3*x2(k)) ...
        +0.1*sin(0.7*x1(k))+0.8*sin(1.6*x2(k))+0.2*sin(1.4*x2(k)));
%     u(k) = u(k) + 1*exp(-0.2*k)*(sin(0.1*x1(k))+.5*sin(0.3*x2(k)) ...
%         +0.1*sin(0.7*x1(k))+0.8*sin(1.6*x2(k))+0.2*sin(1.4*x2(k)));
    x1(k+1) = Thetaf1(1)*x1(k) + Thetaf1(2)*x2(k);
    x2(k+1) = Thetaf2(1)*x2(k) + 0.1*h*u(k) + Thetaf2(2)*sin(x1(k));

    phi(:, k) = [x1(k); x2(k); sin(x1(k))];

    yh1(k) = Thetahatf1(:, k)'*phi(:, k);
    yh2(k) = Thetahatf2(:, k)'*phi(:, k);

    v1(k) = -vb(ivb)+2*vb(ivb)*rand(); % TODO: How much noise?
    v2(k) = -vb(ivb)+2*vb(ivb)*rand(); % TODO: How much noise?



    y1(k) = x1(k+1) + v1(k);
    y2(k) = x2(k+1) + v2(k);

    e1(k) = yh1(k) - y1(k);
    e2(k) = yh2(k) - y2(k);



    yh1_SGD(k) = Thetahatf1_SGD(:, k)'*phi(:, k);
    yh2_SGD(k) = Thetahatf2_SGD(:, k)'*phi(:, k);

    e1_SGD(k) = yh1_SGD(k) - y1(k);
    e2_SGD(k) = yh2_SGD(k) - y2(k);


    if k <= q
        M(:, mi) = phi(:, k);
        Yh1(mi) = y1(k);
        Yh2(mi) = y2(k);
        mi = mi+1;
    end

%     rank(M)

    if k > q
        M_new = M;
        M_best = M;
        Yh1_new = Yh1;
        Yh2_new = Yh2;
        Yh1_best = Yh1;
        Yh2_best = Yh2;
        condition_num_best = 0;
        S = zeros(q,q);
        for j=1:q
            S = S + M_best(:, j)*M_best(:, j)';
        end
%         S = M_best*M_best';
        eigs = eig(S);
        condition_num_best = abs(eigs(1)/eigs(q));
        for i=1:P
            M_new(:, i) = phi(:, k);
            Yh1_new(i) = y1(k);
            Yh2_new(i) = y2(k);
            S = zeros(q,q);
            for j=1:q
                S = S + M_new(:, j)*M_new(:, j)';
            end
%             S = M_new*M_new';
            eigs = eig(S);
            if abs(eigs(1)/eigs(q)) > condition_num_best
                M_best = M_new;
                Yh1_best = Yh1_new;
                Yh2_best = Yh2_new;
                condition_num_best = abs(eigs(1)/eigs(q));
            end

            M_new = M;
            Yh1_new = Yh1;
            Yh2_new = Yh2;
        end

        M = M_best;
        Yh1 = Yh1_best;
        Yh2 = Yh2_best;
        
    end

    PHIE1 = zeros(q, 1);
    PHIE2 = zeros(q, 1);
    for j=1:q
        yhk1 = Thetahatf1(:, k)'*M(:, j);
        ehk1 = yhk1 - Yh1(j);
        PHIE1 = PHIE1 + M(:, j)*ehk1;

        yhk2 = Thetahatf2(:, k)'*M(:, j);
        ehk2 = yhk2 - Yh2(j);
        PHIE2 = PHIE2 + M(:, j)*ehk2;

    end
    Thetahatf1(:, k+1) = Thetahatf1(:, k) - blkdiag(eye(2), 0)*(EG*phi(:, k)*e1(k) + EC*PHIE1);
    Thetahatf2(:, k+1) = Thetahatf2(:, k) - blkdiag(0, eye(2))*(EG*phi(:, k)*e2(k) + EC*PHIE2);

    Thetahatf1_SGD(:, k+1) = Thetahatf1_SGD(:, k) - blkdiag(eye(2), zeros(1))*(EG_SGD*phi(:, k)*e1_SGD(k));
    Thetahatf2_SGD(:, k+1) = Thetahatf2_SGD(:, k) - blkdiag(0, eye(2))*(EG_SGD*phi(:, k)*e2_SGD(k));

    % ThetaTilda for SCL
    ThetaTilda_SCL_F1_norm(1, k) = norm([Thetahatf1(1,k)-Thetaf1(1), Thetahatf1(2, k)-Thetaf1(2)]);
    ThetaTilda_SCL_F2_norm(1, k) = norm([Thetahatf2(2,k)-Thetaf2(1), Thetahatf2(3, k)-Thetaf2(2)]);

    % Fuction approximation error integral for SCL
    F1hat = Thetahatf1(1, k)*X1 + Thetahatf1(2, k)*X2;
    F2hat = Thetahatf2(2, k)*X2 + Thetahatf2(3, k)*sin(X1);
    F1 = Thetaf1(1)*X1 + Thetaf1(2)*X2;
    F2 = Thetaf2(1)*X2 + Thetaf2(2)*sin(X1);
    E_SCL(1, k) = trapz(x2_trapz, trapz(x1_trapz, abs(F1hat-F1), 2));
    E_SCL(2, k) = trapz(x2_trapz, trapz(x1_trapz, abs(F2hat-F2), 2));

    % Fuction approximation error integral for SGD
    F1hat_SGD = Thetahatf1_SGD(1, k)*X1 + Thetahatf1_SGD(2, k)*X2;
    F2hat_SGD = Thetahatf2_SGD(2, k)*X2 + Thetahatf2_SGD(3, k)*sin(X1);
    E_SGD(1, k) = trapz(x2_trapz, trapz(x1_trapz, abs(F1hat_SGD-F1), 2));
    E_SGD(2, k) = trapz(x2_trapz, trapz(x1_trapz, abs(F2hat_SGD-F2), 2));

    end

    var_vb(ivb) = var(v1);

    ins
    Thetahatf1_SCL_1(ins, :) = Thetahatf1(1, :);
    Thetahatf1_SCL_2(ins, :) = Thetahatf1(2, :);

    Thetahatf2_SCL_1(ins, :) = Thetahatf2(2, :);
    Thetahatf2_SCL_2(ins, :) = Thetahatf2(3, :);

    E_SCL_1(i, :) = E_SCL(1, :);
    E_SCL_2(i, :) = E_SCL(2, :);

    Thetahatf1_SGD_1(ins, :) = Thetahatf1_SGD(1, :);
    Thetahatf1_SGD_2(ins, :) = Thetahatf1_SGD(2, :);

    Thetahatf2_SGD_1(ins, :) = Thetahatf2_SGD(2, :);
    Thetahatf2_SGD_2(ins, :) = Thetahatf2_SGD(3, :);

    E_SGD_1(i, :) = E_SGD(1, :);
    E_SGD_2(i, :) = E_SGD(2, :);

    ThetaTilda_SCL_F1(i, :) = ThetaTilda_SCL_F1_norm;
    ThetaTilda_SCL_F2(i, :) = ThetaTilda_SCL_F2_norm;


end

%% Statistical Computation

ThetaTilda_SCL_F1_min = min(ThetaTilda_SCL_F1);
ThetaTilda_SCL_F1_max = max(ThetaTilda_SCL_F1);
ThetaTilda_SCL_F1_mean = mean(ThetaTilda_SCL_F1);

ThetaTilda_SCL_F2_min = min(ThetaTilda_SCL_F2);
ThetaTilda_SCL_F2_max = max(ThetaTilda_SCL_F2);
ThetaTilda_SCL_F2_mean = mean(ThetaTilda_SCL_F2);

E_SCL_1_min = min(E_SCL_1);
E_SCL_1_max = max(E_SCL_1);
E_SCL_1_mean = mean(E_SCL_1);

E_SCL_2_min = min(E_SCL_2);
E_SCL_2_max = max(E_SCL_2);
E_SCL_2_mean = mean(E_SCL_2);


E_SGD_1_min = min(E_SGD_1);
E_SGD_1_max = max(E_SGD_1);
E_SGD_1_mean = mean(E_SGD_1);

E_SGD_2_min = min(E_SGD_2);
E_SGD_2_max = max(E_SGD_2);
E_SGD_2_mean = mean(E_SGD_2);

Thetahatf1_SCL_1_min = min(Thetahatf1_SCL_1);
Thetahatf1_SCL_1_max = max(Thetahatf1_SCL_1);
Thetahatf1_SCL_1_mean = mean(Thetahatf1_SCL_1);

Thetahatf1_SCL_2_min = min(Thetahatf1_SCL_2);
Thetahatf1_SCL_2_max = max(Thetahatf1_SCL_2);
Thetahatf1_SCL_2_mean = mean(Thetahatf1_SCL_2);

Thetahatf2_SCL_1_min = min(Thetahatf2_SCL_1);
Thetahatf2_SCL_1_max = max(Thetahatf2_SCL_1);
Thetahatf2_SCL_1_mean = mean(Thetahatf2_SCL_1);

Thetahatf2_SCL_2_min = min(Thetahatf2_SCL_2);
Thetahatf2_SCL_2_max = max(Thetahatf2_SCL_2);
Thetahatf2_SCL_2_mean = mean(Thetahatf2_SCL_2);


Thetahatf1_SGD_1_min = min(Thetahatf1_SGD_1);
Thetahatf1_SGD_1_max = max(Thetahatf1_SGD_1);
Thetahatf1_SGD_1_mean = mean(Thetahatf1_SGD_1);

Thetahatf1_SGD_2_min = min(Thetahatf1_SGD_2);
Thetahatf1_SGD_2_max = max(Thetahatf1_SGD_2);
Thetahatf1_SGD_2_mean = mean(Thetahatf1_SGD_2);

Thetahatf2_SGD_1_min = min(Thetahatf2_SGD_1);
Thetahatf2_SGD_1_max = max(Thetahatf2_SGD_1);
Thetahatf2_SGD_1_mean = mean(Thetahatf2_SGD_1);

Thetahatf2_SGD_2_min = min(Thetahatf2_SGD_2);
Thetahatf2_SGD_2_max = max(Thetahatf2_SGD_2);
Thetahatf2_SGD_2_mean = mean(Thetahatf2_SGD_2);


figure(1) % Ef1(k)
if ivb == 1
    subplot(2, 1, 1)
    p_scl=fill([1:N+1, N+1:-1:1], [E_SGD_1_max, E_SGD_1_min(end:-1:1)],'r');
    p_scl.FaceColor = [1 0.8 0.8];      
    p_scl.EdgeColor = 'none';
    hold on 
    plt_sgd = plot(1:N+1,E_SGD_1_mean,'-.r');
    p_scl = fill([1:N+1, N+1:-1:1], [E_SCL_1_max, E_SCL_1_min(end:-1:1)],'b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_scl = plot(1:N+1,E_SCL_1_mean,'--b');
    legend([plt_scl, plt_sgd], {'$CL-SGD, \sigma^2=3 \times 10^{-7}$','$SGD,\sigma^2=3 \times 10^{-7} $'},'Interpreter','latex')
    ylabel({'$E_{f_1}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
else
    subplot(2, 1, 2)
    p_scl=fill([1:N+1, N+1:-1:1], [E_SGD_1_max, E_SGD_1_min(end:-1:1)],'r');
    p_scl.FaceColor = [1 0.8 0.8];      
    p_scl.EdgeColor = 'none';
    hold on 
    plt_sgd = plot(1:N+1,E_SGD_1_mean,'-.r');
    p_scl = fill([1:N+1, N+1:-1:1], [E_SCL_1_max, E_SCL_1_min(end:-1:1)],'b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_scl = plot(1:N+1,E_SCL_1_mean,'--b');
    legend([plt_scl, plt_sgd], {'$CL-SGD,\sigma^2=3 \times 10^{-5} $','$SGD,\sigma^2=3 \times 10^{-5} $'},'Interpreter','latex')
    ylabel({'$E_{f_1}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
end


figure(2) %Ef2(k)
if ivb == 1
    subplot(2, 1, 1)
    p_scl=fill([1:N+1, N+1:-1:1], [E_SGD_2_max, E_SGD_2_min(end:-1:1)],'r');
    p_scl.FaceColor = [1 0.8 0.8];      
    p_scl.EdgeColor = 'none';
    hold on 
    plt_sgd = plot(1:N+1,E_SGD_2_mean,'-.r');
    p_scl = fill([1:N+1, N+1:-1:1], [E_SCL_2_max, E_SCL_2_min(end:-1:1)],'b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_scl = plot(1:N+1,E_SCL_2_mean,'--b');
    legend([plt_scl, plt_sgd], {'$CL-SGD,\sigma^2=3 \times 10^{-7} $','$SGD,\sigma^2=3 \times 10^{-7} $'},'Interpreter','latex')
    ylabel({'$E_{f_2}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
else 
    subplot(2, 1, 2)
    p_scl=fill([1:N+1, N+1:-1:1], [E_SGD_2_max, E_SGD_2_min(end:-1:1)],'r');
    p_scl.FaceColor = [1 0.8 0.8];      
    p_scl.EdgeColor = 'none';
    hold on 
    plt_sgd = plot(1:N+1,E_SGD_2_mean,'-.r');
    p_scl = fill([1:N+1, N+1:-1:1], [E_SCL_2_max, E_SCL_2_min(end:-1:1)],'b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_scl = plot(1:N+1,E_SCL_2_mean,'--b');
    legend([plt_scl, plt_sgd], {'$CL-SGD,\sigma^2=3 \times 10^{-5} $','$SGD,\sigma^2=3 \times 10^{-5} $'},'Interpreter','latex')
    ylabel({'$E_{f_2}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
end 


figure(4)
if ivb == 1
    subplot(2, 1, 1)
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf1_SCL_1_max, Thetahatf1_SCL_1_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf1_SCL_1_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf1_SGD_1_max, Thetahatf1_SGD_1_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf1_SGD_1_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf1_SCL_2_max, Thetahatf1_SCL_2_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf1_SCL_2_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf1_SGD_2_max, Thetahatf1_SGD_2_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf1_SGD_2_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    plt_true = plot(1:N+1, Thetaf1(1)*ones(1, N+1),'k','LineWidth',1.);
    hold on
    plt_true = plot(1:N+1, Thetaf1(2)*ones(1, N+1),'k','LineWidth',1.);
    legend([plt_true, plt_scl, plt_sgd], {'$True$', '$CL-SGD, \sigma^2=3 \times 10^{-7}$','$SGD, \sigma^2=3 \times 10^{-7}$'},'Interpreter','latex')
    ylabel({'Parameters of $f_1(x_k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
else 
    subplot(2, 1, 2)
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf1_SCL_1_max, Thetahatf1_SCL_1_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf1_SCL_1_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf1_SGD_1_max, Thetahatf1_SGD_1_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf1_SGD_1_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf1_SCL_2_max, Thetahatf1_SCL_2_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf1_SCL_2_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf1_SGD_2_max, Thetahatf1_SGD_2_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf1_SGD_2_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    plt_true = plot(1:N+1, Thetaf1(1)*ones(1, N+1),'k','LineWidth',1.);
    hold on
    plt_true = plot(1:N+1, Thetaf1(2)*ones(1, N+1),'k','LineWidth',1.);
    legend([plt_true, plt_scl, plt_sgd], {'$True$', '$CL-SGD, \sigma^2=3 \times 10^{-5}$','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
    ylabel({'Parameters of $f_1(x_k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
end 


figure(5)
if ivb == 1
    subplot(2, 1, 1)
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf2_SCL_1_max, Thetahatf2_SCL_1_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf2_SCL_1_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf2_SGD_1_max, Thetahatf2_SGD_1_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf2_SGD_1_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
        p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf2_SCL_2_max, Thetahatf2_SCL_2_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf2_SCL_2_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf2_SGD_2_max, Thetahatf2_SGD_2_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf2_SGD_2_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    plt_true = plot(1:N+1, Thetaf2(1)*ones(1, N+1),'k','LineWidth',1.);
    hold on
    plt_true = plot(1:N+1, Thetaf2(2)*ones(1, N+1),'k','LineWidth',1.);
    legend([plt_true, plt_scl, plt_sgd], {'$True$', '$CL-SGD, \sigma^2=3 \times 10^{-7}$','$SGD, \sigma^2=3 \times 10^{-7}$'},'Interpreter','latex')
    ylabel({'Parameters of $f_2(x_k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
else
    subplot(2, 1, 2)
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf2_SCL_1_max, Thetahatf2_SCL_1_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf2_SCL_1_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf2_SGD_1_max, Thetahatf2_SGD_1_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf2_SGD_1_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    p_scl=fill([1:N+1, N+1:-1:1], [Thetahatf2_SCL_2_max, Thetahatf2_SCL_2_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,Thetahatf2_SCL_2_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    p_sgd=fill([1:N+1, N+1:-1:1], [Thetahatf2_SGD_2_max, Thetahatf2_SGD_2_min(end:-1:1)],'r');
    hold on 
    plt_sgd = plot(1:N+1,Thetahatf2_SGD_2_mean,'-.r');
    p_sgd.FaceColor = [1 0.8 0.8];      
    p_sgd.EdgeColor = 'none';
    plt_true = plot(1:N+1, Thetaf2(1)*ones(1, N+1),'k','LineWidth',1.);
    hold on
    plt_true = plot(1:N+1, Thetaf2(2)*ones(1, N+1),'k','LineWidth',1.);
    legend([plt_true, plt_scl, plt_sgd], {'$True$', '$CL-SGD, \sigma^2=3 \times 10^{-5}$','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
    ylabel({'Parameters of $f_2(x_k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')

end


figure(7)
if ivb == 1
    subplot(2, 1, 1)
    p_scl=fill([1:N+1, N+1:-1:1], [ThetaTilda_SCL_F1_max, ThetaTilda_SCL_F1_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,ThetaTilda_SCL_F1_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_true = plot(1:N+1, 1*ones(1, N+1),'k','LineWidth',1.); % TODO: Change 1 to sqrt(b/eps)
    legend([plt_true, plt_scl], {'$\sqrt{\frac{b}{\epsilon}}$', '$CL-SGD, \sigma^2=3 \times 10^{-7}$'},'Interpreter','latex')
    ylabel({'$\tilde \Theta_{f_1}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')

else 
    subplot(2, 1, 2)
    p_scl=fill([1:N+1, N+1:-1:1], [ThetaTilda_SCL_F1_max, ThetaTilda_SCL_F1_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,ThetaTilda_SCL_F1_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];
    p_scl.EdgeColor = 'none';
    plt_true = plot(1:N+1, 1*ones(1, N+1),'k','LineWidth',1.); % TODO: Change 1 to sqrt(b/eps)
    legend([plt_true, plt_scl], {'$\sqrt{\frac{b}{\epsilon}}$', '$CL-SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
    ylabel({'$\tilde \Theta_{f_1}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')
end

figure(8)
if ivb == 1
    subplot(2, 1, 1)
    p_scl=fill([1:N+1, N+1:-1:1], [ThetaTilda_SCL_F2_max, ThetaTilda_SCL_F2_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,ThetaTilda_SCL_F2_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_true = plot(1:N+1, 1*ones(1, N+1),'k','LineWidth',1.); % TODO: Change 1 to sqrt(b/eps)
    legend([plt_true, plt_scl], {'$\sqrt{\frac{b}{\epsilon}}$', '$CL-SGD, \sigma^2=3 \times 10^{-7}$'},'Interpreter','latex')
    ylabel({'$\tilde \Theta_{f_2}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')

else 
    subplot(2, 1, 2)
    p_scl=fill([1:N+1, N+1:-1:1], [ThetaTilda_SCL_F2_max, ThetaTilda_SCL_F2_min(end:-1:1)],'b');
    hold on 
    plt_scl = plot(1:N+1,ThetaTilda_SCL_F2_mean,'--b');
    p_scl.FaceColor = [0.8 0.8 1];      
    p_scl.EdgeColor = 'none';
    plt_true = plot(1:N+1, 1*ones(1, N+1),'k','LineWidth',1.); % TODO: Change 1 to sqrt(b/eps)
    legend([plt_true, plt_scl], {'$\sqrt{\frac{b}{\epsilon}}$', '$CL-SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
    ylabel({'$\tilde \Theta_{f_2}(k)$'}, 'Interpreter', 'latex')
    xlabel('k (time steps)')

end



end





