clc;
clear;
close all;

%% System Matrix
A = [ 0.2 0.7 0 0;
    0 0.4 0.7 0;
    0 0 0.5 0.5;
    0.7 0 0 0.4];
B = eye(4);

Q = eye(4);
R = eye(4);

[n, m] = size(B);

Gamma = blkdiag(Q, R);


%% Optimization 
H = sdpvar(n+m, n+m);
W = sdpvar(n, n);
W_a = [W zeros(n,m); zeros(m, n+m)];

c_1 =  (H-W_a) >= 0;

ABI = [[A, B], zeros(n, m); zeros(n, n+m), eye(m)];
GammaH = [Gamma-H, zeros(m+n, m); zeros(m, 2*m+n)];
c_2 = (ABI'*H*ABI + GammaH) >= 0;


constraints = c_1 + c_2;

cost = -trace(W);


ops = sdpsettings('solver','mosek','verbose',0);
diagnostics = optimize(constraints, cost, ops)

W_opt = value(W)
H_opt = value(H)
K = -inv(H_opt(n+1:n+m, n+1:n+m))*H_opt(1:n, n+1:n+m)'


