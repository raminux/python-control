clear all
clc

A=[0.3 0.4 0 0;0 0.3 0.3 0;0 0 0.4 0.4;0.4 0 0 0.4];
% A=[-1 0 0 -1;-2 -1 -3 0; 1 -3 2 5; 0 -1 -1 0];
% B=[1 1 0;0 1 1;0 0 1; 1 1 2];
B=eye(4);
x(:,1)=[1 2 3 4]';
N=8;
m=size(B,2);
n=size(A,1);
for j=1:N
    for l=1:m
    u(j,l)=0.1*rand;
    end
end
u=u';

% u=[0.0184    0.0231    0.0032    0.0425    0.0625
%     0.0087    0.0909    0.0594    0.0522    0.0255
%     0.0309    0.0937    0.0044    0.0840    0.0905];

for i=1:N
    x(:,i+1)=A*x(:,i)+B*u(:,i);
end

X0=[];
for i=1:N
    X0=[X0 x(:,i)];
end

X1=[];
for i=2:N+1
    X1=[X1 x(:,i)];
end

XU=[u;X0];

X0
BA=(X1)/XU
B
B
A