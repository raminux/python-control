% Discrete-time Finite Time Concurent learning with Beta=0

clear all

X1SCL_vs=[];
X2SCL_vs=[];
X3SCL_vs=[];
Norm_tiltSCL_vs=[];
FxhatSCL_vs=[];
Ef_L_Xrange_SCL_vs=[];

X1SGD_svs=[];
X2SGD_svs=[];
X3SGD_svs=[];
Norm_tiltSGD_svs=[];
FxhatSGD_svs=[];
Ef_L_Xrange_SGD_svs=[];

X1SCL_vb=[];
X2SCL_vb=[];
X3SCL_vb=[];
Norm_tiltSCL_vb=[];
FxhatSCL_vb=[];
Ef_L_Xrange_SCL_vb=[];

X1SGD_bvb=[];
X2SGD_bvb=[];
X3SGD_bvb=[];
Norm_tiltSGD_bvb=[];
FxhatSGD_bvb=[];
Ef_L_Xrange_SGD_bvb=[];

% My Finite Time Concurrent Learning

for i=1:10
%close all;
clc;

% intial conditions
NN=[];
x0=[0,0,0,0*rand(1,2)];

save x0

x1_vs(1)=x0(1);

x2_vs(1)=x0(2);
x3_vs(1)=x0(3);

% time span
Ns=10000;
xL=0;
xH=2;


%memory stack length
M1=2;    
Q1_vs=[];
Y_vs=[];

% no. of unknown parameters
P_vs=zeros(1,2);

noisAmp=0.01;

gamaSCL=0.1;  % original
gamaSCL1=0.01; % original

gamaSGD=0.1;  % original

mv=1;

%kk=0;

for k=1:Ns
k
noise= -noisAmp + (noisAmp+noisAmp)*rand;
NN=[NN,noise];
% y(k) = -0.5*exp(-x1(k))+0.5*(exp(-x1(k))*cos(x1(k))) + noise;
y_vs(k) = -0.5*exp(-x1_vs(k))+0.5*(exp(-x1_vs(k))*sin(x1_vs(k))) + noise;

% x1hat(k)=x2(k)*exp(-x1(k))+x3(k)*(exp(-x1(k))*cos(x1(k)));
x1hat_vs(k)=x2_vs(k)*exp(-x1_vs(k))+x3_vs(k)*(exp(-x1_vs(k))*sin(x1_vs(k)));

% the current error of learning
e1_vs(k)=y_vs(k)-x1hat_vs(k);

Norm_tiltheta_vs(k)=norm([x2_vs(k)+0.5, x3_vs(k)-0.5]);
b_phi(k)=norm([exp(-x1_vs(k)),(exp(-x1_vs(k))*sin(x1_vs(k)))]); 

% to plot functions learning errors for the the whole range of x=[-2,2] on time k
NEs=1000;

xL=0;
xH=2;

xx_vs(1)=xL;

for j=1:NEs+1

xx_vs(j+1)=xx_vs(j)+(xH-xL)/NEs;

% fxhat(j)=x2(k)*exp(-xx(j))+x3(k)*exp(-xx(j))*cos(xx(j));
fxhat_vs(j)=x2_vs(k)*exp(-xx_vs(j))+x3_vs(k)*exp(-xx_vs(j))*sin(xx_vs(j));

% fx(j)=-0.5*exp(-xx(j))+0.5*exp(-xx(j))*cos(xx(j));
fx_vs(j)=-0.5*exp(-xx_vs(j))+0.5*exp(-xx_vs(j))*sin(xx_vs(j));

end


ef_L_Xrange_SCL_vs(k)= trapz(xx_vs(1:NEs+1),sqrt((fx_vs-fxhat_vs).^2));



 % system dynamics
 x1_vs(k+1)= x1_vs(k)+(xH-xL)/(Ns);

%%%%%%%%%%%%% Stochastic Concurrent learning %%%%%%%%%%%%%%%%%%%%%%%%%%
x2_vs(k+1)=x2_vs(k)+gamaSCL*e1_vs(k)*exp(-x1_vs(k))+(gamaSCL1*P_vs(1));
% x3(k+1)=x3(k)+gamaSCL*e1(k)*(exp(-x1(k))*cos(x1(k)))+(gamaSCL1*P(2));
x3_vs(k+1)=x3_vs(k)+gamaSCL*e1_vs(k)*(exp(-x1_vs(k))*sin(x1_vs(k)))+(gamaSCL1*P_vs(2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%% Stochastic Concurrent learning (decreasing learning rate)  %%%%%%%%%%%%%%%%%%%%%%%%%%
% x2(k+1)=x2(k)+gamaSCL/k*e1(k)*exp(-x1(k))+(gamaSCL1/k*P(1));
% % x3(k+1)=x3(k)+gamaSCL*e1(k)*(exp(-x1(k))*cos(x1(k)))+(gamaSCL1*P(2));
% x3(k+1)=x3(k)+gamaSCL/k*e1(k)*(exp(-x1(k))*sin(x1(k)))+(gamaSCL1/k*P(2));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if k==Ns
    p1_vs=x2_vs(k+1);
    p2_vs=x3_vs(k+1);
end

%%%%%%%%%%%%% Stochastic GD %%%%%%%%%%%%%%%%%%%%%%%%%%
% x2(k+1)=x2(k)+gamaSGD*e1(k)*exp(-x1(k)));
% x3(k+1)=x3(k)+gamaSGD*e1(k)*(exp(-x1(k))*cos(x1(k)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Collecting memory stack of regressors
if k<=M1 
% Q1=[Q1; exp(-x1(k))    (exp(-x1(k))*cos(x1(k)))];
Q1_vs=[Q1_vs; exp(-x1_vs(k))    (exp(-x1_vs(k))*sin(x1_vs(k)))];
Y_vs=[Y_vs; y_vs(k)];
end

%% Algorithm for maximizing lambda_min(S)/lambda_max(S)
if k==M1
Q2_vs=Q1_vs(1:M1,1:M1);
end

if k>M1
    Lamb=eig(Q2_vs'*Q2_vs);
    ratio=Lamb(1)/Lamb(M1);
%     data_new=[exp(-x1(k))    (exp(-x1(k))*cos(x1(k)))];
    data_new=[exp(-x1_vs(k))    (exp(-x1_vs(k))*sin(x1_vs(k)))];
    ratio_new=zeros(1,M1);
    for h=1:M1
        Q2_new_vs=Q2_vs;
        Q2_new_vs(h,:)=data_new;
        Lamb_new=eig(Q2_new_vs'*Q2_new_vs);
        ratio_new(h)= Lamb_new(1)/Lamb_new(M1);
        Lamb_max_S_vs(k)=Lamb_new(M1);
        Lamb_min_S_vs(k)=Lamb_new(1);
    end
    if max(ratio_new)> ratio
    h_max=find(ratio_new==max(ratio_new));
    Q2_vs(h_max(1),:)=data_new;
    Y_vs(h_max(1),:)=y_vs(k);
    ratio_vec_vs(k)=max(ratio_new);
    %
%     Q1=[Q2,ones(M1)];
    Q1_vs=[Q2_vs];
    %
    else
%     Q1=[Q2,ones(M1)];
    Q1_vs=[Q2_vs];
    ratio_vec_vs(k)=ratio;
    end
end

% Concurrent learning error using current parameters and memory stack of regressors 
ee1_vs=Y_vs(:,1)-Q1_vs(:,1:2)*[x2_vs(k);x3_vs(k)];

% Calculate the concurrent learning term    
P_vs(1)=sum(ee1_vs.*Q1_vs(:,1));
P_vs(2)=sum(ee1_vs.*Q1_vs(:,2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1SCL_vs=[X1SCL_vs;x1_vs];
X2SCL_vs=[X2SCL_vs;x2_vs];
X3SCL_vs=[X3SCL_vs;x3_vs];
Norm_tiltSCL_vs=[Norm_tiltSCL_vs;Norm_tiltheta_vs];
FxhatSCL_vs=[FxhatSCL_vs;fxhat_vs];
Ef_L_Xrange_SCL_vs=[Ef_L_Xrange_SCL_vs;ef_L_Xrange_SCL_vs];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


F_Xrange_IAE_SCL_vs=trapz(1:Ns,sqrt((ef_L_Xrange_SCL_vs).^2))

b_phi_max_vs=max(b_phi)
b_phi_min_vs=min(b_phi)

true1=[-0.5*ones(1,Ns)];
true2=[0.5*ones(1,Ns)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Stochastic Gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% intial conditions

x0=[0,0,0,0*rand(1,2)];

save x0

x1_svs(1)=x0(1);

x2_svs(1)=x0(2);
x3_svs(1)=x0(3);


%memory stack length
M1=2;    
Q1_svs=[];
Y_svs=[];

% no. of unknown parameters
P_svs=zeros(1,2);


mv=1;

%kk=0;

for k=1:Ns
%k
noise= -noisAmp + (noisAmp+noisAmp)*rand;

% y(k) = -0.5*exp(-x1(k))+0.5*(exp(-x1(k))*cos(x1(k))) + noise;
y_svs(k) = -0.5*exp(-x1_svs(k))+0.5*(exp(-x1_svs(k))*sin(x1_svs(k))) + noise;

% x1hat(k)=x2(k)*exp(-x1(k))+x3(k)*(exp(-x1(k))*cos(x1(k)));
x1hat_svs(k)=x2_svs(k)*exp(-x1_svs(k))+x3_svs(k)*(exp(-x1_svs(k))*sin(x1_svs(k)));

% the current error of learning
e1_svs(k)=y_svs(k)-x1hat_svs(k);

Norm_tiltheta_svs(k)=norm([x2_svs(k)+0.5, x3_svs(k)-0.5]);

% to plot functions learning errors for the the whole range of x=[-2,2] on time k
NEs=1000;

xL=0;
xH=2;

xx_svs(1)=xL;

for j=1:NEs+1

xx_svs(j+1)=xx_svs(j)+(xH-xL)/NEs;

% fxhat(j)=x2(k)*exp(-xx(j))+x3(k)*exp(-xx(j))*cos(xx(j));
fxhat_svs(j)=x2_svs(k)*exp(-xx_svs(j))+x3_svs(k)*exp(-xx_svs(j))*sin(xx_svs(j));

% fx(j)=-0.5*exp(-xx(j))+0.5*exp(-xx(j))*cos(xx(j));
fx_svs(j)=-0.5*exp(-xx_svs(j))+0.5*exp(-xx_svs(j))*sin(xx_svs(j));

end


ef_L_Xrange_SGD_svs(k)= trapz(xx_svs(1:NEs+1),sqrt((fx_svs-fxhat_svs).^2));



 % system dynamics
 x1_svs(k+1)= x1_svs(k)+(xH-xL)/(Ns);

% %%%%%%%%%%%%% Stochastic Concurrent learning %%%%%%%%%%%%%%%%%%%%%%%%%%
% x2(k+1)=x2(k)+gamaSCL*e1(k)*exp(-x1(k))+(gamaSCL1*P(1));
% x3(k+1)=x3(k)+gamaSCL*e1(k)*(exp(-x1(k))*cos(x1(k)))+(gamaSCL1*P(2));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if k==Ns
%     p1=x2(k+1);
%     p2=x3(k+1);
% end

%%%%%%%%%%%%% Stochastic GD  %%%%%%%%%%%%%%%%%%%%%%%%%%
x2_svs(k+1)=x2_svs(k)+gamaSGD*e1_svs(k)*exp(-x1_svs(k));
% x3(k+1)=x3(k)+gamaSGD*e1(k)*(exp(-x1(k))*cos(x1(k)));
x3_svs(k+1)=x3_svs(k)+gamaSGD*e1_svs(k)*(exp(-x1_svs(k))*sin(x1_svs(k)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%% Stochastic GD (decreasing learning rate) %%%%%%%%%%%%%%%%%%%%%%%%%%
% x2(k+1)=x2(k)+gamaSGD/k*e1(k)*exp(-x1(k));
% % x3(k+1)=x3(k)+gamaSGD*e1(k)*(exp(-x1(k))*cos(x1(k)));
% x3(k+1)=x3(k)+gamaSGD/k*e1(k)*(exp(-x1(k))*sin(x1(k)));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if k==Ns
    p1_svs=x2_svs(k+1);
    p2_svs=x3_svs(k+1);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1SGD_svs=[X1SGD_svs;x1_svs];
X2SGD_svs=[X2SGD_svs;x2_svs];
X3SGD_svs=[X3SGD_svs;x3_svs];
Norm_tiltSGD_svs=[Norm_tiltSGD_svs;Norm_tiltheta_svs];
FxhatSGD_svs=[FxhatSGD_svs;fxhat_svs];
Ef_L_Xrange_SGD_svs=[Ef_L_Xrange_SGD_svs;ef_L_Xrange_SGD_svs];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


F_Xrange_IAE_SGD_svs=trapz(1:Ns,sqrt((ef_L_Xrange_SGD_svs).^2))

true1=[-0.5*ones(1,Ns)];
true2=[0.5*ones(1,Ns)];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Higher variance (noisAmp=0.1;)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% intial conditions

x0=[0,0,0,0*rand(1,2)];
NNb=[];
save x0

x1_vb(1)=x0(1);

x2_vb(1)=x0(2);
x3_vb(1)=x0(3);

% time span
Ns=10000;
xL=0;
xH=2;


%memory stack length
M1=2;    
Q1_vb=[];
Y_vb=[];

% no. of unknown parameters
P_vb=zeros(1,2);

noisAmp=0.1;

gamaSCL=0.1;  % original
gamaSCL1=0.01; % original

gamaSGD=0.1;  % original

mv=1;

%kk=0;

for k=1:Ns,

noise= -noisAmp + (noisAmp+noisAmp)*rand;
NNb=[NNb,noise];
% y(k) = -0.5*exp(-x1(k))+0.5*(exp(-x1(k))*cos(x1(k))) + noise;
y_vb(k) = -0.5*exp(-x1_vb(k))+0.5*(exp(-x1_vb(k))*sin(x1_vb(k))) + noise;

% x1hat(k)=x2(k)*exp(-x1(k))+x3(k)*(exp(-x1(k))*cos(x1(k)));
x1hat_vb(k)=x2_vb(k)*exp(-x1_vb(k))+x3_vb(k)*(exp(-x1_vb(k))*sin(x1_vb(k)));

% the current error of learning
e1_vb(k)=y_vb(k)-x1hat_vb(k);

Norm_tiltheta_vb(k)=norm([x2_vb(k)+0.5, x3_vb(k)-0.5]);



% to plot functions learning errors for the the whole range of x=[-2,2] on time k
NEs=1000;

xL=0;
xH=2;

xx_vb(1)=xL;

for j=1:NEs+1

xx_vb(j+1)=xx_vb(j)+(xH-xL)/NEs;

% fxhat(j)=x2(k)*exp(-xx(j))+x3(k)*exp(-xx(j))*cos(xx(j));
fxhat_vb(j)=x2_vb(k)*exp(-xx_vb(j))+x3_vb(k)*exp(-xx_vb(j))*sin(xx_vb(j));

% fx(j)=-0.5*exp(-xx(j))+0.5*exp(-xx(j))*cos(xx(j));
fx_vb(j)=-0.5*exp(-xx_vb(j))+0.5*exp(-xx_vb(j))*sin(xx_vb(j));

end


ef_L_Xrange_SCL_vb(k)= trapz(xx_vb(1:NEs+1),sqrt((fx_vb-fxhat_vb).^2));



 % system dynamics
 x1_vb(k+1)= x1_vb(k)+(xH-xL)/(Ns);

%%%%%%%%%%%%% Stochastic Concurrent learning %%%%%%%%%%%%%%%%%%%%%%%%%%
x2_vb(k+1)=x2_vb(k)+gamaSCL*e1_vb(k)*exp(-x1_vb(k))+(gamaSCL1*P_vb(1));
% x3(k+1)=x3(k)+gamaSCL*e1(k)*(exp(-x1(k))*cos(x1(k)))+(gamaSCL1*P(2));
x3_vb(k+1)=x3_vb(k)+gamaSCL*e1_vb(k)*(exp(-x1_vb(k))*sin(x1_vb(k)))+(gamaSCL1*P_vb(2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if k==Ns
    p1_vb=x2_vb(k+1);
    p2_vb=x3_vb(k+1);
end

%%%%%%%%%%%%% Stochastic GD %%%%%%%%%%%%%%%%%%%%%%%%%%
% x2(k+1)=x2(k)+gamaSGD*e1(k)*exp(-x1(k)));
% x3(k+1)=x3(k)+gamaSGD*e1(k)*(exp(-x1(k))*cos(x1(k)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Collecting memory stack of regressors
if k<=M1 
% Q1=[Q1; exp(-x1(k))    (exp(-x1(k))*cos(x1(k)))];
Q1_vb=[Q1_vb; exp(-x1_vb(k))    (exp(-x1_vb(k))*sin(x1_vb(k)))];
Y_vb=[Y_vb; y_vb(k)];
end

%% Algorithm for maximizing lambda_min(S)/lambda_max(S)
if k==M1
Q2_vb=Q1_vb(1:M1,1:M1);
end

if k>M1
    Lamb_vb=eig(Q2_vb'*Q2_vb);
    ratio_vb=Lamb_vb(1)/Lamb_vb(M1);
%     data_new=[exp(-x1(k))    (exp(-x1(k))*cos(x1(k)))];
    data_new_vb=[exp(-x1_vb(k))    (exp(-x1_vb(k))*sin(x1_vb(k)))];
    ratio_new_vb=zeros(1,M1);
    for h=1:M1
        Q2_new_vb=Q2_vb;
        Q2_new_vb(h,:)=data_new_vb;
        Lamb_new_vb=eig(Q2_new_vb'*Q2_new_vb);
        ratio_new_vb(h)= Lamb_new_vb(1)/Lamb_new_vb(M1);
        Lamb_max_S_vb(k)=Lamb_new_vb(M1);
        Lamb_min_S_vb(k)=Lamb_new_vb(1);
    end
    if max(ratio_new_vb)> ratio_vb
    h_max_vb=find(ratio_new_vb==max(ratio_new_vb));
    Q2_vb(h_max_vb(1),:)=data_new_vb;
    Y_vb(h_max_vb(1),:)=y_vb(k);
    ratio_vec_vb(k)=max(ratio_new_vb);
    %
%     Q1=[Q2,ones(M1)];
    Q1_vb=[Q2_vb];
    %
    else
%     Q1=[Q2,ones(M1)];
    Q1_vb=[Q2_vb];
    ratio_vec_vb(k)=ratio_vb;
    end
end

% Concurrent learning error using current parameters and memory stack of regressors 
ee1_vb=Y_vb(:,1)-Q1_vb(:,1:2)*[x2_vb(k);x3_vb(k)];

% Calculate the concurrent learning term    
P_vb(1)=sum(ee1_vb.*Q1_vb(:,1));
P_vb(2)=sum(ee1_vb.*Q1_vb(:,2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1SCL_vb=[X1SCL_vb;x1_vb];
X2SCL_vb=[X2SCL_vb;x2_vb];
X3SCL_vb=[X3SCL_vb;x3_vb];
Norm_tiltSCL_vb=[Norm_tiltSCL_vb;Norm_tiltheta_vb];
FxhatSCL_vb=[FxhatSCL_vb;fxhat_vb];
Ef_L_Xrange_SCL_vb=[Ef_L_Xrange_SCL_vb;ef_L_Xrange_SCL_vb];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


F_Xrange_IAE_SCL_vb=trapz(1:Ns,sqrt((ef_L_Xrange_SCL_vb).^2))
sigb_vb=var(NNb)


true1=[-0.5*ones(1,Ns)];
true2=[0.5*ones(1,Ns)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Stochastic Gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% intial conditions

x0=[0,0,0,0*rand(1,2)];

save x0

x1_bvb(1)=x0(1);

x2_bvb(1)=x0(2);
x3_bvb(1)=x0(3);


%memory stack length
M1=2;    
Q1_bvb=[];
Y_bvb=[];

% no. of unknown parameters
P_bvb=zeros(1,2);


mv=1;

%kk=0;

for k=1:Ns,
%k
noise= -noisAmp + (noisAmp+noisAmp)*rand;

% y(k) = -0.5*exp(-x1(k))+0.5*(exp(-x1(k))*cos(x1(k))) + noise;
y_bvb(k) = -0.5*exp(-x1_bvb(k))+0.5*(exp(-x1_bvb(k))*sin(x1_bvb(k))) + noise;

% x1hat(k)=x2(k)*exp(-x1(k))+x3(k)*(exp(-x1(k))*cos(x1(k)));
x1hat_bvb(k)=x2_bvb(k)*exp(-x1_bvb(k))+x3_bvb(k)*(exp(-x1_bvb(k))*sin(x1_bvb(k)));

% the current error of learning
e1_bvb(k)=y_bvb(k)-x1hat_bvb(k);

Norm_tiltheta_bvb(k)=norm([x2_bvb(k)+0.5, x3_bvb(k)-0.5]);

% to plot functions learning errors for the the whole range of x=[-2,2] on time k
NEs=1000;

xL=0;
xH=2;

xx_bvb(1)=xL;

for j=1:NEs+1

xx_bvb(j+1)=xx_bvb(j)+(xH-xL)/NEs;

% fxhat(j)=x2(k)*exp(-xx(j))+x3(k)*exp(-xx(j))*cos(xx(j));
fxhat_bvb(j)=x2_bvb(k)*exp(-xx_bvb(j))+x3_bvb(k)*exp(-xx_bvb(j))*sin(xx_bvb(j));

% fx(j)=-0.5*exp(-xx(j))+0.5*exp(-xx(j))*cos(xx(j));
fx_bvb(j)=-0.5*exp(-xx_bvb(j))+0.5*exp(-xx_bvb(j))*sin(xx_bvb(j));

end


ef_L_Xrange_SGD_bvb(k)= trapz(xx_bvb(1:NEs+1),sqrt((fx_bvb-fxhat_bvb).^2));



 % system dynamics
 x1_bvb(k+1)= x1_bvb(k)+(xH-xL)/(Ns);

% %%%%%%%%%%%%% Stochastic Concurrent learning %%%%%%%%%%%%%%%%%%%%%%%%%%
% x2(k+1)=x2(k)+gamaSCL*e1(k)*exp(-x1(k))+(gamaSCL1*P(1));
% x3(k+1)=x3(k)+gamaSCL*e1(k)*(exp(-x1(k))*cos(x1(k)))+(gamaSCL1*P(2));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if k==Ns
%     p1=x2(k+1);
%     p2=x3(k+1);
% end

%%%%%%%%%%%%% Stochastic GD %%%%%%%%%%%%%%%%%%%%%%%%%%
x2_bvb(k+1)=x2_bvb(k)+gamaSGD*e1_bvb(k)*exp(-x1_bvb(k));
% x3(k+1)=x3(k)+gamaSGD*e1(k)*(exp(-x1(k))*cos(x1(k)));
x3_bvb(k+1)=x3_bvb(k)+gamaSGD*e1_bvb(k)*(exp(-x1_bvb(k))*sin(x1_bvb(k)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if k==Ns
    p1_bvb=x2_bvb(k+1);
    p2_bvb=x3_bvb(k+1);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1SGD_bvb=[X1SGD_bvb;x1_bvb];
X2SGD_bvb=[X2SGD_bvb;x2_bvb];
X3SGD_bvb=[X3SGD_bvb;x3_bvb];
Norm_tiltSGD_bvb=[Norm_tiltSGD_bvb;Norm_tiltheta_bvb];
FxhatSGD_bvb=[FxhatSGD_bvb;fxhat_bvb];
Ef_L_Xrange_SGD_bvb=[Ef_L_Xrange_SGD_bvb;ef_L_Xrange_SGD_bvb];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


F_Xrange_IAE_SGD_bvb=trapz(1:Ns,sqrt((ef_L_Xrange_SGD_bvb).^2))

true1=[-0.5*ones(1,Ns)];
true2=[0.5*ones(1,Ns)];

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(x1_vs)
X1SCL_min_vs(i)=min(X1SCL_vs(:,i));
X1SCL_max_vs(i)=max(X1SCL_vs(:,i));
X1SCL_mean_vs(i)=mean(X1SCL_vs(:,i));
end

for i=1:length(x2_vs)
X2SCL_min_vs(i)=min(X2SCL_vs(:,i));
X2SCL_max_vs(i)=max(X2SCL_vs(:,i));
X2SCL_mean_vs(i)=mean(X2SCL_vs(:,i));
end

for i=1:length(x3_vs)
X3SCL_min_vs(i)=min(X3SCL_vs(:,i));
X3SCL_max_vs(i)=max(X3SCL_vs(:,i));
X3SCL_mean_vs(i)=mean(X3SCL_vs(:,i));
end

for i=1:length(Norm_tiltheta_vs)
Norm_tiltSCL_min_vs(i)=min(Norm_tiltSCL_vs(:,i));
Norm_tiltSCL_max_vs(i)=max(Norm_tiltSCL_vs(:,i));
Norm_tiltSCL_mean_vs(i)=mean(Norm_tiltSCL_vs(:,i));
end

for i=1:length(fxhat_vs)
FxhatSCL_min_vs(i)=min(FxhatSCL_vs(:,i));
FxhatSCL_max_vs(i)=max(FxhatSCL_vs(:,i));
FxhatSCL_mean_vs(i)=mean(FxhatSCL_vs(:,i));
end

for i=1:length(ef_L_Xrange_SCL_vs)
Ef_L_Xrange_SCL_min_vs(i)=min(Ef_L_Xrange_SCL_vs(:,i));
Ef_L_Xrange_SCL_max_vs(i)=max(Ef_L_Xrange_SCL_vs(:,i));
Ef_L_Xrange_SCL_mean_vs(i)=mean(Ef_L_Xrange_SCL_vs(:,i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(x1_svs)
X1SGD_svs_min(i)=min(X1SGD_svs(:,i));
X1SGD_svs_max(i)=max(X1SGD_svs(:,i));
X1SGD_svs_mean(i)=mean(X1SGD_svs(:,i));
end

for i=1:length(x2_svs)
X2SGD_min_svs(i)=min(X2SGD_svs(:,i));
X2SGD_max_svs(i)=max(X2SGD_svs(:,i));
X2SGD_mean_svs(i)=mean(X2SGD_svs(:,i));
end

for i=1:length(x3_svs)
X3SGD_min_svs(i)=min(X3SGD_svs(:,i));
X3SGD_max_svs(i)=max(X3SGD_svs(:,i));
X3SGD_mean_svs(i)=mean(X3SGD_svs(:,i));
end

for i=1:length(Norm_tiltheta_svs)
Norm_tiltSGD_min_svs(i)=min(Norm_tiltSGD_svs(:,i));
Norm_tiltSGD_max_svs(i)=max(Norm_tiltSGD_svs(:,i));
Norm_tiltSGD_mean_svs(i)=mean(Norm_tiltSGD_svs(:,i));
end

for i=1:length(fxhat_svs)
FxhatSGD_min_svs(i)=min(FxhatSGD_svs(:,i));
FxhatSGD_max_svs(i)=max(FxhatSGD_svs(:,i));
FxhatSGD_mean_svs(i)=mean(FxhatSGD_svs(:,i));
end

for i=1:length(ef_L_Xrange_SGD_svs)
Ef_L_Xrange_SGD_min_svs(i)=min(Ef_L_Xrange_SGD_svs(:,i));
Ef_L_Xrange_SGD_max_svs(i)=max(Ef_L_Xrange_SGD_svs(:,i));
Ef_L_Xrange_SGD_mean_svs(i)=mean(Ef_L_Xrange_SGD_svs(:,i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(x1_vb)
X1SCL_vb_min(i)=min(X1SCL_vb(:,i));
X1SCL_vb_max(i)=max(X1SCL_vb(:,i));
X1SCL_vb_mean(i)=mean(X1SCL_vb(:,i));
end

for i=1:length(x2_vb)
X2SCL_min_vb(i)=min(X2SCL_vb(:,i));
X2SCL_max_vb(i)=max(X2SCL_vb(:,i));
X2SCL_mean_vb(i)=mean(X2SCL_vb(:,i));
end

for i=1:length(x3_vb)
X3SCL_min_vb(i)=min(X3SCL_vb(:,i));
X3SCL_max_vb(i)=max(X3SCL_vb(:,i));
X3SCL_mean_vb(i)=mean(X3SCL_vb(:,i));
end

for i=1:length(Norm_tiltheta_vb)
Norm_tiltSCL_min_vb(i)=min(Norm_tiltSCL_vb(:,i));
Norm_tiltSCL_max_vb(i)=max(Norm_tiltSCL_vb(:,i));
Norm_tiltSCL_mean_vb(i)=mean(Norm_tiltSCL_vb(:,i));
end

for i=1:length(fxhat_vb)
FxhatSCL_min_vb(i)=min(FxhatSCL_vb(:,i));
FxhatSCL_max_vb(i)=max(FxhatSCL_vb(:,i));
FxhatSCL_mean_vb(i)=mean(FxhatSCL_vb(:,i));
end

for i=1:length(ef_L_Xrange_SCL_vb)
Ef_L_Xrange_SCL_min_vb(i)=min(Ef_L_Xrange_SCL_vb(:,i));
Ef_L_Xrange_SCL_max_vb(i)=max(Ef_L_Xrange_SCL_vb(:,i));
Ef_L_Xrange_SCL_mean_vb(i)=mean(Ef_L_Xrange_SCL_vb(:,i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(x1_bvb)
X1SGD_bvb_min(i)=min(X1SGD_bvb(:,i));
X1SGD_bvb_max(i)=max(X1SGD_bvb(:,i));
X1SGD_bvb_mean(i)=mean(X1SGD_bvb(:,i));
end

for i=1:length(x2_bvb)
X2SGD_min_bvb(i)=min(X2SGD_bvb(:,i));
X2SGD_max_bvb(i)=max(X2SGD_bvb(:,i));
X2SGD_mean_bvb(i)=mean(X2SGD_bvb(:,i));
end

for i=1:length(x3_bvb)
X3SGD_min_bvb(i)=min(X3SGD_bvb(:,i));
X3SGD_max_bvb(i)=max(X3SGD_bvb(:,i));
X3SGD_mean_bvb(i)=mean(X3SGD_bvb(:,i));
end

for i=1:length(Norm_tiltheta_bvb)
Norm_tiltSGD_min_bvb(i)=min(Norm_tiltSGD_bvb(:,i));
Norm_tiltSGD_max_bvb(i)=max(Norm_tiltSGD_bvb(:,i));
Norm_tiltSGD_mean_bvb(i)=mean(Norm_tiltSGD_bvb(:,i));
end

for i=1:length(fxhat_bvb)
FxhatSGD_min_bvb(i)=min(FxhatSGD_bvb(:,i));
FxhatSGD_max_bvb(i)=max(FxhatSGD_bvb(:,i));
FxhatSGD_mean_bvb(i)=mean(FxhatSGD_bvb(:,i));
end

for i=1:length(ef_L_Xrange_SGD_bvb)
Ef_L_Xrange_SGD_min_bvb(i)=min(Ef_L_Xrange_SGD_bvb(:,i));
Ef_L_Xrange_SGD_max_bvb(i)=max(Ef_L_Xrange_SGD_bvb(:,i));
Ef_L_Xrange_SGD_mean_bvb(i)=mean(Ef_L_Xrange_SGD_bvb(:,i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wts = [repmat(1/mv,mv,1)];
x2_vs = conv(x2_vs,wts,'valid');
x3_vs = conv(x3_vs,wts,'valid');

x2_svs = conv(x2_svs,wts,'valid');
x3_svs = conv(x3_svs,wts,'valid');

x2_vb = conv(x2_vb,wts,'valid');
x3_vb = conv(x3_vb,wts,'valid');

x2_bvb = conv(x2_bvb,wts,'valid');
x3_bvb = conv(x3_bvb,wts,'valid');


figure(2)
subplot(2,1,1)
pp202=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X2SCL_max_vs, X2SCL_min_vs(end:-1:1)],'b');
pp202.FaceColor = [ 0.8 0.8 1];      
pp202.EdgeColor = 'none';
hold on
pt202=plot([1:Ns-mv+2],[X2SCL_mean_vs],'--b');
hold on
pp2020=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X3SCL_max_vs, X3SCL_min_vs(end:-1:1)],'b');
pp2020.FaceColor = [ 0.8 0.8 1];      
pp2020.EdgeColor = 'none';
hold on
pp202020=plot([1:Ns-mv+2],[X3SCL_mean_vs],'--b');
hold on
phnp3=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X2SGD_max_svs, X2SGD_min_svs(end:-1:1)],'r');
phnp3.FaceColor = [1 0.8 0.8];      
phnp3.EdgeColor = 'none';
hold on
pp340=plot([1:Ns-mv+2],[X2SGD_mean_svs],'-.r');
hold on
pp33=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X3SGD_max_svs, X3SGD_min_svs(end:-1:1)],'r');
pp33.FaceColor = [1 0.8 0.8];      
pp33.EdgeColor = 'none';
hold on
pp304=plot([1:Ns-mv+2],[X3SGD_mean_svs],'-.r');
hold on
pp1=plot(1:Ns,true1,'-k');
hold on
plot(1:Ns,true2,'-k')
hold on
xlabel('k (time steps)')
ylabel('Parameters')
legend([pp1,pp202020,pp304],{'$True$','$CL-SGD,\sigma^2=3 \times 10^{-5} $','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
xlim([0 Ns+1])
%ylim([-0.6 0.6])
%ylim([-1.2 1.7])

subplot(2,1,2)
pp4=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X2SCL_max_vb, X2SCL_min_vb(end:-1:1)],'b');
pp4.FaceColor = [ 0.8 0.8 1];      
pp4.EdgeColor = 'none';
hold on
pg4=plot([1:Ns-mv+2],[X2SCL_mean_vb],'--b');
hold on
pp404=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X3SCL_max_vb, X3SCL_min_vb(end:-1:1)],'b');
pp404.FaceColor = [ 0.8 0.8 1];      
pp404.EdgeColor = 'none';
hold on
pp405=plot([1:Ns-mv+2],[X3SCL_mean_vb],'--b');
hold on
pc5=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X2SGD_max_bvb, X2SGD_min_bvb(end:-1:1)],'r');
pc5.FaceColor = [1 0.8 0.8];      
pc5.EdgeColor = 'none';
hold on
pf5=plot([1:Ns-mv+2],[X2SGD_mean_bvb],'-.r');
hold on
pp55=fill([1:Ns-mv+2 Ns-mv+2:-1:1],[X3SGD_max_bvb, X3SGD_min_bvb(end:-1:1)],'r');
pp55.FaceColor = [1 0.8 0.8];      
pp55.EdgeColor = 'none';
hold on
pp585=plot([1:Ns-mv+2],[X3SGD_mean_bvb],'-.r');
hold on
pjp1=plot(1:Ns,true1,'-k');
hold on
plot(1:Ns,true2,'-k')
hold on
xlabel('k (time steps)')
ylabel('Parameters')
legend([pjp1,pp405,pp585],{'$True$','$CL-SGD,\sigma^2=3 \times 10^{-3} $','$SGD, \sigma^2=3 \times 10^{-3}$'},'Interpreter','latex')
xlim([0 Ns+1])
%ylim([-0.6 0.6])


figure(3)
subplot(2,1,1)
f1=fill([1:Ns,Ns:-1:1],[Norm_tiltSCL_max_vs, Norm_tiltSCL_min_vs(end:-1:1)],'b');
f1.FaceColor = [ 0.8 0.8 1];      
f1.EdgeColor = 'none';
hold on
ft1=plot([1:Ns],[Norm_tiltSCL_mean_vs],'--b');
hold on
f2=plot(1:Ns,0.06*ones(1,Ns),'k','LineWidth',1.);
xlim([0 Ns+1])
legend([f2,ft1],{'$\sqrt {\frac{b}{\epsilon}}$','$CL-SGD,\sigma^2=3 \times 10^{-5} $'},'Interpreter','latex');
xlabel('k (time steps)')
ylabel({'$\tilde \Theta(k)$'},'Interpreter','latex')
%legend([pp1,pp2,pp3],{'$True$','$SCL,\sigma^2=3 \times 10^{-5} $','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')

subplot(2,1,2)
ff2=fill([1:Ns,Ns:-1:1],[Norm_tiltSCL_max_vb, Norm_tiltSCL_min_vb(end:-1:1)],'b');
ff2.FaceColor = [ 0.8 0.8 1];      
ff2.EdgeColor = 'none';
hold on
ftf2=plot([1:Ns],[Norm_tiltSCL_mean_vb],'--b');
hold on
f4=plot(1:Ns,0.6*ones(1,Ns),'k','LineWidth',1.);
xlim([0 Ns+1])
legend([f4,ftf2],{'$\sqrt {\frac{b}{\epsilon}}$','$CL-SGD,\sigma^2=3 \times 10^{-3} $'},'Interpreter','latex')
xlabel('k (time steps)')
ylabel({'$\tilde \Theta(k)$'},'Interpreter','latex')
%legend([pp1,pp2,pp3],{'$True$','$SCL,\sigma^2=3 \times 10^{-5} $','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')


figure(503)
subplot(2,1,1)
pep2=fill([1:Ns-mv+1, Ns-mv+1:-1:1],[Ef_L_Xrange_SCL_max_vs,Ef_L_Xrange_SCL_min_vs(end:-1:1)],'b');
pep2.FaceColor = [ 0.8 0.8 1];      
pep2.EdgeColor = 'none';
hold on
ptp2=plot([1:Ns-mv+1],[Ef_L_Xrange_SCL_mean_vs],'--b');
hold on
pep22=fill([1:Ns-mv+1, Ns-mv+1:-1:1],[Ef_L_Xrange_SGD_max_svs,Ef_L_Xrange_SGD_min_svs(end:-1:1)],'r');
pep22.FaceColor = [1 0.8 0.8];      
pep22.EdgeColor = 'none';
hold on
pep222=plot([1:Ns-mv+1],[Ef_L_Xrange_SGD_mean_svs],'-.r');
legend([ptp2,pep222],{'$CL-SGD,\sigma^2=3 \times 10^{-5} $','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
ylabel('E(k)')
xlabel('k (time steps)')
%ylim([-2 0.1])

subplot(2,1,2)
pmp2=fill([1:Ns-mv+1, Ns-mv+1:-1:1],[Ef_L_Xrange_SCL_max_vb,Ef_L_Xrange_SCL_min_vb(end:-1:1)],'b');
pmp2.FaceColor = [ 0.8 0.8 1];      
pmp2.EdgeColor = 'none';
hold on
tmp2=plot([1:Ns-mv+1],[Ef_L_Xrange_SCL_mean_vb],'--b');
hold on
pep202=fill([1:Ns-mv+1, Ns-mv+1:-1:1],[Ef_L_Xrange_SGD_max_bvb,Ef_L_Xrange_SGD_min_bvb(end:-1:1)],'r');
pep202.FaceColor = [1 0.8 0.8];      
pep202.EdgeColor = 'none';
hold on
pzp202=plot([1:Ns-mv+1],[Ef_L_Xrange_SGD_mean_bvb],'-.r');
hold on
legend([tmp2,pzp202],{'$CL-SGD,\sigma^2=3 \times 10^{-3} $','$SGD, \sigma^2=3 \times 10^{-3}$'},'Interpreter','latex')
ylabel('E(k)')
xlabel('k (time steps)')
%ylim([-2 0.1])

figure(6)
subplot(2,1,1)
pzzp1=plot(xx_vs(1:NEs+1),fx_vs,'k','LineWidth',1.); % true f
 hold on
ppsp2=fill([xx_vs(1:NEs+1), xx_vs(NEs+1:-1:1)],[FxhatSCL_max_vs,FxhatSCL_min_vs(end:-1:1)],'b');% CL estimated f
ppsp2.FaceColor = [ 0.8 0.8 1];      
ppsp2.EdgeColor = 'none';
hold on
pttp2=plot([xx_vs(1:NEs+1)],[FxhatSCL_mean_vs],'--b');% CL estimated f
hold on
pp55=fill([xx_svs(1:NEs+1), xx_svs(NEs+1:-1:1)],[FxhatSGD_max_svs,FxhatSGD_min_svs(end:-1:1)],'r');% CL estimated f
pp55.FaceColor = [1 0.8 0.8];      
pp55.EdgeColor = 'none';
hold on
pp515=plot([xx_svs(1:NEs+1)],[FxhatSGD_mean_svs],'-.r');% CL estimated f
hold on
%title('Simple CL ')
%legend('f(x)','fhat(x)')
legend([pzzp1,pttp2,pp515],{'$True$','$CL-SGD,\sigma^2=3 \times 10^{-5} $','$SGD, \sigma^2=3 \times 10^{-5}$'},'Interpreter','latex')
xlabel('x')
ylabel('f(x)')
xlim([0 2])
%ylim([-0.6 0.1])

subplot(2,1,2)
pppp1=plot(xx_vb(1:NEs+1),fx_vb,'k','LineWidth',1.); % true f
 hold on
pmpp2=fill([xx_vb(1:NEs+1), xx_vb(NEs+1:-1:1)],[FxhatSCL_max_vb,FxhatSCL_min_vb(end:-1:1)],'b');% CL estimated f
pmpp2.FaceColor = [ 0.8 0.8 1];      
pmpp2.EdgeColor = 'none';
hold on
pmmm2=plot([xx_vb(1:NEs+1)],[FxhatSCL_mean_vb],'--b');% CL estimated f
hold on
pp66=fill([xx_bvb(1:NEs+1), xx_bvb(NEs+1:-1:1)],[FxhatSGD_max_bvb,FxhatSGD_min_bvb(end:-1:1)],'r');% CL estimated f
pp66.FaceColor = [1 0.8 0.8];      
pp66.EdgeColor = 'none';
hold on
pdp=plot([xx_bvb(1:NEs+1)],[FxhatSGD_mean_bvb],'-.r');% CL estimated f
hold on
legend([pppp1,pmmm2,pdp],{'$True$','$CL-SGD,\sigma^2=3 \times 10^{-3} $','$SGD, \sigma^2=3 \times 10^{-3}$'},'Interpreter','latex')
%title('Simple CL ')
%legend('f(x)','fhat(x)')
xlabel('x')
ylabel('f(x)')
xlim([0 2])
%ylim([-0.6 0.1])