function identifiercompleteedit_Compare
% with out filtering & without l
clear all;
close all;
clc;
%global P;


global x011
global x012

global x021
global x022

global x031
global x032


global kk
global T
global P

global gama
global gama1

global Ns
global a 

global amp

%global p7 p8 p9  p10 p11  p12  p13  p14  p15  p16  p17  p18  p19  p20  p21  p22  p23  p24       

% 
% p0=randn(1,62)*0.1;
% x0=[[0.5651    0.9001    0.2233   -0.4922   -0.0460   -0.4657    0.0759   -0.9188   -0.4195   -0.0364   -0.2251   -0.9024    0.3742	...
%     -0.2333    0.4039    0.1925   -0.6848    0.4131    0.5017    0.0831]...
%     ,zeros(1,40),zeros(1,20),p0];

% p0=randn(1,62)*0.1;
% x0=[randn(1,20)*0.1,zeros(1,40),p0];

% intial conditions
%p0=1*rand(1,18);
%x0=[randn(1,6),zeros(1,12),p0];

x0=[1.68914546929209,1.43704449250503,-2.25107830229662,0.356489129614670,-0.850241546233911,-0.299551039013598,0,0,0,0,0,0,0,0,0,0,0,0,0.576775860240990,0.944029193841664,0.871452148941678,0.507602207944769,0.788823165012461,0.473030640899700,0.828801698488441,0.322481588061953,0.976146535227356,0.278211040263144,0.0728308048124289,0.751223605872008,0.831188620534169,0.922338093331131,0.327024293370015,0.804069262570712,0.538250333300567,0.463294879079307];
%x0=[1.68914546929209,1.43704449250503,-2.25107830229662,0.356489129614670,-0.850241546233911,-0.299551039013598,0,0,0,0,0,0,0,0,0,0,0,0,p0];

save x0

x011=x0(1);
x012=x0(2);
x021=x0(3);
x022=x0(4);
x031=x0(5);
x032=x0(6);


%%%%% design parameter %%%%
%Ns=100000; 
Ns=10000; 
%________________________________________________________________________
%T=0.001;
T=0.05;
%________________________________________________________________________
% Learning weights
gama=3; 
%gama1=0.25;
gama1=0.3;

% Finite time power
a=0.5;


% Noise amplitude
%amp=3;
% amp=1;
amp=0;

%________________________________________________________________________
%memory stack length
S1=30;
%________________________________________________________________________

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fnite-time Distributed Concurrent Learning 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Q1=[];
% X=[];
% P=zeros(1,10);
QQ1=[]; QQ2=[]; QQ3=[];
XX1=[];XX2=[];XX3=[]; 

P=zeros(1,18);

kk=0;
for k=1:Ns
    k
    kk=kk+1;
    tspan=[0 T];
    [t,x]= ode45(@odefile,tspan,x0);
    
    %size(x,1)
    x1=x(size(x,1),1);
    x2=x(size(x,1),2);
    x3=x(size(x,1),3);
    x4=x(size(x,1),4);
    x5=x(size(x,1),5);
    x6=x(size(x,1),6);
    
    x7=x(size(x,1),7);
    x8=x(size(x,1),8);
    x9=x(size(x,1),9);
    x10=x(size(x,1),10);
    x11=x(size(x,1),11);
    x12=x(size(x,1),12);
    x13=x(size(x,1),13);
    x14=x(size(x,1),14);
    x15=x(size(x,1),15);
    x16=x(size(x,1),16);
    x17=x(size(x,1),17);
    x18=x(size(x,1),18);
    
    x19=x(size(x,1),19);
    x20=x(size(x,1),20);
    x21=x(size(x,1),21);
    x22=x(size(x,1),22);    
    x23=x(size(x,1),23);
    x24=x(size(x,1),24);
    x25=x(size(x,1),25);
    x26=x(size(x,1),26);
    x27=x(size(x,1),27);
    x28=x(size(x,1),28);
    x29=x(size(x,1),29);
    x30=x(size(x,1),30);
    x31=x(size(x,1),31);
    x32=x(size(x,1),32);
    x33=x(size(x,1),33);
    x34=x(size(x,1),34);
    x35=x(size(x,1),35);
    x36=x(size(x,1),36); 

%% to plot functions learning errors for the the whole range of x1=[-6,6] (rad) and x2=[-4,4] (rad/s) on time k
NEs=150;

x1L=-6; x2L=-4;  x3L=x1L;  x4L=x2L;  x5L=x1L;  x6L=x2L;
x1H=6;  x2H=4;   x3H=x1H;  x4H=x2H;  x5H=x1H;  x6H=x2H;

xx1(1)=x1L;
xx2(1)=x2L;
xx3(1)=x3L;
xx4(1)=x4L;
xx5(1)=x5L;
xx6(1)=x6L;

yy1=x1L:(x1H-x1L)/NEs:x1H;
yy2=x2L:(x2H-x2L)/NEs:x2H;
yy3=x3L:(x3H-x3L)/NEs:x3H;
yy4=x4L:(x4H-x4L)/NEs:x4H;
yy5=x5L:(x5H-x5L)/NEs:x5H;
yy6=x5L:(x6H-x6L)/NEs:x6H;

[Y11,Y21]=meshgrid(yy1,yy2);
[Y12,Y22]=meshgrid(yy3,yy4);
[Y13,Y23]=meshgrid(yy5,yy6);

[Y3_1,Y5_1]=meshgrid(yy3,yy5);
[Y1_2,Y5_2]=meshgrid(yy1,yy5);
[Y1_3,Y3_3]=meshgrid(yy1,yy3);


for j=1:NEs+1

% x values between [-6,6] 
xx1(j+1)=xx1(j)+(x1H-x1L)/NEs;
xx3(j+1)=xx3(j)+(x3H-x3L)/NEs;
xx5(j+1)=xx5(j)+(x5H-x5L)/NEs;

% x2 values between [-4,4] 
xx2(j+1)=xx2(j)+(x2H-x2L)/NEs;
xx4(j+1)=xx4(j)+(x4H-x4L)/NEs;
xx6(j+1)=xx6(j)+(x6H-x6L)/NEs;

% fxhat(j)=x12*exp(-(xx(j)-(-2))^2/(2*sp^2))+x13*exp(-(xx(j)-(-1))^2/(2*sp^2))+x14*exp(-(xx(j)-(0))^2/(2*sp^2))+x15*exp(-(xx(j)-(1))^2/(2*sp^2))+x16*exp(-(xx(j)-(2))^2/(2*sp^2));
% gxhat(j)=x17*(exp(-(xx(j)-(-2.1))^2/(2*sp^2)))+x18*(exp(-(xx(j)-(-1.1))^2/(2*sp^2)))+x19*(exp(-(xx(j)-(0.1))^2/(2*sp^2)))+x20*(exp(-(xx(j)-(1.1))^2/(2*sp^2)))+x21*(exp(-(xx(j)-(2.1))^2/(2*sp^2))); 

f11xhat(j)=x19*xx2(j);
f12xhat(j)=x20*sin(xx1(j))+x22*sin(xx1(j))*cos(xx1(j));
f13xhat(j)=x23*sin(xx3(j))*cos(xx3(j));
f14xhat(j)=x24*sin(xx5(j))*cos(xx5(j));
g1xhat(j)=x21; 

f21xhat(j)=x25*xx4(j);
f22xhat(j)=x26*sin(xx3(j))+x28*sin(xx3(j))*cos(xx3(j));
f23xhat(j)=x29*sin(xx1(j))*cos(xx1(j));
f24xhat(j)=x30*sin(xx5(j))*cos(xx5(j));
g2xhat(j)=x27; 

f31xhat(j)=x31*xx6(j);
f32xhat(j)=x32*sin(xx5(j))+x34*sin(xx5(j))*cos(xx5(j));
f33xhat(j)=x35*sin(xx3(j))*cos(xx3(j));
f34xhat(j)=x36*sin(xx1(j))*cos(xx1(j));
g3xhat(j)=x33; 

f11x(j)=1*xx2(j);
f12x(j)=5*sin(xx1(j))-3*sin(xx1(j))*cos(xx1(j));
f13x(j)=1.5*sin(xx3(j))*cos(xx3(j));
f14x(j)=1.5*sin(xx5(j))*cos(xx5(j));
g1x(j)=1;
 
f21x(j)=1*xx4(j);
f22x(j)=5*sin(xx3(j))-3.5*sin(xx3(j))*cos(xx3(j));
f23x(j)=1.5*sin(xx1(j))*cos(xx1(j));
f24x(j)=2*sin(xx5(j))*cos(xx5(j));
g2x(j)=1; 

f31x(j)=1*xx6(j);
f32x(j)=5*sin(xx5(j))-3.5*sin(xx5(j))*cos(xx5(j));
f33x(j)=2*sin(xx3(j))*cos(xx3(j));
f34x(j)=1.5*sin(xx1(j))*cos(xx1(j));
g3x(j)=1;

% fx(j)=0.5*xx(j)*sin(0.5*xx(j));
% gx(j)=(2+cos(xx(j)));

end

F1=sqrt(((1-x19).*(Y21)+(5-x20).*sin(Y11)+(-3-x22).*sin(Y11).*cos(Y11)).^2);
EF1_CL(k)=trapz(yy2,trapz(yy1,F1,2),1);
%ef11_Total_L_Xrange_CL(k)= trapz(xx2(1:NEs+1),sqrt((f11x-f11xhat).^2));
%ef12_Total_L_Xrange_CL(k)= trapz(xx1(1:NEs+1),sqrt((f12x-f12xhat).^2));
Delta1=sqrt(((1.5-x23).*sin(Y3_1).*cos(Y3_1)+(1.5-x24).*sin(Y5_1).*cos(Y5_1)).^2);
EDelta1_CL(k)=trapz(yy3,trapz(yy5,Delta1,2),1);
eg1_Total_L_Xrange_CL(k)= trapz(sqrt((g1x-g1xhat).^2));

F2=sqrt(((1-x25).*(Y22)+(5-x26).*sin(Y12)+(-3.5-x28).*sin(Y12).*cos(Y12)).^2);
EF2_CL(k)=trapz(yy4,trapz(yy3,F2,2),1);
%ef21_Total_L_Xrange_CL(k)= trapz(xx4(1:NEs+1),sqrt((f21x-f21xhat).^2));
%ef22_Total_L_Xrange_CL(k)= trapz(xx3(1:NEs+1),sqrt((f22x-f22xhat).^2));
Delta2=sqrt(((1.5-x29).*sin(Y1_2).*cos(Y1_2)+(2-x30).*sin(Y5_2).*cos(Y5_2)).^2);
EDelta2_CL(k)=trapz(yy1,trapz(yy5,Delta2,2),1);
eg2_Total_L_Xrange_CL(k)= trapz(sqrt((g2x-g2xhat).^2));

F3=sqrt(((1-x31).*(Y23)+(5-x32).*sin(Y13)+(-3.5-x34).*sin(Y13).*cos(Y13)).^2);
EF3_CL(k)=trapz(yy6,trapz(yy5,F3,2),1);
%ef31_Total_L_Xrange_CL(k)= trapz(xx6(1:NEs+1),sqrt((f31x-f31xhat).^2));
%ef32_Total_L_Xrange_CL(k)= trapz(xx5(1:NEs+1),sqrt((f32x-f32xhat).^2));
Delta3=sqrt(((2-x35).*sin(Y3_3).*cos(Y3_3)+(1.5-x36).*sin(Y1_3).*cos(Y1_3)).^2);
EDelta3_CL(k)=trapz(yy1,trapz(yy3,Delta3,2),1);
eg3_Total_L_Xrange_CL(k)= trapz(sqrt((g3x-g3xhat).^2));


%%

p1(k)=x1;
p2(k)=x2;
p3(k)=x3;
p4(k)=x4;
p5(k)=x5;
p6(k)=x6;

p7(k)=x19;
p8(k)=x20;
p9(k)=x21;
p10(k)=x22;
p11(k)=x23;
p12(k)=x24;
p13(k)=x25;
p14(k)=x26;
p15(k)=x27;
p16(k)=x28;
p17(k)=x29;
p18(k)=x30;
p19(k)=x31;
p20(k)=x32;
p21(k)=x33;
p22(k)=x34;
p23(k)=x35;
p24(k)=x36;

x0=[x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36];

tmm=(kk-1)*T+t(size(t,1)); 
%%-----------------------------------------------------------------------
 if k<=S1
     QQ1=[QQ1; x7 1 x8 x9 x10 x14 x18 1];
     QQ2=[QQ2; x11 1 x12 x13 x14 x10 x18 1];
     QQ3=[QQ3; x15 1 x16 x17 x18 x14 x10 1];
     
     XX1=[XX1; x1 x2];
     XX2=[XX2; x3 x4];
     XX3=[XX3; x5 x6];
     
 end
%%------------------------------------------------------------------------
ee11=XX1(:,1)-(QQ1(:,1:2)*[x19;x011]);
ee12=XX1(:,2)-(QQ1(:,3:8)*[x20;x21;x22;x23;x24;x012]);
%size(ee11)

ee21=XX2(:,1)-(QQ2(:,1:2)*[x25;x021]);
ee22=XX2(:,2)-(QQ2(:,3:8)*[x26;x27;x28;x29;x30;x022]);

ee31=XX3(:,1)-(QQ3(:,1:2)*[x31;x031]);
ee32=XX3(:,2)-(QQ3(:,3:8)*[x32;x33;x34;x35;x36;x032]);


if k>S1+1
    for i=1:S1
ee11(i)=(abs(ee11(i)))^a*sign(ee11(i));
ee12(i)=(abs(ee12(i)))^a*sign(ee12(i));    
    
ee21(i)=(abs(ee21(i)))^a*sign(ee21(i));
ee22(i)=(abs(ee22(i)))^a*sign(ee22(i));
    
ee31(i)=(abs(ee31(i)))^a*sign(ee31(i));
ee32(i)=(abs(ee32(i)))^a*sign(ee32(i));

    end
end
%%--------------------------------------------------------------------
%subsystem 1

P(1)=sum(ee11.*QQ1(:,1));
P(2)=sum(ee12.*QQ1(:,3));
P(3)=sum(ee12.*QQ1(:,4));
P(4)=sum(ee12.*QQ1(:,5));
P(5)=sum(ee12.*QQ1(:,6));
P(6)=sum(ee12.*QQ1(:,7));

P(7)=sum(ee21.*QQ2(:,1));
P(8)=sum(ee22.*QQ2(:,3));
P(9)=sum(ee22.*QQ2(:,4));
P(10)=sum(ee22.*QQ2(:,5));
P(11)=sum(ee22.*QQ2(:,6));
P(12)=sum(ee22.*QQ2(:,7));


P(13)=sum(ee31.*QQ3(:,1));
P(14)=sum(ee32.*QQ3(:,3));
P(15)=sum(ee32.*QQ3(:,4));
P(16)=sum(ee32.*QQ3(:,5));
P(17)=sum(ee32.*QQ3(:,6));
P(18)=sum(ee32.*QQ3(:,7));

end
save p1
%%----------------------------------------------------------------------
q1=rank(QQ1);
q2=rank(QQ2);
q3=rank(QQ3);

%-----------------------------------------------------------------------
% ee1;
% ee2;
%PP=x(size(x,1),16:25)

NN = max(size(p1));
TT=Ns*T;
tt=linspace(0,TT,NN);

figure(1)
plot(tt,p1,'r')
hold on
plot(tt,p2,'g')
ylabel('Subsystem 1 states')
xlabel('Time (s)')
legend('x_{11}','x_{12}')

figure(2)
plot(tt,p3,'r')
hold on
plot(tt,p4,'g')
ylabel('Subsystem 2 states')
xlabel('Time (s)')
legend('x_{21}','x_{22}')

figure(3)
plot(tt,p5,'r')
hold on
plot(tt,p6,'g')
ylabel('Subsystem 3 states')
xlabel('Time (s)')
legend('x_{31}','x_{32}')

% subsystem 1
true11=1*ones(1,Ns);
true12=5*ones(1,Ns);
true13=1*ones(1,Ns);
true14=-3*ones(1,Ns);
true15=1.5*ones(1,Ns);
true16=1.5*ones(1,Ns);

% subsystem 2
true21=1*ones(1,Ns);
true22=5*ones(1,Ns);
true23=1*ones(1,Ns);
true24=-3.5*ones(1,Ns);
true25=1.5*ones(1,Ns);
true26=2*ones(1,Ns);

% subsystem 3
true31=1*ones(1,Ns);
true32=5*ones(1,Ns);
true33=1*ones(1,Ns);
true34=-3.5*ones(1,Ns);
true35=2*ones(1,Ns);
true36=1.5*ones(1,Ns);

figure(11)
subplot(3,2,1)
CL_sys1=plot(tt,p7,'b','LineWidth',1.5);
hold on
plot(tt,p8,'b','LineWidth',1.5)
hold on
plot(tt,p9,'b','LineWidth',1.5)
hold on
plot(tt,p10,'b','LineWidth',1.5)
hold on
plot(tt,p11,'b','LineWidth',1.5)
hold on
plot(tt,p12,'b','LineWidth',1.5)
hold on
True_sys1=plot([1:Ns]*T,true11,'--k');
hold on
plot([1:Ns]*T,true12,'--k')
hold on
plot([1:Ns]*T,true13,'--k')
hold on
plot([1:Ns]*T,true14,'--k')
hold on
plot([1:Ns]*T,true15,'--k')
hold on
plot([1:Ns]*T,true16,'--k')
ylim([-4 6])
title('Subsystem 1')
ylabel('Parameters by FTDCL')
xlabel('Time (s)')
legend([CL_sys1,True_sys1],{'FTDCL','True'});
%legend('p_1^1','p_2^1','p_3^1','p_4^1','p_5^1','p_6^1')

%figure(12)
subplot(3,2,3)
CL_sys2=plot(tt,p13,'r','LineWidth',1.5);
hold on
plot(tt,p14,'r','LineWidth',1.5)
hold on
plot(tt,p15,'r','LineWidth',1.5)
hold on
plot(tt,p16,'r','LineWidth',1.5)
hold on
plot(tt,p17,'r','LineWidth',1.5)
hold on
plot(tt,p18,'r','LineWidth',1.5)
hold on
True_sys2=plot([1:Ns]*T,true21,'--k');
hold on
plot([1:Ns]*T,true22,'--k')
hold on
plot([1:Ns]*T,true23,'--k')
hold on
plot([1:Ns]*T,true24,'--k')
hold on
plot([1:Ns]*T,true25,'--k')
hold on
plot([1:Ns]*T,true26,'--k')
ylim([-4 6])
title('Subsystem 2')
ylabel('Parameters by FTDCL')
xlabel('Time (s)')
legend([CL_sys2,True_sys2],{'FTDCL','True'});
%legend('p_1^2','p_2^2','p_3^2','p_4^2','p_5^2','p_6^2')


%figure(13)
subplot(3,2,5)
CL_sys3=plot(tt,p19,'g','LineWidth',1.5);
hold on
plot(tt,p20,'g','LineWidth',1.5)
hold on
plot(tt,p21,'g','LineWidth',1.5)
hold on
plot(tt,p22,'g','LineWidth',1.5)
hold on
plot(tt,p23,'g','LineWidth',1.5)
hold on
plot(tt,p24,'g','LineWidth',1.5)
hold on
True_sys3=plot([1:Ns]*T,true31,'--k');
hold on
plot([1:Ns]*T,true32,'--k')
hold on
plot([1:Ns]*T,true33,'--k')
hold on
plot([1:Ns]*T,true34,'--k')
hold on
plot([1:Ns]*T,true35,'--k')
hold on
plot([1:Ns]*T,true36,'--k')
ylim([-4 6])
xlabel('Time (s)')
title('Subsystem 3')
ylabel('Parameters by FTDCL')
legend([CL_sys3,True_sys3],{'FTDCL','True'});
%legend('p_1^3','p_2^3','p_3^3','p_4^3','p_5^3','p_6^3')


figure(5)
subplot(3,1,1)
%CL_ef=plot([1:Ns]*T,ef11_L_Xrange_CL,'g-o','LineWidth',1.5,'MarkerIndices',1:2000:length(ef_L_Xrange_CL));
% plot([1:Ns]*T,EF11,'b','LineWidth',1.5);
% hold on 
CL_ef1=plot([1:Ns]*T,EF1_CL,'b','LineWidth',1.5);
hold on 
% CL_ef12=plot([1:Ns]*T,ef12_Total_L_Xrange_CL,'b','LineWidth',1.5);
%  hold on 
% plot([1:Ns]*T,ef12_Total_L_Xrange_CL,'--r','LineWidth',1.5);
ylabel('E_{f_1}(t)')
%xlabel('Time (s)')
%ylim([-2 0.1])
%hold on

subplot(3,1,3)
% CL_eg=plot([1:Ns]*T,EG1,'g','LineWidth',1.5);
% hold on 
CL_edelta1=plot([1:Ns]*T,EDelta1_CL,'b','LineWidth',1.5);
hold on 
ylabel('E_{\Delta_1}(t)')
xlabel('Time (s)')
%hold on

subplot(3,1,2)
% CL_eg=plot([1:Ns]*T,EG1,'g','LineWidth',1.5);
% hold on 
CL_eg1=plot([1:Ns]*T,eg1_Total_L_Xrange_CL,'b','LineWidth',1.5);
hold on 
ylabel('E_{g_1}(t)')
%xlabel('Time (s)')
%hold on


figure(6)
subplot(3,1,1)
%CL_ef=plot([1:Ns]*T,ef11_L_Xrange_CL,'g-o','LineWidth',1.5,'MarkerIndices',1:2000:length(ef_L_Xrange_CL));
% plot([1:Ns]*T,EF21,'b','LineWidth',1.5);
% hold on 
CL_ef2=plot([1:Ns]*T,EF2_CL,'r','LineWidth',1.5);
hold on 
% plot([1:Ns]*T,ef22_Total_L_Xrange_CL,'b','LineWidth',1.5);
%  hold on 
% plot([1:Ns]*T,ef22_Total_L_Xrange_CL,'--r','LineWidth',1.5);
ylabel('E_{f_2}(t)')
%xlabel('Time (s)')

subplot(3,1,3)
% CL_eg=plot([1:Ns]*T,EG2,'g','LineWidth',1.5);
% hold on
CL_edelta2=plot([1:Ns]*T,EDelta2_CL,'r','LineWidth',1.5);
hold on 
ylabel('E_{\Delta_2}(t)')
xlabel('Time (s)')
%hold on

subplot(3,1,2)
% CL_eg=plot([1:Ns]*T,EG2,'g','LineWidth',1.5);
% hold on
CL_eg2=plot([1:Ns]*T,eg2_Total_L_Xrange_CL,'r','LineWidth',1.5);
hold on 
ylabel('E_{g_2}(t)')
%xlabel('Time (s)')
%hold on

figure(7)
subplot(3,1,1)
%CL_ef=plot([1:Ns]*T,ef11_L_Xrange_CL,'g-o','LineWidth',1.5,'MarkerIndices',1:2000:length(ef_L_Xrange_CL));
% plot([1:Ns]*T,EF31,'b','LineWidth',1.5);
% hold on 
CL_ef3=plot([1:Ns]*T,EF3_CL,'g','LineWidth',1.5);
hold on 
% plot([1:Ns]*T,ef32_Total_L_Xrange_CL,'b','LineWidth',1.5);
%  hold on 
% plot([1:Ns]*T,ef32_Total_L_Xrange_CL,'--r','LineWidth',1.5);
ylabel('E_{f_3}(t)')
%xlabel('Time (s)')
%ylim([-2 0.1])
%hold on

subplot(3,1,3)
% CL_eg=plot([1:Ns]*T,EG3,'g','LineWidth',1.5);
% hold on
CL_edelta3=plot([1:Ns]*T,EDelta3_CL,'g','LineWidth',1.5);
hold on 
ylabel('E_{\Delta_3}(t)')
xlabel('Time (s)')
%hold on

subplot(3,1,2)
% CL_eg=plot([1:Ns]*T,EG3,'g','LineWidth',1.5);
% hold on
CL_eg3=plot([1:Ns]*T,eg3_Total_L_Xrange_CL,'g','LineWidth',1.5);
hold on 
ylabel('E_{g_3}(t)')
%xlabel('Time (s)')
%hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fnite-time Distributed Gradient Descent 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0=[1.68914546929209,1.43704449250503,-2.25107830229662,0.356489129614670,-0.850241546233911,-0.299551039013598,0,0,0,0,0,0,0,0,0,0,0,0,0.576775860240990,0.944029193841664,0.871452148941678,0.507602207944769,0.788823165012461,0.473030640899700,0.828801698488441,0.322481588061953,0.976146535227356,0.278211040263144,0.0728308048124289,0.751223605872008,0.831188620534169,0.922338093331131,0.327024293370015,0.804069262570712,0.538250333300567,0.463294879079307];

kk=0;
for k=1:Ns
    k
    kk=kk+1;
    tspan=[0 T];
    [t,x]= ode45(@odefile_GD,tspan,x0);
    
    %size(x,1)
    x1=x(size(x,1),1);
    x2=x(size(x,1),2);
    x3=x(size(x,1),3);
    x4=x(size(x,1),4);
    x5=x(size(x,1),5);
    x6=x(size(x,1),6);
    
    x7=x(size(x,1),7);
    x8=x(size(x,1),8);
    x9=x(size(x,1),9);
    x10=x(size(x,1),10);
    x11=x(size(x,1),11);
    x12=x(size(x,1),12);
    x13=x(size(x,1),13);
    x14=x(size(x,1),14);
    x15=x(size(x,1),15);
    x16=x(size(x,1),16);
    x17=x(size(x,1),17);
    x18=x(size(x,1),18);
    
    x19=x(size(x,1),19);
    x20=x(size(x,1),20);
    x21=x(size(x,1),21);
    x22=x(size(x,1),22);    
    x23=x(size(x,1),23);
    x24=x(size(x,1),24);
    x25=x(size(x,1),25);
    x26=x(size(x,1),26);
    x27=x(size(x,1),27);
    x28=x(size(x,1),28);
    x29=x(size(x,1),29);
    x30=x(size(x,1),30);
    x31=x(size(x,1),31);
    x32=x(size(x,1),32);
    x33=x(size(x,1),33);
    x34=x(size(x,1),34);
    x35=x(size(x,1),35);
    x36=x(size(x,1),36); 

%% to plot functions learning errors for the the whole range of x1=[-6,6] (rad) and x2=[-4,4] (rad/s) on time k
NEs=150;

x1L=-6; x2L=-4;  x3L=x1L;  x4L=x2L;  x5L=x1L;  x6L=x2L;
x1H=6;  x2H=4;   x3H=x1H;  x4H=x2H;  x5H=x1H;  x6H=x2H;

xx1(1)=x1L;
xx2(1)=x2L;
xx3(1)=x3L;
xx4(1)=x4L;
xx5(1)=x5L;
xx6(1)=x6L;

yy1=x1L:(x1H-x1L)/NEs:x1H;
yy2=x2L:(x2H-x2L)/NEs:x2H;
yy3=x3L:(x3H-x3L)/NEs:x3H;
yy4=x4L:(x4H-x4L)/NEs:x4H;
yy5=x5L:(x5H-x5L)/NEs:x5H;
yy6=x5L:(x6H-x6L)/NEs:x6H;

[Y3_1,Y5_1]=meshgrid(yy3,yy5);
[Y1_2,Y5_2]=meshgrid(yy1,yy5);
[Y1_3,Y3_3]=meshgrid(yy1,yy3);


for j=1:NEs+1

% x values between [-6,6] 
xx1(j+1)=xx1(j)+(x1H-x1L)/NEs;
xx3(j+1)=xx3(j)+(x3H-x3L)/NEs;
xx5(j+1)=xx5(j)+(x5H-x5L)/NEs;

% x2 values between [-4,4] 
xx2(j+1)=xx2(j)+(x2H-x2L)/NEs;
xx4(j+1)=xx4(j)+(x4H-x4L)/NEs;
xx6(j+1)=xx6(j)+(x6H-x6L)/NEs;

% fxhat(j)=x12*exp(-(xx(j)-(-2))^2/(2*sp^2))+x13*exp(-(xx(j)-(-1))^2/(2*sp^2))+x14*exp(-(xx(j)-(0))^2/(2*sp^2))+x15*exp(-(xx(j)-(1))^2/(2*sp^2))+x16*exp(-(xx(j)-(2))^2/(2*sp^2));
% gxhat(j)=x17*(exp(-(xx(j)-(-2.1))^2/(2*sp^2)))+x18*(exp(-(xx(j)-(-1.1))^2/(2*sp^2)))+x19*(exp(-(xx(j)-(0.1))^2/(2*sp^2)))+x20*(exp(-(xx(j)-(1.1))^2/(2*sp^2)))+x21*(exp(-(xx(j)-(2.1))^2/(2*sp^2))); 

f11xhat(j)=x19*xx2(j);
f12xhat(j)=x20*sin(xx1(j))+x22*sin(xx1(j))*cos(xx1(j));
f13xhat(j)=x23*sin(xx3(j))*cos(xx3(j));
f14xhat(j)=x24*sin(xx5(j))*cos(xx5(j));
g1xhat(j)=x21; 

f21xhat(j)=x25*xx4(j);
f22xhat(j)=x26*sin(xx3(j))+x28*sin(xx3(j))*cos(xx3(j));
f23xhat(j)=x29*sin(xx1(j))*cos(xx1(j));
f24xhat(j)=x30*sin(xx5(j))*cos(xx5(j));
g2xhat(j)=x27; 

f31xhat(j)=x31*xx6(j);
f32xhat(j)=x32*sin(xx5(j))+x34*sin(xx5(j))*cos(xx5(j));
f33xhat(j)=x35*sin(xx3(j))*cos(xx3(j));
f34xhat(j)=x36*sin(xx1(j))*cos(xx1(j));
g3xhat(j)=x33; 

f11x(j)=1*xx2(j);
f12x(j)=5*sin(xx1(j))-3*sin(xx1(j))*cos(xx1(j));
f13x(j)=1.5*sin(xx3(j))*cos(xx3(j));
f14x(j)=1.5*sin(xx5(j))*cos(xx5(j));
g1x(j)=1;
 
f21x(j)=1*xx4(j);
f22x(j)=5*sin(xx3(j))-3.5*sin(xx3(j))*cos(xx3(j));
f23x(j)=1.5*sin(xx1(j))*cos(xx1(j));
f24x(j)=2*sin(xx5(j))*cos(xx5(j));
g2x(j)=1; 

f31x(j)=1*xx6(j);
f32x(j)=5*sin(xx5(j))-3.5*sin(xx5(j))*cos(xx5(j));
f33x(j)=2*sin(xx3(j))*cos(xx3(j));
f34x(j)=1.5*sin(xx1(j))*cos(xx1(j));
g3x(j)=1;

end

F1=sqrt(((1-x19).*(Y21)+(5-x20).*sin(Y11)+(-3-x22).*sin(Y11).*cos(Y11)).^2);
EF1_GD(k)=trapz(yy2,trapz(yy1,F1,2),1);
%ef11_Total_L_Xrange_CL(k)= trapz(xx2(1:NEs+1),sqrt((f11x-f11xhat).^2));
%ef12_Total_L_Xrange_CL(k)= trapz(xx1(1:NEs+1),sqrt((f12x-f12xhat).^2));
Delta1=sqrt(((1.5-x23).*sin(Y3_1).*cos(Y3_1)+(1.5-x24).*sin(Y5_1).*cos(Y5_1)).^2);
EDelta1_GD(k)=trapz(yy3,trapz(yy5,Delta1,2),1);
eg1_Total_L_Xrange_GD(k)= trapz(sqrt((g1x-g1xhat).^2));

F2=sqrt(((1-x25).*(Y22)+(5-x26).*sin(Y12)+(-3.5-x28).*sin(Y12).*cos(Y12)).^2);
EF2_GD(k)=trapz(yy4,trapz(yy3,F2,2),1);
%ef21_Total_L_Xrange_CL(k)= trapz(xx4(1:NEs+1),sqrt((f21x-f21xhat).^2));
%ef22_Total_L_Xrange_CL(k)= trapz(xx3(1:NEs+1),sqrt((f22x-f22xhat).^2));
Delta2=sqrt(((1.5-x29).*sin(Y1_2).*cos(Y1_2)+(2-x30).*sin(Y5_2).*cos(Y5_2)).^2);
EDelta2_GD(k)=trapz(yy1,trapz(yy5,Delta2,2),1);
eg2_Total_L_Xrange_GD(k)= trapz(sqrt((g2x-g2xhat).^2));

F3=sqrt(((1-x31).*(Y23)+(5-x32).*sin(Y13)+(-3.5-x34).*sin(Y13).*cos(Y13)).^2);
EF3_GD(k)=trapz(yy6,trapz(yy5,F3,2),1);
%ef31_Total_L_Xrange_CL(k)= trapz(xx6(1:NEs+1),sqrt((f31x-f31xhat).^2));
%ef32_Total_L_Xrange_CL(k)= trapz(xx5(1:NEs+1),sqrt((f32x-f32xhat).^2));
Delta3=sqrt(((2-x35).*sin(Y3_3).*cos(Y3_3)+(1.5-x36).*sin(Y1_3).*cos(Y1_3)).^2);
EDelta3_GD(k)=trapz(yy1,trapz(yy3,Delta3,2),1);
eg3_Total_L_Xrange_GD(k)= trapz(sqrt((g3x-g3xhat).^2));


%%

p1(k)=x1;
p2(k)=x2;
p3(k)=x3;
p4(k)=x4;
p5(k)=x5;
p6(k)=x6;

p7(k)=x19;
p8(k)=x20;
p9(k)=x21;
p10(k)=x22;
p11(k)=x23;
p12(k)=x24;
p13(k)=x25;
p14(k)=x26;
p15(k)=x27;
p16(k)=x28;
p17(k)=x29;
p18(k)=x30;
p19(k)=x31;
p20(k)=x32;
p21(k)=x33;
p22(k)=x34;
p23(k)=x35;
p24(k)=x36;

x0=[x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36];


end


%-----------------------------------------------------------------------
% ee1;
% ee2;
%PP=x(size(x,1),16:25)

NN = max(size(p1));
TT=Ns*T;
tt=linspace(0,TT,NN);

% figure(1)
% plot(tt,p1,'r')
% hold on
% plot(tt,p2,'g')
% 
% 
% figure(2)
% plot(tt,p3,'r')
% hold on
% plot(tt,p4,'g')
% 
% figure(3)
% plot(tt,p5,'r')
% hold on
% plot(tt,p6,'g')

figure(11)
subplot(3,2,2)
CL_sys1=plot(tt,p7,'b','LineWidth',1.5);
hold on
plot(tt,p8,'b','LineWidth',1.5)
hold on
plot(tt,p9,'b','LineWidth',1.5)
hold on
plot(tt,p10,'b','LineWidth',1.5)
hold on
plot(tt,p11,'b','LineWidth',1.5)
hold on
plot(tt,p12,'b','LineWidth',1.5)
hold on
True_sys1=plot([1:Ns]*T,true11,'--k');
hold on
plot([1:Ns]*T,true12,'--k')
hold on
plot([1:Ns]*T,true13,'--k')
hold on
plot([1:Ns]*T,true14,'--k')
hold on
plot([1:Ns]*T,true15,'--k')
hold on
plot([1:Ns]*T,true16,'--k')
ylim([-4 6])
title('Subsystem 1')
ylabel('Parameters by FTDGD')
xlabel('Time (s)')
legend([CL_sys1,True_sys1],{'FTDGD','True'});
%legend('p_1^1','p_2^1','p_3^1','p_4^1','p_5^1','p_6^1')

%figure(12)
subplot(3,2,4)
CL_sys2=plot(tt,p13,'r','LineWidth',1.5);
hold on
plot(tt,p14,'r','LineWidth',1.5)
hold on
plot(tt,p15,'r','LineWidth',1.5)
hold on
plot(tt,p16,'r','LineWidth',1.5)
hold on
plot(tt,p17,'r','LineWidth',1.5)
hold on
plot(tt,p18,'r','LineWidth',1.5)
hold on
True_sys2=plot([1:Ns]*T,true21,'--k');
hold on
plot([1:Ns]*T,true22,'--k')
hold on
plot([1:Ns]*T,true23,'--k')
hold on
plot([1:Ns]*T,true24,'--k')
hold on
plot([1:Ns]*T,true25,'--k')
hold on
plot([1:Ns]*T,true26,'--k')
ylim([-4 6])
title('Subsystem 2')
ylabel('Parameters by FTDGD')
xlabel('Time (s)')
legend([CL_sys2,True_sys2],{'FTDGD','True'});
%legend('p_1^2','p_2^2','p_3^2','p_4^2','p_5^2','p_6^2')


%figure(13)
subplot(3,2,6)
CL_sys3=plot(tt,p19,'g','LineWidth',1.5);
hold on
plot(tt,p20,'g','LineWidth',1.5)
hold on
plot(tt,p21,'g','LineWidth',1.5)
hold on
plot(tt,p22,'g','LineWidth',1.5)
hold on
plot(tt,p23,'g','LineWidth',1.5)
hold on
plot(tt,p24,'g','LineWidth',1.5)
hold on
True_sys3=plot([1:Ns]*T,true31,'--k');
hold on
plot([1:Ns]*T,true32,'--k')
hold on
plot([1:Ns]*T,true33,'--k')
hold on
plot([1:Ns]*T,true34,'--k')
hold on
plot([1:Ns]*T,true35,'--k')
hold on
plot([1:Ns]*T,true36,'--k')
ylim([-4 6])
xlabel('Time (s)')
title('Subsystem 3')
ylabel('Parameters by FTDGD')
legend([CL_sys3,True_sys3],{'FTDGD','True'});
%legend('p_1^3','p_2^3','p_3^3','p_4^3','p_5^3','p_6^3')


figure(5)
subplot(3,1,1)
%CL_ef=plot([1:Ns]*T,ef11_L_Xrange_CL,'g-o','LineWidth',1.5,'MarkerIndices',1:2000:length(ef_L_Xrange_CL));
% plot([1:Ns]*T,EF11,'b','LineWidth',1.5);
% hold on 
% plot([1:Ns]*T,ef11_Total_L_Xrange_GD,'--r','LineWidth',1.5);
% hold on 
GD_ef1=plot([1:Ns]*T,EF1_GD,'-.k','LineWidth',1);
% hold on 
% plot([1:Ns]*T,ef12_Total_L_Xrange_CL,'--r','LineWidth',1.5);
ylabel('E_{f_1}(t)')
legend([CL_ef1,GD_ef1],{'FTDCL','FTDGD'});
%xlabel('Time (s)')
%ylim([-2 0.1])
%hold on

subplot(3,1,3)
% CL_eg=plot([1:Ns]*T,EG1,'g','LineWidth',1.5);
% hold on 
GD_edelta1=plot([1:Ns]*T,EDelta1_GD,'-.k','LineWidth',1);
ylabel('E_{\Delta_1}(t)')
xlabel('Time (s)')
legend([CL_edelta1,GD_edelta1],{'FTDCL','FTDGD'});
%hold on

subplot(3,1,2)
% CL_eg=plot([1:Ns]*T,EG1,'g','LineWidth',1.5);
% hold on 
GD_eg1=plot([1:Ns]*T,eg1_Total_L_Xrange_GD,'-.k','LineWidth',1);
ylabel('E_{g_1}(t)')
legend([CL_eg1,GD_eg1],{'FTDCL','FTDGD'});
%xlabel('Time (s)')
%hold on


figure(6)
subplot(3,1,1)
%CL_ef=plot([1:Ns]*T,ef11_L_Xrange_CL,'g-o','LineWidth',1.5,'MarkerIndices',1:2000:length(ef_L_Xrange_CL));
% plot([1:Ns]*T,EF21,'b','LineWidth',1.5);
% hold on 
GD_ef2=plot([1:Ns]*T,EF2_GD,'-.k','LineWidth',1);
% hold on 
% plot([1:Ns]*T,ef22_Total_L_Xrange_GD,'--b','LineWidth',1.5);
% hold on 
% plot([1:Ns]*T,ef22_Total_L_Xrange_CL,'--r','LineWidth',1.5);
ylabel('E_{f_2}(t)')
legend([CL_ef2,GD_ef2],{'FTDCL','FTDGD'});
%xlabel('Time (s)')

subplot(3,1,3)
% CL_eg=plot([1:Ns]*T,EG2,'g','LineWidth',1.5);
% hold on
GD_edelta2=plot([1:Ns]*T,EDelta2_GD,'-.k','LineWidth',1);
ylabel('E_{\Delta_2}(t)')
xlabel('Time (s)')
legend([CL_edelta2,GD_edelta2],{'FTDCL','FTDGD'});
%hold on

subplot(3,1,2)
% CL_eg=plot([1:Ns]*T,EG2,'g','LineWidth',1.5);
% hold on
GD_eg2=plot([1:Ns]*T,eg2_Total_L_Xrange_GD,'-.k','LineWidth',1);
ylabel('E_{g_2}(t)')
legend([CL_eg2,GD_eg2],{'FTDCL','FTDGD'});
%xlabel('Time (s)')
%hold on

figure(7)
subplot(3,1,1)
%CL_ef=plot([1:Ns]*T,ef11_L_Xrange_CL,'g-o','LineWidth',1.5,'MarkerIndices',1:2000:length(ef_L_Xrange_CL));
% plot([1:Ns]*T,EF31,'b','LineWidth',1.5);
% hold on 
GD_ef3=plot([1:Ns]*T,EF3_GD,'-.k','LineWidth',1);
% hold on 
% plot([1:Ns]*T,ef32_Total_L_Xrange_GD,'--b','LineWidth',1.5);
% hold on 
% plot([1:Ns]*T,ef32_Total_L_Xrange_CL,'--r','LineWidth',1.5);
ylabel('E_{f_3}(t)')
legend([CL_ef3,GD_ef3],{'FTDCL','FTDGD'});
%xlabel('Time (s)')
%ylim([-2 0.1])
%hold on

subplot(3,1,3)
% CL_eg=plot([1:Ns]*T,EG3,'g','LineWidth',1.5);
% hold on
GD_edelta3=plot([1:Ns]*T,EDelta3_GD,'-.k','LineWidth',1);
ylabel('E_{\Delta_3}(t)')
xlabel('Time (s)')
legend([CL_edelta3,GD_edelta3],{'FTDCL','FTDGD'});
%hold on

subplot(3,1,2)
% CL_eg=plot([1:Ns]*T,EG3,'g','LineWidth',1.5);
% hold on
GD_eg3=plot([1:Ns]*T,eg3_Total_L_Xrange_GD,'-.k','LineWidth',1);
ylabel('E_{g_3}(t)')
legend([CL_eg3,GD_eg3],{'FTDCL','FTDGD'});
%xlabel('Time (s)')
%hold on

%__________________________________________________________________________________________________

function xdot=odefile(t,x)
%global P;
%global P1

global x011
global x012

global x021
global x022

global x031
global x032

global kk
global T
global P

global gama
global gama1

global Ns
global a 

global amp

g=10;
m1=0.25; m2=0.25; m3=0.25; m4=0.25; m5=0.25; m6=0.25; m7=0.25; m8=0.25; m9=0.25; m10=0.25;
l=2;
aa=1;
k1=1.5;  
k2=2;  
%k3=0.5;
k3=1.5;


 
 %a=100;
tm=(kk-1)*T+t;



ns1=1; ns2=1; ns3=1; ns4=1; ns5=1;
ns6=1; ns7=1; ns8=1; ns9=1; ns10=1;

% ns1= 1+([x(21),x(22),x(23),x(24),x(28)]*[x(21),x(22),x(23),x(24),x(28)]')+([x(61),x(62)]*[x(61),x(62)]');
% ns2= 1+([x(25),x(26),x(27),x(28),x(24),x(32)]*[x(25),x(26),x(27),x(28),x(24),x(32)]')+([x(63),x(64)]*[x(63),x(64)]');
% ns3= 1+([x(29),x(30),x(31),x(32),x(28),x(36)]*[x(29),x(30),x(31),x(32),x(28),x(36)]')+([x(65),x(66)]*[x(65),x(66)]');
% ns4= 1+([x(33),x(34),x(35),x(36),x(32),x(40),x(48)]*[x(33),x(34),x(35),x(36),x(32),x(40),x(48)]')+([x(67),x(68)]*[x(67),x(68)]');
% ns5= 1+([x(37),x(38),x(39),x(40),x(36),x(44)]*[x(37),x(38),x(39),x(40),x(36),x(44)]')+([x(69),x(70)]*[x(69),x(70)]');
% ns6= 1+([x(41),x(42),x(43),x(44),x(40),x(48),x(52)]*[x(41),x(42),x(43),x(44),x(40),x(48),x(52)]')+([x(71),x(72)]*[x(71),x(72)]');
% ns7= 1+([x(45),x(46),x(47),x(48),x(36),x(44)]*[x(45),x(46),x(47),x(48),x(36),x(44)]')+([x(73),x(74)]*[x(73),x(74)]');
% ns8= 1+([x(49),x(50),x(51),x(52),x(44),x(56),x(60)]*[x(49),x(50),x(51),x(52),x(44),x(56),x(60)]')+([x(75),x(76)]*[x(75),x(76)]');
% ns9= 1+([x(53),x(54),x(55),x(56),x(52),x(60)]*[x(53),x(54),x(55),x(56),x(52),x(60)]')+([x(77),x(78)]*[x(77),x(78)]');
% ns10=1+([x(57),x(58),x(59),x(60),x(52),x(56)]*[x(57),x(58),x(59),x(60),x(52),x(56)]')+([x(79),x(80)]*[x(79),x(80)]');

x1=x(1);
x2=x(2);
x3=x(3);
x4=x(4);
x5=x(5);
x6=x(6);

x7=x(7);
x8=x(8);
x9=x(9);
x10=x(10);
x11=x(11);
x12=x(12);
x13=x(13);
x14=x(14);
x15=x(15);
x16=x(16);
x17=x(17);
x18=x(18);

x19=x(19);
x20=x(20);
x21=x(21);
x22=x(22);
x23=x(23);
x24=x(24);
x25=x(25);
x26=x(26);
x27=x(27);
x28=x(28);
x29=x(29);
x30=x(30);
x31=x(31);
x32=x(32);
x33=x(33);
x34=x(34);
x35=x(35);
x36=x(36);


tmm=(kk-1)*T+t(size(t,1));

Y1=x19*x7+x011;
Y2=x20*x8+x21*x9+x22*x10+x23*x14+x24*x18+x012;

Y3=x25*x11+x021;
Y4=x26*x12+x27*x13+x28*x14+x29*x10+x30*x18+x022;

Y5=x31*x15+x031;
Y6=x32*x16+x33*x17+x34*x18+x35*x14+x36*x10+x032;


e1=x1-Y1;
e2=x2-Y2;

e3=x3-Y3;
e4=x4-Y4;

e5=x5-Y5;
e6=x6-Y6;

% % too good for stabilization but not good for identification
%  u1=[-40 -40]*[x1,x2]'+[-1 0]*[x3,x4]';
%  u2=[-40 -40]*[x3,x4]'+[-3 0]*[x1,x2]';

% % good for identification
%  u1=[-40 -40]*[x1,x2]'+[-1 0]*[x3,x4]';
%  u2=[-40 -40]*[x3,x4]'+[-3 0]*[x1,x2]'+[-3 0]*[x5,x6]';
%  u3=[-40 -40]*[x5,x6]'+[-1 0]*[x3,x4]';

% % good for identification
%  u1=[-5 -5]*[x1,x2]';
%  u2=[-5 -5]*[x3,x4]';
%  u3=[-5 -5]*[x5,x6]';
%  u4=[-5 -5]*[x7,x8]';
%  u5=[-5 -5]*[x9,x10]';
%  u6=[-5 -5]*[x11,x12]';
%  u7=[-5 -5]*[x13,x14]';
%  u8=[-5 -5]*[x15,x16]';
%  u9=[-5 -5]*[x17,x18]';
%  u10=[-5 -5]*[x19,x20]';

% good for identification
 u1=[-0.06 -0.06]*[x1,x2]';
 u2=[-0.06 -0.06]*[x3,x4]';
 u3=[-0.06 -0.06]*[x5,x6]';
 
 
% % good for identification  
%  u1=[-20 -20]*[x1,x2]';
%  u2=[-20 -20]*[x3,x4]';
%  u3=[-20 -20]*[x5,x6]';
%  u4=[-20 -20]*[x7,x8]';
%  u5=[-20 -20]*[x9,x10]';
%  u6=[-20 -20]*[x11,x12]';
%  u7=[-20 -20]*[x13,x14]';
%  u8=[-20 -20]*[x15,x16]';
%  u9=[-20 -20]*[x17,x18]';
%  u10=[-20 -20]*[x19,x20]';
 
% u1new=u1;
% u2new=u2;
% u3new=u3;
% u4new=u4;
% u5new=u5;
% u6new=u6;
% u7new=u7;
% u8new=u8;
% u9new=u9;
% u10new=u10;


%amplitude=0.6;

if tm<=Ns*T

  u1new=u1+amp*(exp(-0.2*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);
  u2new=u2+amp*(exp(-0.23*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);
  u3new=u3+amp*(exp(-0.25*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);

%   u1new=u1+amp*(exp(-1.3*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);
%   u2new=u2+amp*(exp(-1.33*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);
%   u3new=u3+amp*(exp(-1.35*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);

else
u1new=u1;
u2new=u2;
u3new=u3;
end

x1d=x2;
x2d=(g/l)*sin(x1)+(1/(m1*l^2))*u1new+((k1*aa^2)/(m1*l^2))*(sin(x3)*cos(x3)-sin(x1)*cos(x1))+((k3*aa^2)/(m1*l^2))*(sin(x5)*cos(x5)-sin(x1)*cos(x1));

x3d=x4;
x4d=(g/l)*sin(x3)+(1/(m2*l^2))*u2new+((k1*aa^2)/(m2*l^2))*(sin(x1)*cos(x1)-sin(x3)*cos(x3))+((k2*aa^2)/(m2*l^2))*(sin(x5)*cos(x5)-sin(x3)*cos(x3));

x5d=x6;
x6d=(g/l)*sin(x5)+(1/(m3*l^2))*u3new+((k2*aa^2)/(m3*l^2))*(sin(x3)*cos(x3)-sin(x5)*cos(x5))+((k3*aa^2)/(m3*l^2))*(sin(x1)*cos(x1)-sin(x5)*cos(x5));



x7d=x2;
x8d=sin(x1);
x9d=u1new;
x10d=sin(x1)*cos(x1);
x11d=x4;
x12d=sin(x3);
x13d=u2new;
x14d=sin(x3)*cos(x3);
x15d=x6;
x16d=sin(x5);
x17d=u3new;
x18d=sin(x5)*cos(x5);




%%

x19d=(gama*(abs(e1))^a*sign(e1)*x7+gama1*0.1*P(1));
x20d=(gama*(abs(e2))^a*sign(e2)*x8+gama1*P(2));
x21d=(gama*(abs(e2))^a*sign(e2)*x9+gama1*P(3));
x22d=(gama*(abs(e2))^a*sign(e2)*x10+gama1*P(4));
x23d=(gama*(abs(e2))^a*sign(e2)*x14+gama1*P(5));
x24d=(gama*(abs(e2))^a*sign(e2)*x18+gama1*P(6));


x25d=(gama*(abs(e3))^a*sign(e3)*x11+gama1*0.5*P(7));
x26d=(gama*(abs(e4))^a*sign(e4)*x12+gama1*P(8));
x27d=(gama*(abs(e4))^a*sign(e4)*x13+gama1*P(9));
x28d=(gama*(abs(e4))^a*sign(e4)*x14+gama1*P(10));
x29d=(gama*(abs(e4))^a*sign(e4)*x10+gama1*P(11));
x30d=(gama*(abs(e4))^a*sign(e4)*x18+gama1*P(12));


x31d=(gama*(abs(e5))^a*sign(e5)*x15+gama1*0.5*P(13));
x32d=(gama*(abs(e6))^a*sign(e6)*x16+gama1*P(14));
x33d=(gama*(abs(e6))^a*sign(e6)*x17+gama1*P(15));
x34d=(gama*(abs(e6))^a*sign(e6)*x18+gama1*P(16));
x35d=(gama*(abs(e6))^a*sign(e6)*x14+gama1*P(17));
x36d=(gama*(abs(e6))^a*sign(e6)*x10+gama1*P(18));


xdot=[x1d;x2d;x3d;x4d;x5d;x6d;x7d;x8d;x9d;x10d;x11d;x12d;x13d;x14d;x15d;x16d;x17d;x18d;x19d;x20d;x21d;x22d;x23d;x24d;x25d;x26d;x27d;x28d;x29d;x30d;...
    x31d;x32d;x33d;x34d;x35d;x36d];

function xdot=odefile_GD(t,x)
%global P;
%global P1

global x011
global x012

global x021
global x022

global x031
global x032

global kk
global T
%global P

global gama
%global gama1

global Ns
global a 

global amp

g=10;
m1=0.25; m2=0.25; m3=0.25; m4=0.25; m5=0.25; m6=0.25; m7=0.25; m8=0.25; m9=0.25; m10=0.25;
l=2;
aa=1;
k1=1.5;  
k2=2;  
%k3=0.5;
k3=1.5;


 
 %a=100;
tm=(kk-1)*T+t;



ns1=1; ns2=1; ns3=1; ns4=1; ns5=1;
ns6=1; ns7=1; ns8=1; ns9=1; ns10=1;

x1=x(1);
x2=x(2);
x3=x(3);
x4=x(4);
x5=x(5);
x6=x(6);

x7=x(7);
x8=x(8);
x9=x(9);
x10=x(10);
x11=x(11);
x12=x(12);
x13=x(13);
x14=x(14);
x15=x(15);
x16=x(16);
x17=x(17);
x18=x(18);

x19=x(19);
x20=x(20);
x21=x(21);
x22=x(22);
x23=x(23);
x24=x(24);
x25=x(25);
x26=x(26);
x27=x(27);
x28=x(28);
x29=x(29);
x30=x(30);
x31=x(31);
x32=x(32);
x33=x(33);
x34=x(34);
x35=x(35);
x36=x(36);


tmm=(kk-1)*T+t(size(t,1));

Y1=x19*x7+x011;
Y2=x20*x8+x21*x9+x22*x10+x23*x14+x24*x18+x012;

Y3=x25*x11+x021;
Y4=x26*x12+x27*x13+x28*x14+x29*x10+x30*x18+x022;

Y5=x31*x15+x031;
Y6=x32*x16+x33*x17+x34*x18+x35*x14+x36*x10+x032;


e1=x1-Y1;
e2=x2-Y2;

e3=x3-Y3;
e4=x4-Y4;

e5=x5-Y5;
e6=x6-Y6;



% good for identification
 u1=[-0.06 -0.06]*[x1,x2]';
 u2=[-0.06 -0.06]*[x3,x4]';
 u3=[-0.06 -0.06]*[x5,x6]';
 
 

if tm<=Ns*T

  u1new=u1+amp*(exp(-0.2*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);
  u2new=u2+amp*(exp(-0.23*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);
  u3new=u3+amp*(exp(-0.25*tm))*(0.4*sin(0.1*tm)^6*cos(1.5*tm)+0.3*sin(2.3*tm)^4*cos(0.7*tm)+0.4*sin(2.6*tm)^5+0.7*sin(3*tm)^2*cos(4*tm)+0.3*sin(0.3*tm)*cos(1.2*tm)^2+0.4*sin(1.12*tm)^3+0.5*cos(2.4*tm)*sin(8*tm)^2+0.3*sin(1*tm)*cos(0.8*tm)^2+0.3*sin(4*tm)^3+0.4*cos(2*tm)*sin(5*tm)^8+0.4*sin(3.5*tm)^5);

else
u1new=u1;
u2new=u2;
u3new=u3;
end

x1d=x2;
x2d=(g/l)*sin(x1)+(1/(m1*l^2))*u1new+((k1*aa^2)/(m1*l^2))*(sin(x3)*cos(x3)-sin(x1)*cos(x1))+((k3*aa^2)/(m1*l^2))*(sin(x5)*cos(x5)-sin(x1)*cos(x1));

x3d=x4;
x4d=(g/l)*sin(x3)+(1/(m2*l^2))*u2new+((k1*aa^2)/(m2*l^2))*(sin(x1)*cos(x1)-sin(x3)*cos(x3))+((k2*aa^2)/(m2*l^2))*(sin(x5)*cos(x5)-sin(x3)*cos(x3));

x5d=x6;
x6d=(g/l)*sin(x5)+(1/(m3*l^2))*u3new+((k2*aa^2)/(m3*l^2))*(sin(x3)*cos(x3)-sin(x5)*cos(x5))+((k3*aa^2)/(m3*l^2))*(sin(x1)*cos(x1)-sin(x5)*cos(x5));



x7d=x2;
x8d=sin(x1);
x9d=u1new;
x10d=sin(x1)*cos(x1);
x11d=x4;
x12d=sin(x3);
x13d=u2new;
x14d=sin(x3)*cos(x3);
x15d=x6;
x16d=sin(x5);
x17d=u3new;
x18d=sin(x5)*cos(x5);


%%

x19d=(gama*(abs(e1))^a*sign(e1)*x7);
x20d=(gama*(abs(e2))^a*sign(e2)*x8);
x21d=(gama*0.5*(abs(e2))^a*sign(e2)*x9);
x22d=(gama*(abs(e2))^a*sign(e2)*x10);
x23d=(gama*(abs(e2))^a*sign(e2)*x14);
x24d=(gama*(abs(e2))^a*sign(e2)*x18);


x25d=(gama*(abs(e3))^a*sign(e3)*x11);
x26d=(gama*(abs(e4))^a*sign(e4)*x12);
x27d=(gama*0.5*(abs(e4))^a*sign(e4)*x13);
x28d=(gama*(abs(e4))^a*sign(e4)*x14);
x29d=(gama*(abs(e4))^a*sign(e4)*x10);
x30d=(gama*(abs(e4))^a*sign(e4)*x18);


x31d=(gama*(abs(e5))^a*sign(e5)*x15);
x32d=(gama*(abs(e6))^a*sign(e6)*x16);
x33d=(gama*0.5*(abs(e6))^a*sign(e6)*x17);
x34d=(gama*(abs(e6))^a*sign(e6)*x18);
x35d=(gama*(abs(e6))^a*sign(e6)*x14);
x36d=(gama*(abs(e6))^a*sign(e6)*x10);


xdot=[x1d;x2d;x3d;x4d;x5d;x6d;x7d;x8d;x9d;x10d;x11d;x12d;x13d;x14d;x15d;x16d;x17d;x18d;x19d;x20d;x21d;x22d;x23d;x24d;x25d;x26d;x27d;x28d;x29d;x30d;...
    x31d;x32d;x33d;x34d;x35d;x36d];



