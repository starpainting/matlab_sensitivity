%%
clear
clc
syms x;
syms k 
y=2/(2*k+1)*((x-1)/(x+1))^(2*k+1);
int(y,k,0,inf)

%%
t=0:pi/100:pi;
y=sin(t).*sin(9*t);
plot(t,y,'o-','markeredgecolor',[0,1,0],'markerfacecolor',[1,0.8,0])
hold on
plot(t,sin(t),'--r')
plot(t,-sin(t),'--r')

%%
clear
clc
syms x y a;
f=exp(-(x+y)/(x^2+y^2))*(sin(x)^2)/(x^2)*(1+1/y)^(x+a^2*y^2);
L=limit(limit(f,x,1/y^2),y,inf)
%%
clear
clc
syms x y z;
f=x^2+y^2+z^2;
int(int(int(f,z,sqrt(x*y),x^2*y),y,sqrt(x),x^2),x,1,2)

%%
clear
clc
 f=@(A)50*(A(2)-A(1)^2)^2+(1-A(1))^2;
 [A,fval]=fminsearch(f,[-2;3])
 
%%
clear
clc
[t,y] = ode45(@rigid,[0 5],[1 0 1]); 
plot(t,y)
 
%%
clear
clc
x=[1 2 3 5 6 8 10 11];
y=[0 4.2 5.6 6.4 7.2 8.8 7.0 5.3];
xi=1:0.5:11;
yi=interp1(x,y,xi,'spline');
plot(xi,yi) 
 
%%
x1=-2:0.1:2;
y1=-2:0.1:2;
[x,y]=meshgrid(x1,y1);
z=x.*exp(-x.^2-y.^2);
subplot(4,1,1)
surf(x,y,z)
 
%%
a=0;b
for m=1:10
    for n=1:m
        n=n*n;
    end
    a=a+n;
end
 
 
 
 














