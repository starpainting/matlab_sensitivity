length('I like Matlab')
double('I like Matlab')
class('I like Matlab')
ischar('I like Matlab')
findstr('I like Matlab','a')
deblank('I like Matlab ')
str1='hello'
str2='I like Matlab'


syms('a','b','x','y')
[x,y]=solve(a*x-b*y-1,a*x+b*y-5)

sym z
syms x y real
z=x+i*y
a=conj(z)
b=simplify(z*a)

str2syms('[a,b;c,d]')

a=1+2i
c=[a,2]
b=c'


a=str2sym('[a11 a12;a21 a22]')
x=det(a)
y=a.'
z=eig(a)

factor
expand
collect
collect()

syms x y
s=(-7*x^2-8*y^2)*(-x^2+3*y^2)
a=expand(s)
b=collect(s,x)
c=factor(b)


syms x y z
s=3*x^2+5*y+2*x*y+3
r=(x^2+y^2)^2+(x^2-y^2)^2
a=expand(s)
[x,y,z]=solve(2*x+3*y-z-2,8*x+2*y+3*z-4,45*x+3*y+9*z-23)

%% 符号表达式赋值
syms x y
z=x+y
w=subs(z,[x,y],[1,2])
%% 化简
sym x
f=sin(x)^2+cos(x)^2
y=simplify(f)
%% 展开
expand
%% 通分
syms x y
f=x/y+y/x
[N,D]=numden(f)
%% 单变量函数的极限 limit(f,x,a) limit(f,x,a,'right')  limit(f,x,a,'left')
syms x a b
f=x*(1+a/x)^x*sin(b/x)
limit(f,x,inf)
%% 多变量函数的极限 limit(limit(f,x,x0),y,y0)
syms a x y
f=exp(-1/(x^2+y^2))*sin(x)^2/x^2*(1+1/y^2)^(x+a^2*y^2)
%g=limit(f,x,1/sqrt(y)) limit(g,y,inf)
limit(limit(f,y,inf),x,1/sqrt(y))
%%
syms a x y
f=exp(-1/(x^2+y^2))*sin(x)^2/x^2*(1+1/y^2)^(x+a^2*y^2)
g=limit(f,x,1/sqrt(y)) 
limit(g,y,inf)
%% int(f,x,a,b) 定积分
syms x
f=1/(3+2*x+x^2)
int(f,x,-inf,inf)
%% diff(f,x,n)  n是求导的次数
syms x
y=sin(x)/(x^2+4*x+3)
y1=diff(y,x,4)
y2=simplify(y1)
%% 求偏导 diff(diff(z,x),y)
syms x y
z=(x^2-2*x)*exp(-x^2-y^2-x*y)
z1=diff(z,x)
z2=diff(diff(z,x),y)
%%
syms x y z
w=sin(x^2*y)*exp(-x^2*y-z^2)
diff(diff(diff(w,x,2),y),z)
%% Taylor级数展开 taylor(f,x,a,'order',k)按x=a进行泰勒幂级数展开k项 ；taylor(f,x) 按x=0进行泰勒幂级数展开6项 
syms x
f=sin(x)/(x^2+4*x+3)
y=taylor(f,x,2,'order',9)
%% 解方程solve(f,v)
syms x
f=x^3-3*x+1
s=solve(f,x)
%% 解方程组
syms x y z f1 f2 f3
f1=x+2*y-x-27
f2=x+z-3
f3=x^2+3*y^2-28
[x,y,z]=solve(f1,f2,f3)
%%
syms x
f=(exp(x^3)-1)/(1-cos(sqrt(x-sin(x))))
limit(f,x,0,'right')
%%
syms x y z
f=sin(x*y+z)
s=int(f,z)
%%
syms x t
f=(-2*x^2+1)/(2*x^2-3*x+1)^2
y=int(f,x,cos(t),exp(-2*t))
%%
syms x
y=x^3+3*x-2
a=solve(diff(y,x)-4,x)
display(subs(y,x,a))
%%
syms t x a
f=a*log(x)-t^3*t*cos(x)
y1=diff(f,x)
y2=diff(f,t,2)
y3=diff(diff(f,t),x)
%%
syms a x b
f=a*x*sin(x)-b*x
y=int(f,x)
%%
syms u w v z
f=u*w^2+z*w-v
s=solve(f,z)
%%
syms x
assume(real(x)>=0)
assumeAlso(iamg(x)>=0)
f=x^3+475/100*x+5/2
s=solve(f,x)
%% 解微分方程dsolve('equ1','equ2',……,‘特解1,特解2……’，'x') 默认时自变量为t，方程式中用D2y表示dy/dx的二阶导
dsolve('Dy=1+y^2','y(0)=1','x')
%%
a=dsolve('D2y+4*Dy+29*y=0','x')
b=dsolve('D2y+4*Dy+29*y=0','y(0)=0,Dy(0)=15','x')
%%
[x,y,z]=dsolve('Dx=2*x-3*y+3*z','Dy=4*x-5*y+3*z','Dz=4*x-4*y+2*z')
%%
[x,y,z]=dsolve('Dx=2*x-3*y+3*z','Dy=4*x-5*y+3*z','Dz=4*x-4*y+2*z','x(0)=0,y(0)=1,z(0)=2')
%% 曲线图 plot(x,y,'s') s表示线型，默认是蓝色实线 虚线是':' 画圈并连线是'o-' plot（x,y1,s1,y2,s2,……） 
x=linspace(0,2*pi,30)
y=sin(x)
z=cos(x)
plot(x,y,'m',x,z,'go:')
%% 符号函数画图 ezplot('f(x)',[a,b]) 函数[a,b]区间上的图像； 
ezplot('cos(x)',[0,pi])
%% ezplot('x(t)','y(t)',[a,b]) x=x(t),y=y(t)的图像
ezplot('cos(t)^3','sin(t)^3',[0,2*pi])

%% ezplot('f(x,y)',[x1,x2,y1,y2]) 隐函数f(x,y)=0在[x1,x2]、[y1,y2]上的图像
ezplot('exp(x)+sin(x*y)',[-2,0.5,0,2])

%% fplot('fun',lims)
fplot('t=exp(x)+cos(3*x^2)',[-1,2])

%%
flot('tanh',[-10,10])

%% 对数坐标轴
x=logspace(-1,2);
loglog(x,exp(x),'s-')
grid on %标注栅格

%%
x=1:10
semilogy(x,10.^x)

%% plot3(x,y,z,'s') 空间画图
t=0:pi/50:10*pi;
x=sin(t);
y=cos(t);
z=t;
plot3(x,y,z)
rotate3d %可旋转
%% plot3(X,Y,Z)
x=-3:0.1:3;
y=1:0.1:5;
[X,Y]=meshgrid(x,y) %生成y*x的矩阵
Z=(X+Y).^2
plot3(X,Y,Z)
surf(Z)
%% 解微分方程通解
[x,y,z]=dsolve('Dx=2*x-3*y+3*z','Dy=4*x-5*y+3*z','Dz=4*x-4*y+2*z')
%% 解微分方程特解
dsolve('D2y=1+Dy','y(0)=1,Dy(0)=0')
%%
fplot(@(x)exp(2*x)+sin(3*x.^2),[-1,2])
%%
ezplot('[tanh(t),cos(t),sin(t)]',[-2*pi,2*pi])
%%
t=0:pi/50:4*pi;
plot3(sin(t),cos(t),t)
%% 单窗口多曲线绘图
t=0:pi/50:2*pi;
y=sin(t);
y1=sin(t+0.25);
y2=sin(t+0.5);
plot(t,[y',y1',y2'])
%% 单窗口多曲线分图绘图 subplot(m,n,p) 分成m行n列的图 p是要绘制的图的序号
y=sin(t);
y1=sin(t+0.25);
y2=sin(t+0.5);
c
%% 多窗口绘图 figure(n) 创建窗口函数，n为窗口顺序号
t=0:pi/100:2*pi;
y=sin(t);
y1=sin(t+0.25);
y2=sin(t+0.5);
plot(t,y)
figure(2)
plot(t,y1)
figure(3)
plot(t,y2)
%%
t=0:0.1:10;
y=sin(t);
z=cos(t);
plot(t,y,'r',t,z,'b--')
x=[1.7*pi;1.6:pi];
y=[-0.3;0.8];
s=['sin(t)';'cos(t)'];
text(x,y,s);
legend('正弦','余弦')
xlabel('时间t');
ylabel('正弦、余弦');
grid
axis squre
%% 基本二维绘图函数 fill
x=[1 2 3 4 5];
y=[4 1 5 1 4];
fill(x,y,'r')
%% 阶梯图
x=0:0.1:pi*2;
y=sin(x);
stairs(x,y)
%% 极坐标
x=0:pi/90:pi*2;
y=cos(x);
polar(x,y) %polarplot
%% 火柴杆
t=0:0.2:2*pi;
y=cos(t);
stem(y)
%% 彗星曲线图
t=-pi:pi/500:pi;
y=tan(sin(t))-sin(tan(t));
comet(t,y)
%%
x=magic(6);
area(x)
%% 饼图
x=[1 2 3 4 5 6 7];y=[0 1 0 0 0 0 0];
pie(x,y)
%%
x=0:pi/100:2*pi;
y=2*exp(-0.5*x).*sin(2*pi*x);
plot(x,y)
%%
x=0:pi/100:2*pi;
y1=0.2*exp(-0.5*x).*cos(4*pi*x);
y2=2*exp(-0.5*x).*cos(pi*x);
plotyy(x,y1,x,y2)
%%
x=0:pi/100:2*pi;
y1=0.2*exp(-0.5*x).*cos(4*pi*x);
y2=2*exp(-0.5*x).*cos(pi*x);
subplot(2,1,1)
plot(x,y1)
subplot(2,1,2)
plot(x,y2)
%%
t=0:0.01:10;
y=2*t-5*t.^2;
plot(t,y)
%%
y=dsolve('D2y+2*Dy+2*y=0','y(0)=1,Dy(0)=0','x');
ezplot(y);
%%
x=0:pi/90:pi*2;
y=sin(2*x).*cos(2*x);
polarplot(x,y);
%%
x=0:pi/90:pi*2;
y=sin(x);
plot(x,y)
xlabel('自变量X');
ylabel('函数Y');
title('示意图');
grid;
%%
x=[5 10 15 20 10];y=[0 0 0 0 0];
pie(x,y)
%% 单位矩阵
A=eye(3)
%% 随机矩阵
B=rand(4,3)
%% 全零矩阵
zeros(3)
%% 全一矩阵
ones(3)
%% 1.提取对角线数值 2.生成对角矩阵 【连用两次diag()】
A=rand(3,5)
B=diag(A)
C=diag(B)
%% 访问同一个矩阵的元素 A([a,b],[m,n])前一个中括号中代表行，后一个中括号代表列，新数组是所有行列交叉处的元素
A=rand(4)
B=A([1,2],[3,4]) %第一行、第二行和第三列、第四列的交点元素
C=A(1:2,3:4)     %这个与上面一个数组相同，只不过用冒号表示
%%
A=rand(6)
B=A([1,3,5],[2,4,6])   %一行一行挑
C=A(1:2:5,2:2:6)       %按步长挑  
%% 数组排序 sort(A,数组维数，升降序)    默认 维数1，升序 
%% 数组查找函数 find(A) 返回满足条件的单下标索引（从左到右一列一列从1开始标号）
A=[16 2 3 13;5 11 10 8;9 7 6 12;4 14 15 1]
find(A>8)
%% 索引扩展数列 cat horzcat vercat
A=rand(3)
cat(2,A)
%%
syms x y z w; 
A=[4 3 2 1;3 4 3 2;2 3 4 3;1 2 3 4];
B=[x;y;z;w];
f=A*B-[1;2;3;4]
[x,y,z,w]=solve(f,x,y,z,w)
%%
A=[4 3 2 1;3 4 3 2;2 3 4 3;1 2 3 4];
B=[1;2;3;4]
x=A\B
%%
a=[1,2,3;4,5,6;7,8,9];
b=a.^2  %数组的平方
c=a^2   %矩阵的平方
%% 魔方矩阵 magic(n) 每行每列两对角线上元素之和相等的n阶矩阵(从1开始到n^2个元素)
A=magic(5)+100
%% 范德蒙矩阵 vander(V) 最后一列全是1，倒数第二列是V向量，其他各列是其后一列与V向量对应元素的乘积，列数等于V的行数
vander([1;2;3;4;6])
%% 希尔伯特矩阵 hilb(n)  &  n阶希尔伯特矩阵的逆矩阵 invhilb(n)   
A=hilb(4)
B=invhilb(4)
%% 托普利兹矩阵 除了第一行第一列外，其他每个元素都与左上角元素相同
toplize(n)
%% 伴随矩阵 compan(p) p是多项式系数，从左到右，高次到低次
compan([4,0,1,6])
%% 帕斯卡矩阵 pascal(n) 杨辉三角的系数 n是多项式阶次+1
pascal(6)

%%
A=rand(5).*30+20
%% randn(n)正态分布的随机矩阵，方差为1，均值为0
B=0.6+sqrt(0.1)*randn(5)
%% fix()取整 & rem(A,x) 求A矩阵中每个元素除以x的余数
x=fix((90-10+1)*rand(5)+10)
p=rem(x,3)
%% 
A=randn(5)
B=A*diag(1:5)
%%
p=[3,-7,0,5,2,-18]; %A的伴随矩阵
x1=eig(A) 
x2=roots(p)     
%% 标量循环运算法
t=0:0.1:10;
N=length(t);
for k=1:N
    y1(k)=1-exp(-0.5*t)*cos(2*t);
end
%%
x=-0.5:0.01:0.5;
v=1:150;
A=sin(0.25*v*pi)./(0.25*v*pi).*cos(2*pi*v*x);

%% trapz(x,y)梯形法数值积分 x为积分区间矩阵，y为积分函数
x=-pi:pi/100:pi;
y=sin(x);
trapz(x,y)
%%
x=0:0.01:2;
y=1./(x.^3-2*x-5);
trapz(x,y)

%% quad(@（x）function,a,b)辛普森数值积分  x是被积量,如果function已定义则不用写（x）；a,b是积分区间  
%% 方法1 新建脚本定义函数
Q=quad(@myfun,0,2)
%% 方法2 使用@建立联系
y=@(x)1./(x.^3-2*x-5);
quad(y,0,2)
%%
quad(@(x)1./(x.^3-2*x-5),0,2)


%% quadl()科茨数值积分
%% dblquad(function,xmin,xmax,ymin,ymax) 二重积分
z=@(x,y)y*sin(x)+x*cos(y)
z=dblquad(z,pi,2*pi,0,pi)

%% triplequad(fun,xmin,xmax,ymin,ymax,zmin,zmax) 三重积分
triplequad(@(x,y,z)y*sin(x)+z*cos(x),0,pi,0,1,-1,1)

%% 函数的数值微分 diff(x,n) 
A=[1,3,5,7,4,9];
diff(A)
%% 梯度gradient 一维 dx=gradient(F,步长) ；二维[px,py]=gradient(F,x步长,y步长) 先编织网格(meshgird)，再求梯度
v=-2:0.1:2;
[x,y]=meshgrid(v); %编织网格
z=x.*exp(-x.^2-y.^2);
[px,py]=gradient(z,0.2,0.2);
contour(v,v,z)    %绘制等高图 （等高图其实也是三维作图）
hold on
quiver(v,v,px,py)  %绘制矢量箭头
%% surf(x,y,z) mesh(x,y,z) meshz(x,y,z) 三维作图

%% 找函数最小值 fminbnd(fun,a,b)
 y=@(x)x.^3-2*x-5;
 fminbnd(y,0,2)
%% 多元函数最小值 [x,fval]=fminsearch(fun,x0) x是最小值点，fval是精度, x0是定义的每个元的初始迭代点
 f=@(x)100*(x(2)-3*x(1)^2)^2+(1-2*x(1))^2
 [x,fval]=fminsearch(f,[4,5])

%% 
y=@(x)-sin(x);
fminbnd(y,1,3)

%%
w=triplequad(@(x,y,z)abs(sqrt(x^2+y^2+z^2-1)),0,1,0,1,0,1)

%% 一阶常微分方程   inline('函数表达式',‘变量名1’,'变量名2')     
fun=inline('-2*y+2*x^2+2*x','x','y');  %x,y可以不写
[x,y]=ode23(fun,[0,10],1)              %求解并赋给x，y ,[0,10]是x的数值区间，1是初值y(0)=1
plot(x,y)
%%
[t,y]=ode45(@rigid,[0,12],[0 1 1])
plot(t,y)
%% 多项式及其操作  
a=[1 -2 5 6];
poly2sym(a)   %poly2sym 以从高到低的幂次数前的系数构建多项式
%% 以根得到相应的多项式的系数 poly
a=[-5 3+2i 3-2i];
b=poly(a)     %poly 得到以a中的元素为根的多项式的系数
c=poly2sym(b)   %构造多项式
x=root(c)
%% 多项式的求导 polyder
a=[3 5 7 4 2 1 3]; 
b=polyder(a) %b为导函数的系数向量，a为原函数的系数向量
%% 多项式的求值 polyval(p,s) p是多项式系数，自变量的取值为s矩阵中的每一个值
polyval([5 3 0 5],7)
%% 多项式拟合 polyfit(x,y，n) 用二乘法对已知数据进行x，y拟合，结果是n阶多项式系数向量
x=0:pi/50:pi/2;
y=sin(x);
a=polyfit(x,y,5)
m=polyval(a,x);
plot(x,m,'bo',x,y,'g--')
%% 
x=0:pi/50:2*pi;
y=2*sin(x)+cos(x).^2;
a=polyfit(x,y,5)
m=polyval(a,x);
plot(x,m,'bo',x,y,'g--')
%%
x=[0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7 3];
y=[2 2.378 3.944 7.346 13.232 22.25 35.048 52.274 74.576 102.602 137];
a=polyfit(x,y,1);
m=polyval(a,x);
b=polyfit(x,y,3);
n=polyval(b,x);
plot(x,m,'bo-',x,n,'g--')
%% 函数插值    一维插值interp1     二维插值：interp2 largange插值 newton插值
t=[0 3 6 9 12 15 18 21];
T=[18 19 19.5 23 27 25.6 24 20];
ti=1.5:1.5:21;
Ti=interp1(t,T,ti,'linear');   %直线插值法 生成ti时间点时的温度值
plot(ti,Ti)
%% 四种一维插值法比较
x=-pi:pi;
y=sin(x)+cos(x);
xi=-pi:0.1:pi;
yi_linear=interp1(x,y,xi,'linear');
yi_nearest=interp1(x,y,xi,'nearest');
yi_cubic=interp1(x,y,xi,'cubic');
yi_spline=interp1(x,y,xi,'spline');
plot(xi,yi_linear,'b-',xi,yi_nearest,'g*-',xi,yi_cubic,'r-.',xi,yi_spline,'yo-')

%% 网格meshgrid
x=-3:0.5:3;
y=-3:0.5:3;
[X,Y]=meshgrid(x,y);

%% 二维插值
x=-3:0.5:3;
y=-3:0.5:3;
[X,Y]=meshgrid(x,y);
z=peaks(X,Y);
xi=-3:0.1:3;
yi=-3:0.1:3;
[Xi,Yi]=meshgrid(xi,yi);
zi_linear=interp2(X,Y,z,Xi,Yi,'linear');
zi_nearest=interp2(X,Y,z,Xi,Yi,'nearest');
zi_cubic=interp2(X,Y,z,Xi,Yi,'cubic');
zi_spline=interp2(X,Y,z,Xi,Yi,'spline');
%plot3(Xi,Yi,zi_linear,'b',Xi,Yi,zi_nearest,'g',Xi,Yi,zi_cubic,'r',Xi,Yi,zi_spline,'y')
mesh(Xi,Yi,zi_linear)
%% 三维作图 mesh(x,y,z)   若z由x,y得到，因为要维度相同，需要要用meshgrid（网格）
%% 三维作图 surf(x,y,z)   要用meshgrid（网格）
z=[82 81 80 82 84;79 63 61 65 81;84 84 82 85 86];
x=1:5;
y=1:3;
[X,Y]=meshgrid(x,y);
mesh(x,y,z)
figure(2)
surf(X,Y,z)
figure(3)
plot3(x,y,z)
rotate3d

%% griddata(x,y,z,Xi,Yi,'v4')   x,y不用网格，xi，yi需要
x=[129 140 103.5 88 185.5 195 105 157.5 107.5 77 81 162 162 117.5];
y=[7.5 141.5 23 147 22.5 137.5 85.5 -6.5 -81 3 56.5 -66.5 84 -33.5];
z=[4 8 6 8 6 8 8 9 9 8 8 9 4 9];
xi=75:0.5:200;
yi=-50:0.5:150;
[Xi,Yi]=meshgrid(xi,yi);
zi=griddata(x,y,z,Xi,Yi,'v4');

%% 作三维图，画法向量
x=linspace(-1.5,1.5,75);
y=linspace(-1,1,50);
[X,Y]=meshgrid(x,y);
z=X.*exp(-X-Y);
surf(X,Y,z);
hold on
[U,V,W]=surfnorm(X,Y,z); %求法向量
quiver3(X,Y,z,U,V,W)     %画箭头
%%
x0=-8:8;
y0=x';
x=ones(size(y0))*x0;
y=y0*ones(size(x0));
%[x,y]=meshgrid(-8:8);
z=sin(sqrt(x^2+y^2))/sqrt(x^2+y^2);

%%
n=0:30;
y=1./abs(n-6);
plot(n,y,'*','markersize',20)
%%
x=0:pi/100:2*pi;
y=sin(x);
plot(x,y)
text(3*pi/4,sin(3*pi/4),'\fontsize{16}\leftarrowsin(x)=0.707');
%%
t=0:pi/100:pi;
y=sin(t).*sin(9*t);
plot(t,y,'o','markeredgecolor',[0,1,0],'markerfacecolor',[1,0.8,0])
hold on
plot(t,sin(t),'r')
plot(t,-sin(t),'r')
%% 直方图bar
x=-2.9:0.1:2.9;
y=exp(-x.^2);
bar(x,y)
%% 累计式直方图'stack' 
x=1990:5:2000;
a=[90.7 70.6 73.9;281.6 271 214.6;254.8 323.7 326.5];
bar(x,a,'stack')
legend('第一产业','第二产业','第三产业')
%% 分组式直方图'group'
x=1990:5:2000;
a=[90.7 70.6 73.9;281.6 271 214.6;254.8 323.7 326.5];
bar(x,a,'group')
legend('第一产业','第二产业','第三产业')
%% 横排直方图barh
x=1990:5:2000;
a=[90.7 70.6 73.9;281.6 271 214.6;254.8 323.7 326.5];
barh(x,a,'group')
legend('第一产业','第二产业','第三产业')
%% 三维饼图pie3
a=[1 1.6 1.2 0.8 2.1];
subplot(1,2,1)
pie(a)
subplot(1,2,2)
pie3(a)
%% 离散杆图stem
t=-2*pi:pi/20:2*pi;
h=stem(t,cos(t));
%% 画椭圆
th=[0:pi/50:2*pi]';
a=[0.5:0.5:4.5];
x=cos(th)*a;
y=sin(th)*sqrt(25-a.^2);
plot(x,y)
%% 画积分函数图像 cumtrapz(y)*dx
x=0:0.1:4;
y=x.*sin(x);
s=cumtrapz(y)*0.1;
plot(x,s)
%% 获取二维图形数据 [x,y]=ginput(n) 用鼠标从二维图像上获取n个点的坐标(x,y)
x=0:0.01:10;
y=(x+2).*cos(x)+5*sin(x);
plot(x,y)
grid on
[x,y]=ginput(1);
%%
t=0:pi/100:2*pi;
x=sin(t);
y=cos(t);
z=cos(2*t);
plot3(x,y,z)
%%
[x,y]=meshgrid(-4:0.1:4);
z=x.^2+x.^2;
surf(x,y,z)
%% 彩色转黑白
black=rgb2gray(img1);
imshow(black)

%% 条件语句
x=input('x=');
if x>0
    y=x*sin(x);
else
    y=x^2+exp(x);
end
disp('y=')
disp(y)

%% log表示loge，其它加上底数，如log10，log2
x=input('x=');
if x<=0
    y=(x+sqrt(pi))/exp(2)
else
    y=log(x+sqrt(1+x^2))/2
end
%% 
A=input('输入三个三角形的边');  %%输入矩阵，如[1 1 1]
if A(1)+A(2)>A(3)&A(2)+A(3)>A(1)&A(3)+A(1)>A(2)
    p=(A(1)+A(2)+A(3))/2;
    S=sqrt(p*(p-A(1))*(p-A(2))*(p-A(3)))
end

%%
x=input('');
if x<0
    y=-1
elseif x==0
    y=0
else
    y=1
end

%% switch fix()
x=input('x=');
switch fix(x)
    case x<200
        y=x
    case x>200&&x<500
        y=x*0.97
end

%%
A=input('');
B=input('');
try
    D=A*B
catch
    D=A.*B
end
lasterr
    
%%
x=zeros(6);
y=ones(6);
z=diag(diag(y));
a=x+z;
for m=1:5
    a(m,m+1)=2;
end
for m=2:6
    a(m,m-1)=2;
end
disp(a)

%% for循环若赋值为二维数组
data=[1 3 4;5 2 10;5 5 5];
for n=data
    m=n(1)-n(2)
end

%% for循环实质是依次带入数组的每一列进行运算
s=0;
a=[12 13 14;14 15 16;18 19 20;21 22 23];
for k=a
    s=s+k;
end
disp(s')

%% while循环
m=1;s=0;
while m<101
    s=s+m;
    m=m+1;
end
disp(s)

%% break continue
for n=100:200
    
    if rem(n,21)~=0
        continue
    else
        disp(n);
        break
    end
end

%%
for m=1:500
    s=0;
    for n=1:m/2
        if rem(m,n)==0
            s=s+n;
        end
    end
    if s==m
        disp(m);
    end
end

%% 非线性最小二乘拟合 lsqcurvefit
x1=0.1:0.1:1;
y1=[2.3201 2.647 2.9707 3.2885 3.6008 3.909 4.2147 4.5191 4.8232 5.1275];
[A,res]=lsqcurvefit('f1',[1;2;2;2],x1,y1) %A是定义的函数中的各个系数组成的数组，res是残差,[1;2;2;2]是迭代初值

%%
t1=100:100:1000;
y1=[4.54 4.99 5.35 5.65 5.90 6.1 6.26 6.39 6.50 6.59]./1000;
[a,res]=lsqcurvefit('f2',[0.2;0.05;0.05],t1,y1)

%%
syms x y
z=(x-y)/(x+y);
a1=diff(diff(z,x),x)
a2=diff(diff(z,y),x)
a3=diff(diff(z,y),y)

%%
syms x y z
f=x^2+y^2+z^2;
int(int(int(f,z,sqrt(x*y),x^2*y),y,sqrt(x),x^2),x,1,2)

%%
s1=0:pi/100:pi;
t1=0:pi/100:2*pi;
[s,t]=meshgrid(s1,t1);
x=cos(s).*cos(t);
y=cos(s).*sin(t);
z=sin(s);
mesh(x,y,z)









