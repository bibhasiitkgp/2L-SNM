% function A=signed_MP1(~) 
% for slashdot
clc;
clear;
   k_0=300;
xy1=0;
xy2=0;
xy3=0;
xy4=0;
xy5=0;
xy6=0;

abc1=0;
abc2=0;
abc3=0;
abc4=0;
abc5=0;
abc6=0;
  
  
Cs=0;
SG=0;
U=0;
%   newmanB=0;
%  
%   Ue=0;
%   lambdan=0;
  
fileID = fopen('slashdot.txt','r');
sizeA = [3 Inf];
tempElist=fscanf(fileID,'%f', sizeA);
fclose(fileID);

m=length(tempElist);
tempElist=tempElist';
Elist=zeros(2*m,3);
Elist(1:m,:)=tempElist;
Elist((m+1):2*m,1)=tempElist(:,2);
Elist((m+1):2*m,2)=tempElist(:,1);
Elist((m+1):2*m,3)=sign(tempElist(:,3));
% Elist((m+1):2*m,4)=tempElist(:,4);

list=unique(Elist(:,1));

    tempx=find(Elist(:,1)>54318);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>54318);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    tempx=find(Elist(:,1)>59270);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>59270);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    tempx=find(Elist(:,1)>71026);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>71026);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    tempx=find(Elist(:,1)>75314);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>75314);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    

n=length(unique(Elist(:,1)));
n1=max(Elist(:,1));
tri1=zeros(1,n1);
tri2=zeros(1,n1);
tri3=zeros(1,n1);
tri4=zeros(1,n1);
n=n1;
parfor i=1:n1
    
    f1=zeros(n1,1);
%     if(mod(i,10000)==0)
%         i
%     end
    temp=find(Elist(:,1)==i);
    if(~isempty(temp))
    x1=Elist(temp,2);
    f1(x1,1)=Elist(temp,3);
    for j=1:length(x1)
        f2=zeros(n1,1);

        if(f1(x1(j))==1)
            temp=find(Elist(:,1)==x1(j));
            if(~isempty(temp))
                x2=Elist(temp,2);
                f2(x2,1)=Elist(temp,3);
                tempx=f1.*f2;
                    tempy=length(find(tempx==-1));
                    tri2(i)=tri2(i)+tempy; % ++- triangles
                    tempy=(find(tempx==1));
                    tempz=f1(tempy);
                    tempz1=length(find(tempz==1));
                    tri1(i)=tri1(i)+tempz1; % +++
                    tempz1=length(find(tempz==-1));
                    tri3(i)=tri3(i)+tempz1; % +--
                
%                 tri(1,i)=tri(1,i)+sum(f1.*f2);
        %         for k=1:length(x2)
        %         temp=find(Elist(:,1)==k);
        %         x3=Elist(temp,2);
        %         tri(1,i)=tri(1,i)+length(find(x3==i));

        %         end
            end
        else
            temp=find(Elist(:,1)==x1(j));
            if(~isempty(temp))
                x2=Elist(temp,2);
                f2(x2,1)=Elist(temp,3);
                tempx=f1.*f2;
                    tempy=length(find(tempx==-1));
                    tri3(i)=tri3(i)+tempy; % -+- triangles
                    tempy=(find(tempx==1));
                    tempz=f1(tempy);
                    tempz1=length(find(tempz==1));
                    tri2(i)=tri2(i)+tempz1; % -++
                    tempz1=length(find(tempz==-1));
                    tri4(i)=tri4(i)+tempz1; % ---
%                 tri(1,i)=tri(1,i)+sum(f1.*f2);
        %         for k=1:length(x2)
        %         temp=find(Elist(:,1)==k);
        %         x3=Elist(temp,2);
        %         tri(1,i)=tri(1,i)+length(find(x3==i));

        %         end
            end
            
        end
        
    end
    end

end
t3=sum(tri1);
t2=sum(tri2);
t4=sum(tri3);
t0=sum(tri4);





%%


tempElist=Elist';  
% 
% fileID = fopen('epinion.txt','r');
% sizeA = [4 Inf];
% tempElist=fscanf(fileID,'%f', sizeA);
% fclose(fileID);

m=length(tempElist);
tempElist=tempElist';
Elist=zeros(2*m,3);
Elist(1:m,:)=tempElist;
Elist((m+1):2*m,1)=tempElist(:,2);
Elist((m+1):2*m,2)=tempElist(:,1);
Elist((m+1):2*m,3)=sign(tempElist(:,3));
% Elist((m+1):2*m,4)=tempElist(:,4);


  
n=length(unique(Elist(:,1)));
n1=max(Elist(:,1));
n=n1;
   xx1=0;
   xx2=0;
   xx3=0;
   xx4=0;
  
   
 cumdegree5=zeros(1,n);
  
  
  
for abc=1:1
    
fileID = fopen('slashdot.txt','r');
sizeA = [3 Inf];
tempElist=fscanf(fileID,'%f', sizeA);
fclose(fileID);

m=length(tempElist);
tempElist=tempElist';
Elist=zeros(2*m,3);
Elist(1:m,:)=tempElist;
Elist((m+1):2*m,1)=tempElist(:,2);
Elist((m+1):2*m,2)=tempElist(:,1);
Elist((m+1):2*m,3)=sign(tempElist(:,3));
% Elist((m+1):2*m,4)=tempElist(:,4);

list=unique(Elist(:,1));

    tempx=find(Elist(:,1)>54318);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>54318);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    tempx=find(Elist(:,1)>59270);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>59270);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    tempx=find(Elist(:,1)>71026);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>71026);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    tempx=find(Elist(:,1)>75314);
    Elist(tempx,1)=Elist(tempx,1)-1;
    tempx=find(Elist(:,2)>75314);
    Elist(tempx,2)=Elist(tempx,2)-1;
    
    

n=length(unique(Elist(:,1)));
n1=max(Elist(:,1));
  n=n1;
D=tabulate(Elist(:,2));
% D=tabulate(list);
x=zeros(n,1);
if(D(1,1)==0)
x(D(2:length(D),1))=(D(2:length(D),2));
else
x(D(:,1))=(D(:,2));
end

F=tabulate(x);
F(:,2)=F(:,2);
FF=zeros(1,max(F(:,1)));
if(F(1,1)==0)
FF(F(2:length(F),1))=F(2:length(F),2);
else
FF(F(1:length(F),1))=F(1:length(F),2);
end
for i=2:length(FF)
    FF(length(FF)-i+1)=FF(length(FF)-i+2)+FF(length(FF)-i+1);
end
cumdegree=FF/FF(1);
  Degree=x;

%% gamma estimation

        index=1:length(cumdegree);
        X=log(index(k_0:max(Degree)));
        Y=log(cumdegree(k_0:max(Degree)));
        new_b = 0;
        new_m =0;
        b_current = 0;
        m_current=0;
        learningRate=0.01;


for t=1:100000
Error=0;

for i=1:length(X)
    
    Error=Error+(Y(i)-(new_m*X(i)+new_b))^2;
end

N = length(X);
Error=Error/N;
b_gradient = 0;
m_gradient = 0;
    for i=1:length(X)
        b_gradient =b_gradient+ -(2/N) * (Y(i) - ((m_current*X(i)) + b_current));
        m_gradient =m_gradient + -(2/N) * X(i) * (Y(i) - ((m_current *X(i)) + b_current));
    end
    new_b = b_current - (learningRate * b_gradient);
    new_m = m_current - (learningRate * m_gradient);

    b_current=new_b;
    m_current=new_m;
end

% (figure);
% plot(X,Y,'o');
% hold on;
% plot(X,new_m*X+new_b)
% 
% (figure);
% loglog(cumdegree,'k o');
% hold on;
gamma=-new_m;
%%
%   load('soc-sign-bitcoin.mat');
%   
%    x=sum(abs(A));
%    [gamma, xmin, L]=plfit(x(1:length(x)));
%   plplot(x, xmin, gamma,1);
% %    hold on;
%    pause(1)
% %    n=length(A);
%    epsilon=0;
%    for i=1:n
%        epsilon=epsilon+sum(x(1:i))/i;
%    end
   
   
   
   
   
   
   
   
a=0;
        for i=1:n
            a=a+x(i)*(x(i)-1)/2;
        end
        
%         
% t0=409864;
% t4=2979238;
% t2=3471334;
% t3=33504242;
        
Cs_data=((t3+t4)-(t0+t2))/a;
        SG_data=((t3+t4)-(t0+t2))/((t3+t4)+(t0+t2));
        
        U_data=(1-SG_data)/(1+SG_data);




if(abc==1)
T0data=t0/(t0+t2+t3+t4)
T1data=t4/(t0+t2+t3+t4)
T2data=t2/(t0+t2+t3+t4)
T3data=t3/(t0+t2+t3+t4)
end
temp=find(Elist(:,3)==-1);
Nlist=Elist(temp,1);
D=tabulate(Nlist);
% D=tabulate(list);
NegDegree=zeros(n,1);
NegDegree(D(:,1))=(D(:,2));
temp=find(Elist(:,3)==1);
Plist=Elist(temp,1);
D=tabulate(Plist);
% D=tabulate(list);
PosDegree=zeros(n,1);
PosDegree(D(:,1))=(D(:,2));


%    n=length(A);
%    B=A;
%    PosDegree=zeros(n,1);
%    NegDegree=zeros(n,1);
%    for i=1:n
%      PosDegree(i)=length(find(A(:,i)==1));
%      NegDegree(i)=length(find(A(:,i)==-1));
% %    end
% D=x;
% Degree=x;
% D_DistN=zeros(1,max(D));
% 
%     for j=1:n
%         for k=1:n	
%             if(D(1,k)==j)
%                 D_DistN(1,j)=D_DistN(1,j)+1;
%             end
%         end
%     end
%     D_DistN=D_DistN/sum(D_DistN);
% 
%         cumdegree=zeros((max(Degree)),1);
% % cumdegree=zeros(1,n);
%         for i=1:max(Degree)
%             cumdegree(i)=sum(D_DistN(i:max(Degree)));
% %             cumdegree(i)=sum(D_DistN(1:i));
% 
%         end
%         cumdegree=cumdegree/max(cumdegree);
      
% % load('elec.mat');


% temp=sum(abs(A));
% x=find(temp>1);
% B=A(x,x);
% A=B;

%  n=length(A);
%  n=7000;
%  n=length(A);
%  A1=A;
%  A2=A;
%  for i=1:n
%      A(i,i)=0;
%      for j=1:n
%          if(A(i,j)<0)
%              A1(i,j)=0;
%          end
%          if(A(i,j)>0)
%              A2(i,j)=0;
%          end
%      end
%  end
%  
%  pd=sum(A1);
%  nd=sum(A2);
%  plot(pd+nd,'.');
%  hold on;
%  plot(-nd,'. k');
%  break;
% 
% Degree=sum(abs(A));
% 
% D=sum(abs(A));
% 
% D_DistN=zeros(1,max(D));
% 
%     for j=1:n
%         for k=1:n	
%             if(D(1,k)==j)
%                 D_DistN(1,j)=D_DistN(1,j)+1;
%             end
%         end
%     end
%     D_DistN=D_DistN/sum(D_DistN);
% 
%         cumdegree=zeros((max(Degree)),1);
%         for i=1:max(Degree)
%             cumdegree(i)=sum(D_DistN(i:max(Degree)));
% %             cumdegree(i)=sum(D_DistN(1:i));
% 
%         end
%         cumdegree=cumdegree/max(cumdegree);
%         index=1:length(cumdegree);
%         loglog(index,cumdegree,'o k');
%         hold on;
%                     
% T=zeros(4,10);
% pT=zeros(4,10);
% ET=zeros(4,10);
% sT=zeros(4,10);
% pT_0=zeros(4,10);
% Edges=zeros(2,10);
% delta=zeros(1,10);
% p=zeros(1,10);

% pause(1)

% beta=.65;
% p1=.14;
% p2=.05;
% a11=.55;
% a12=.3;
% a22=.3;
% a21=0.18;
navd=ceil(sum(NegDegree)/n1);
pavd=ceil(sum(PosDegree)/n1);
alpha=corrcoef(PosDegree,NegDegree);


a11=1;
a12=alpha(2,1)*a11;
a22=sum(NegDegree)/sum(PosDegree);
a21=alpha(2,1)*a22;
error1=n;
x=0.01:.01:1;
Er=zeros(length(x),length(x),length(x));
epsilon1=x*(t2+t0)/(t0+t3+t2+t4);
Error=n;

deltaplus=t3/(t3+t4);
for i1=1:length(epsilon1)
    epsilon=epsilon1(i1);
for i=1:length(x)
    beta=x(i);
    for j=1:length(x)
    p=x(j);
   c11=a11*beta/2+(1-beta)*p*deltaplus+p*epsilon/(1-epsilon);
   c22=a22*beta/2+(1-beta)*deltaplus*p/2+p*((1-deltaplus)/deltaplus)*epsilon/(1-epsilon);
   c12=a12*beta/2+(1-beta)*p*(1-deltaplus)+p*epsilon/(1-epsilon);
   c21=a21*beta/2+(1-beta)*p*(1-deltaplus)/2+((1-deltaplus)/deltaplus)*p*epsilon/(1-epsilon);
   
%    if(Error>=abs((c11-1/(gamma-1))*(c22-1/(gamma-1))-c12*c21))
%        Error=abs((c11-1/(gamma-1))*(c22-1/(gamma-1))-c12*c21);
%        prob1=p;
%        beta1=beta;
%    end

Er(i,j,i1)=abs((c11-1/(gamma))*(c22-1/(gamma))-c12*c21);
    if(Error>=abs((c11-1/(gamma))*(c22-1/(gamma))-c12*c21))
       Error=abs((c11-1/(gamma))*(c22-1/(gamma))-c12*c21);
       prob1=p;
       beta1=beta;
       epsilon2=epsilon;
    end
   
    end  
end
end

% 
% for i1=1:length(x)
%     for i=1:length(x)
%         for j=1:length(x)
%         plot3(i,j,i1,'.','color',[1,0,0]);
%         hold on;
% 
%         end
%     end
% end


prob1;
beta1;
epsilon2=epsilon2*(t2+t0+t3+t4)/(t0+t2);




xy2=beta1;
y2=prob1;
xy2=0.1:.05:.7;
y2=0.1:.05:.7;
epsilon1=epsilon;   
epsilon2=0.1:0.1:.9;        
        
for t43=1:length(xy2)
for t53=1:length(y2)

    for t66=1:length(epsilon2)      
        
        
NElist=zeros(10*navd,2);
PElist=zeros(10*pavd,2);
pdegree=zeros(1,n);
ndegree=zeros(1,n);
count1=3;
count2=2;
alpha=corrcoef(PosDegree,NegDegree);
cd=zeros(1,n);

        
for t1=1:1
beta=xy2(t43);
epsilon=epsilon2(t66)*(t2+t0)/(t3+t4+t0+t2);
a11=1;
a12=alpha(2,1)*a11;
a21=alpha(2,1)*a22;


%A=zeros(n,n);

i=3;
PElist(1,1)=1;
PElist(1,2)=2;
pdegree(1,1)=pdegree(1,1)+1;
pdegree(1,2)=pdegree(1,2)+1;
PElist(2,1)=1;
PElist(2,2)=3;
pdegree(1,1)=pdegree(1,1)+1;
pdegree(1,3)=pdegree(1,3)+1;
% PElist(3,1)=1;
% PElist(3,2)=4;
% pdegree(1,1)=pdegree(1,1)+1;
% pdegree(1,4)=pdegree(1,4)+1;
% PElist(4,1)=1;
% PElist(4,2)=6;
% pdegree(1,1)=pdegree(1,1)+1;
% pdegree(1,6)=pdegree(1,6)+1;
% PElist(5,1)=1;
% PElist(5,2)=5;
% pdegree(1,1)=pdegree(1,1)+1;
% pdegree(1,5)=pdegree(1,5)+1;

NElist(1,1)=2;
NElist(1,2)=3;
ndegree(1,2)=ndegree(1,2)+1;
ndegree(1,3)=ndegree(1,3)+1;

while(i<n)
    
    
    
 nolink=0;   

%        p1=y2(t53);
    if(i>n-length(find(Degree==1)))

        nolink=1;
        i=i+1;
        
        v1=a11*pdegree+a12*ndegree;
        v1=v1/i;
        v2=a21*pdegree+a22*ndegree;
        v2=v2/i;
    
   
        while(1)
        f1=zeros(1,i-1);
        f2=f1;
        tempx=rand(1,i-1)-0.5;
        x1=find(tempx<=0);
        f1(x1)=1;
        if(rand(1,1)>=0.5)
        y1=find(tempx>0);
        tempx=rand(1,i-1)-v1(1,1:(i-1));
        x1=find(tempx<=0);
        f2(x1)=1;
        f=f1.*f2;
        
        x1=find(f==1);
         
            if(~isempty(x1))
%                 for tkd=1:length(x1)
%                     if(rand(1,1)<=v1(x1(tkd)))
%                     [tempv, index]=sort(v1(x1));
%                     y1=index(length(x1));
                    y1=randi(length(x1),1);
                    x1=x1(y1);
                    
                    PElist(count1:count1+length(x1)-1,1)=x1';
                    PElist(count1:count1+length(x1)-1,2)=i;
                    pdegree(x1)=pdegree(x1)+1;
                    pdegree(i)=pdegree(i)+length(x1);
                    count1=count1+length(x1);
                    break;
%                     end
%                 end
            end
        else
        f1=zeros(1,i-1);
        f2=f1;
        f1(y1)=1;
        tempx=rand(1,i-1)-v2(1,1:(i-1));
        x1=find(tempx<=0);
        f2(x1)=1;
        f=f1.*f2;
        
        x1=find(f==1);
            if(~isempty(x1))
%                 for tkd=1:length(x1)
%                     if(rand(1,1)<=v2(x1(tkd)))
%                     [tempv, index]=sort(v2(x1));
%                     y1=index(length(x1));
                    y1=randi(length(x1),1);
                    x1=x1(y1);

                    NElist(count2:count2+length(x1)-1,1)=x1';
                    NElist(count2:count2+length(x1)-1,2)=i;
                    ndegree(x1)=ndegree(x1)+1;
                    ndegree(i)=ndegree(i)+length(x1);
                    count2=count2+length(x1);
                    break;
%                     end
%                 end
            end
        
        
        end
        end
        
        
        
        
%         p1=.5; %.028;
    else
       p1=y2(t53);
%     end
%     p2=p1*sum(NegDegree)/sum(PosDegree);
p2=p1/2;
    
    if(rand(1,1)<=(1-epsilon))%0.95
    if(rand(1,1)<=beta)
    
% d1=sum(A);
% d2=sum(B);
    % positive links
    v1=a11*pdegree+a12*ndegree;
    v1=v1/i;
    v2=a21*pdegree+a22*ndegree;
    v2=v2/i;
    linked=0;
    i=i+1;
    while(1)
        f1=zeros(1,i-1);
        f2=f1;
        tempx=rand(1,i-1)-0.5;
        x1=find(tempx<=0);
        f1(x1)=1;
        
        y1=find(tempx>0);
        tempx=rand(1,i-1)-v1(1,1:(i-1));
        x1=find(tempx<=0);
        f2(x1)=1;
        f=f1.*f2;
        
        x1=find(f==1);
        if(~isempty(x1))
        PElist(count1:count1+length(x1)-1,1)=x1';
        PElist(count1:count1+length(x1)-1,2)=i;
        pdegree(x1)=pdegree(x1)+1;
        pdegree(i)=pdegree(i)+length(x1);
        count1=count1+length(x1);
        linked=1;
        end
        f1=zeros(1,i-1);
        f2=f1;
        f1(y1)=1;
        tempx=rand(1,i-1)-v2(1,1:(i-1));
        x1=find(tempx<=0);
        f2(x1)=1;
        f=f1.*f2;
        
        x1=find(f==1);
        if(~isempty(x1))
        NElist(count2:count2+length(x1)-1,1)=x1';
        NElist(count2:count2+length(x1)-1,2)=i;
        ndegree(x1)=ndegree(x1)+1;
        ndegree(i)=ndegree(i)+length(x1);
        count2=count2+length(x1);
        linked=1;
        end
        if(linked==1)
            break;
        end
    end
        
        
        
        
%           while(1)
%             for j=1:i-1
%                 if(rand(1,1)<=0.5)
%                     if(rand(1,1)<=v1(j))&&(B(i,j)==0)
%                         f=1;
%                         A(i,j)=1;
%                         A(j,i)=1;
%                     end
%                 else
%                     if(rand(1,1)<=v2(j))&&(A(i,j)==0)
%                         f=1;
%                         B(i,j)=1;
%                         B(j,i)=1;
%                     end
%                 end
%             end
%             if(f==1)
%                 break;
%             end
%           end
    
%             for j=1:i-1
%                 if(rand(1,1)<=v2(j)/2)&&(A(i,j)==0)
%                     B(i,j)=1;
%                     B(j,i)=1;
%                 end
%             end
%     end
    else
        if(nolink==0)
        i=i+1;
        j=randi(i-1,1);
%      if(rand(1,1)<=sum(NegDegree)/sum(PosDegree+NegDegree))
     if(rand(1,1)<=sum(t4)/sum(t4+t3))
%         B(i,j)=1;
%         B(j,i)=1;
        NElist(count2,1)=j;
        NElist(count2,2)=i;
        count2=count2+1;
        ndegree(i)=ndegree(i)+1;
        ndegree(j)=ndegree(j)+1;
        
%         Nb1=find(A(:,j)==1);
        tempx=find(PElist(:,2)==j);
        tempx=PElist(tempx,1);
        tempx1=find(PElist(:,1)==j);
        tempx1=PElist(tempx1,2);
        Nb1=[tempx',tempx1'];
        
        
%         Nb2=find(B(:,j)==1);
        
        tempx=find(NElist(:,2)==j);
        tempx=NElist(tempx,1);
        tempx1=find(NElist(:,1)==j);
        tempx1=NElist(tempx1,2);
        Nb2=[tempx',tempx1'];
        
        
        
%         for j=1:length(Nb1)
%             if(rand(1,1)<=p1)&&(A(i,Nb1(j))==0)
%                 B(i,Nb1(j))=1;
%                 B(Nb1(j),i)=1;
%             end
%         end
        
        tempx=rand(length(Nb1),1)-p2;
        tempx=find(tempx<=0);
        NElist(count2:count2+length(tempx)-1,1)=Nb1(tempx)';
        NElist(count2:count2+length(tempx)-1,2)=i;
        count2=count2+length(tempx);
        ndegree(Nb1(tempx))=ndegree(Nb1(tempx))+1;
        ndegree(i)=ndegree(i)+length(tempx);

        
        
        
        
%         for j=1:length(Nb2)
%             if(rand(1,1)<=p2)&&(B(i,Nb2(j))==0)
%                 A(i,Nb2(j))=1;
%                 A(Nb2(j),i)=1;
%             end
%         end
        
        tempx=rand(length(Nb2),1)-p1;
        tempx=find(tempx<=0);
        PElist(count1:count1+length(tempx)-1,1)=Nb2(tempx)';
        PElist(count1:count1+length(tempx)-1,2)=i;
        count1=count1+length(tempx);
        pdegree(Nb2(tempx))=pdegree(Nb2(tempx))+1;
        pdegree(i)=pdegree(i)+length(tempx);
        
        
        
        
        else
%         A(i,j)=1;
%         A(j,i)=1;
        PElist(count1,1)=j;
        PElist(count1,2)=i;
        count1=count1+1;
        
%         Nb1=find(A(:,j)==1);
        %         Nb1=find(A(:,j)==1);
        tempx=find(PElist(:,2)==j);
        tempx=PElist(tempx,1);
        tempx1=find(PElist(:,1)==j);
        tempx1=PElist(tempx1,2);
        Nb1=[tempx',tempx1'];
        
        
%         Nb2=find(B(:,j)==1);
        %         Nb2=find(B(:,j)==1);
        
        tempx=find(NElist(:,2)==j);
        tempx=NElist(tempx,1);
        tempx1=find(NElist(:,1)==j);
        tempx1=NElist(tempx1,2);
        Nb2=[tempx',tempx1'];
        
        
        
        
        
%         for j=1:length(Nb1)
%             if(rand(1,1)<=p1)&&(B(i,Nb1(j))==0)
%                 A(i,Nb1(j))=1;
%                 A(Nb1(j),i)=1;
%             end
%         end
        

        tempx=rand(length(Nb1),1)-p1;
        tempx=find(tempx<=0);
        PElist(count1:count1+length(tempx)-1,1)=Nb1(tempx)';
        PElist(count1:count1+length(tempx)-1,2)=i;
        count1=count1+length(tempx);
        pdegree(Nb1(tempx))=pdegree(Nb1(tempx))+1;
        pdegree(i)=pdegree(i)+length(tempx);
        
        
        
        
        
        
        
        
        
%         
%         for j=1:length(Nb2)
%             if(rand(1,1)<=p2)&&(A(i,Nb2(j))==0)
%                 B(i,Nb2(j))=1;
%                 B(Nb2(j),i)=1;
%             end
%         end
        
        

        tempx=rand(length(Nb2),1)-p2;
        tempx=find(tempx<=0);
        NElist(count2:count2+length(tempx)-1,1)=Nb2(tempx)';
        NElist(count2:count2+length(tempx)-1,2)=i;
        count2=count2+length(tempx);
        ndegree(Nb2(tempx))=ndegree(Nb2(tempx))+1;
        ndegree(i)=ndegree(i)+length(tempx);
        
        
    end
        
        end
    end
    else
            
        if(nolink==0)
%          d1=sum(A);
%         d2=sum(B);
        
    % positive links
%     v1=a11*d1+a12*d2;
%     v1=.5*v1/i;
%     v2=a21*d1+a22*d2;
%     v2=.5*v2/i;
v1=pdegree/i;
v2=ndegree/i;
    
%     temp1=rand(1,n);
%     temp1=v1-temp1;
%     x=find(temp1>=0);
    x=randi(i-1,1);
%     if(~isempty(x))
%         for j=1:1%length(x)
%             temp1=rand(1,n);
%             temp1=v1-temp1;
%             y=find(temp1>=0);
%             temp1=rand(1,n);
%             temp1=v2-temp1;
%             y1=find(temp1>=0);

%            Nb1=find(A(:,x)==1);
        tempx=find(PElist(:,2)==x);
        tempx=PElist(tempx,1);
        tempx1=find(PElist(:,1)==x);
        tempx1=PElist(tempx1,2);
        Nb1=[tempx',tempx1'];
           
%             Nb2=find(B(:,x)==1);
        
        tempx=find(NElist(:,2)==x);
        tempx=NElist(tempx,1);
        tempx1=find(NElist(:,1)==x);
        tempx1=NElist(tempx1,2);
        Nb2=[tempx',tempx1'];
            
            
            
%             for j=1:i-1
            if(rand(1,1)<0.5)
                
                f1=zeros(1,n);
                f1(Nb1)=1;
            for k=1:length(Nb1) %++-
                f2=zeros(1,n);
                
%                 y=find(A(:,Nb1(k))==1);
                tempx=find(PElist(:,2)==Nb1(k));
                tempx=PElist(tempx,1);
                tempx1=find(PElist(:,1)==Nb1(k));
                tempx1=PElist(tempx1,2);
                y=[tempx',tempx1'];
                
                
                if(~isempty(y))
%                 
%                 y1=randi(length(y),1);
% %                 for y1=1:length(y)
%                 if(rand(1,1)<p2)%&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
% %                     B(x,y(y1))=1;
% %                     B(y(y1),x)=1;
%                     NElist(count2,1)=y(y1);
%                     NElist(count2,2)=x;
%                     count2=count2+1;
%                     ndegree(y(y1))=ndegree(y(y1))+1;
%                     ndegree(x)=ndegree(x)+1;
% 
%                     
%                     
%                 end
% %                 end
                    f2(y)=1;
                    f2(x)=0;
                    f=f1.*f2;
                    f2=f2-f;
                    y=find(f2==1);
                    if(~isempty(y))
                    y1=randi(length(y),1);
                    y=y(y1);
                    
                    
                    
                    tempx=rand(length(y),1)-p2;
                    tempx=find(tempx<=0);
                    if(~isempty(tempx))
                    NElist(count2:count2+length(tempx)-1,1)=y(tempx)';
                    NElist(count2:count2+length(tempx)-1,2)=x;
                    count2=count2+length(tempx);
                    ndegree(y(tempx))=ndegree(y(tempx))+1;
                    ndegree(x)=ndegree(x)+length(tempx);
                    end
                    end





                end
            end
            
            for k=1:length(Nb1) %+-+
                f2=zeros(1,n);
%                 y=find(B(:,Nb1(k))==1);
                tempx=find(NElist(:,2)==Nb1(k));
                tempx=NElist(tempx,1);
                tempx1=find(NElist(:,1)==Nb1(k));
                tempx1=NElist(tempx1,2);
                y=[tempx',tempx1'];
                
                
                if(~isempty(y))
%                 
%                 y1=randi(length(y),1);
% %                 for y1=1:length(y)
%                 if(rand(1,1)<p1)%&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
% %                     A(x,y(y1))=1;
% %                     A(y(y1),x)=1;
%                     PElist(count1,1)=y(y1);
%                     PElist(count1,2)=x;
%                     count1=count1+1;
%                     pdegree(y(y1))=pdegree(y(y1))+1;
%                     pdegree(x)=pdegree(x)+1;
% 
%                     
%                     
%                 end
% %                 end
                    f2(y)=1;
                    f2(x)=0;
                    f=f1.*f2;
                    f2=f2-f;
                    y=find(f2==1);
                    if(~isempty(y))
                    y1=randi(length(y),1);
                    y=y(y1);
                    
                    
                    tempx=rand(length(y),1)-p1;
                    tempx=find(tempx<=0);
                    if(~isempty(tempx))
                    PElist(count1:count1+length(tempx)-1,1)=y(tempx)';
                    PElist(count1:count1+length(tempx)-1,2)=x;
                    count1=count1+length(tempx);
                    pdegree(y(tempx))=pdegree(y(tempx))+1;
                    pdegree(x)=pdegree(x)+length(tempx);
                    end
                    end





                end
            end
            
            else
                f1=zeros(1,n);
                f1(Nb2)=1;
             for k=1:length(Nb2) %-++
                 f2=zeros(1,n);
                 
%                 y=find(A(:,Nb2(k))==1);
                tempx=find(PElist(:,2)==Nb2(k));
                tempx=PElist(tempx,1);
                tempx1=find(PElist(:,1)==Nb2(k));
                tempx1=PElist(tempx1,2);
                y=[tempx',tempx1'];
                
                
                if(~isempty(y))
%                 y1=randi(length(y),1);
% %                 for y1=1:length(y)
%                 if(rand(1,1)<p1)%&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
% %                     A(x,y(y1))=1;
% %                     A(y(y1),x)=1;
%                     PElist(count1,1)=y(y1);
%                     PElist(count1,2)=x;
%                     count1=count1+1;
%                     pdegree(y(y1))=pdegree(y(y1))+1;
%                     pdegree(x)=pdegree(x)+1;
%                 end
% %                 end
                    f2(y)=1;
                    f2(x)=0;
                    f=f1.*f2;
                    f2=f2-f;
                    y=find(f2==1);
                    if(~isempty(y))
                    y1=randi(length(y),1);
                    y=y(y1);
                    
                    tempx=rand(length(y),1)-p1;
                    tempx=find(tempx<=0);
                    if(~isempty(tempx))
                    PElist(count1:count1+length(tempx)-1,1)=y(tempx)';
                    PElist(count1:count1+length(tempx)-1,2)=x;
                    count1=count1+length(tempx);
                    pdegree(y(tempx))=pdegree(y(tempx))+1;
                    pdegree(x)=pdegree(x)+length(tempx);
                    end
                    end




                end
             end
            
             for k=1:length(Nb2) %---
                 f2=zeros(1,n);
%                 y=find(B(:,Nb2(k))==1);
                tempx=find(NElist(:,2)==Nb2(k));
                tempx=NElist(tempx,1);
                tempx1=find(NElist(:,1)==Nb2(k));
                tempx1=NElist(tempx1,2);
                y=[tempx',tempx1'];
                
                
                if(~isempty(y))
%                 y1=randi(length(y),1);
% %                 for y1=1:length(y)
%                 if(rand(1,1)<p2)%&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
% %                     B(x,y(y1))=1;
% %                     B(y(y1),x)=1;
%                     NElist(count2,1)=y(y1);
%                     NElist(count2,2)=x;
%                     count2=count2+1;
%                     ndegree(y(y1))=ndegree(y(y1))+1;
%                     ndegree(x)=ndegree(x)+1;
%                 end
% %                 end
                    f2(y)=1;
                    f2(x)=0;
                    f=f1.*f2;
                    f2=f2-f;
                    y=find(f2==1);
                    if(~isempty(y))
                    y1=randi(length(y),1);
                    y=y(y1);
                    
                    tempx=rand(length(y),1)-p2;
                    tempx=find(tempx<=0);
                    if(~isempty(tempx))
                    NElist(count2:count2+length(tempx)-1,1)=y(tempx)';
                    NElist(count2:count2+length(tempx)-1,2)=x;
                    count2=count2+length(tempx);
                    ndegree(y(tempx))=ndegree(y(tempx))+1;
                    ndegree(x)=ndegree(x)+length(tempx);
                    end
                    end





                end
            end
             
             

            end
%             end
%         end
%     end
%     temp1=rand(1,n);
%     temp1=v2-temp1;
%     x=find(temp1>=0);
%     if(~isempty(x))
%         for j=1:length(x)
%             temp1=rand(1,n);
%             temp1=v2-temp1;
%             y=find(temp1>=0);
%             if(~isempty(y))
%             for k=1:length(y)
%                 if(A(x(j),y(k))==0)&&(B(x(j),y(k))==0)
%                     B(x(j),y(k))=1;
%                     B(y(k),x(j))=1;
%                 end
%             end
%             end
%         end
%       
%         
%     end
        end
    end
    end
if(mod(i,10000)==0)
%     
%     [T1,T2,T3,T0]=property(A(1:i,1:i),B(1:i,1:i));
%     BT(1,i/1000)=T1+T3;
%     BT(2,i/1000)=T2;
    
    i
end
    
end 
 
 %AA=A+B;
% BB=A+B;
% A=BB;
% B=AA;
% A=AA;
% 

count=count1+count2-2;
Elist=zeros(count,2);
Elist1=zeros(count,3);
Elist(1:count1-1,1:2)=PElist(1:count1-1,1:2);
Elist(1:count1-1,3)=1;
Elist(count1:count,1:2)=NElist(1:count2-1,1:2);
Elist(count1:count,3)=-1;

list=zeros(2*count,1);

list(1:count,1)=Elist(:,1);
list(count+1:2*count,1)=Elist(:,2);




D=tabulate(list);
% D=tabulate(list);
x=zeros(n,1);
x(D(:,1))=(D(:,2));


F=tabulate(x);
F(:,2)=F(:,2);
FF=zeros(1,max(F(:,1)));
if(F(1,1)==0)
FF(F(2:length(F),1))=F(2:length(F),2);
else
FF(F(1:length(F),1))=F(1:length(F),2);
end
for i=2:length(FF)
    FF(length(FF)-i+1)=FF(length(FF)-i+2)+FF(length(FF)-i+1);
end
cumdegree1=FF/FF(1);


% 
% Degree=sum(AA);
% 
% D=sum(AA);
% 
% D_DistN=zeros(1,max(D));
% 
%     for j=1:n
%         for k=1:n	
%             if(D(1,k)==j)
%                 D_DistN(1,j)=D_DistN(1,j)+1;
%             end
%         end
%     end
%     D_DistN=D_DistN/sum(D_DistN);
% 
%         cumdegree1=zeros((max(Degree)),1);
% % cumdegree=zeros(1,n);
%         for i=1:max(Degree)
%             cumdegree1(i)=sum(D_DistN(i:max(Degree)));
% %             cumdegree(i)=sum(D_DistN(1:i));
% 
%         end
%         cumdegree1=cumdegree1/max(cumdegree1);
%         
%       








        
        %%
        
tempElist=Elist';
m=length(tempElist);
tempElist=tempElist';
Elist=zeros(2*m,3);
Elist(1:m,:)=tempElist;
Elist((m+1):2*m,1)=tempElist(:,2);
Elist((m+1):2*m,2)=tempElist(:,1);
Elist((m+1):2*m,3)=sign(tempElist(:,3));
% Elist((m+1):2*m,4)=tempElist(:,4);

n=length(unique(Elist(:,1)));
n1=max(Elist(:,1));
tri1=zeros(1,n1);
tri2=zeros(1,n1);
tri3=zeros(1,n1);
tri4=zeros(1,n1);

parfor i=1:n1
    
    f1=zeros(n1,1);
%     if(mod(i,10000)==0)
%         i
%     end
    temp=find(Elist(:,1)==i);
    if(~isempty(temp))
    x1=Elist(temp,2);
    f1(x1,1)=Elist(temp,3);
    for j=1:length(x1)
        f2=zeros(n1,1);

        if(f1(x1(j))==1)
            temp=find(Elist(:,1)==x1(j));
            if(~isempty(temp))
                x2=Elist(temp,2);
                f2(x2,1)=Elist(temp,3);
                tempx=f1.*f2;
                    tempy=length(find(tempx==-1));
                    tri2(i)=tri2(i)+tempy; % ++- triangles
                    tempy=(find(tempx==1));
                    tempz=f1(tempy);
                    tempz1=length(find(tempz==1));
                    tri1(i)=tri1(i)+tempz1; % +++
                    tempz1=length(find(tempz==-1));
                    tri3(i)=tri3(i)+tempz1; % +--
                
%                 tri(1,i)=tri(1,i)+sum(f1.*f2);
        %         for k=1:length(x2)
        %         temp=find(Elist(:,1)==k);
        %         x3=Elist(temp,2);
        %         tri(1,i)=tri(1,i)+length(find(x3==i));

        %         end
            end
        else
            temp=find(Elist(:,1)==x1(j));
            if(~isempty(temp))
                x2=Elist(temp,2);
                f2(x2,1)=Elist(temp,3);
                tempx=f1.*f2;
                    tempy=length(find(tempx==-1));
                    tri3(i)=tri3(i)+tempy; % -+- triangles
                    tempy=(find(tempx==1));
                    tempz=f1(tempy);
                    tempz1=length(find(tempz==1));
                    tri2(i)=tri2(i)+tempz1; % -++
                    tempz1=length(find(tempz==-1));
                    tri4(i)=tri4(i)+tempz1; % ---
%                 tri(1,i)=tri(1,i)+sum(f1.*f2);
        %         for k=1:length(x2)
        %         temp=find(Elist(:,1)==k);
        %         x3=Elist(temp,2);
        %         tri(1,i)=tri(1,i)+length(find(x3==i));

        %         end
            end
            
        end
        
    end
    end

end
xyz3= sum(tri1)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));
xyz2=sum(tri2)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));
xyz1=sum(tri3)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));
xyz0=sum(tri4)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));


        %%
        
        
        
        

        if(length(cumdegree1)<=length(cumdegree))
      error=sum(abs((log(cumdegree1))-(log(cumdegree(1:length(cumdegree1))))))/n+sum(abs(log(cumdegree(1+length(cumdegree1):length(cumdegree)))))/n+abs((t3-sum(tri1))/t3)+abs(xyz3-T3data)/T3data+abs(xyz2-T2data)/T2data+abs(xyz1-T1data)/T1data+abs(xyz0-T0data)/T0data;
        else
      error=sum(abs(log(cumdegree)-log(cumdegree1(1:length(cumdegree)))))/n+sum(abs(log(cumdegree1(1+length(cumdegree):length(cumdegree1)))))/n+abs((t3-sum(tri1))/t3)+abs(xyz3-T3data)/T3data+abs(xyz2-T2data)/T2data+abs(xyz1-T1data)/T1data+abs(xyz0-T0data)/T0data;
        end
        
        
        if (error<error1)
            error1=error
            cumdegree2=cumdegree1;
            prob=p1
            Betaa=beta
%             AA1=A;
%             BB1=B;

FinalElist=Elist;



        cumdegree5(1:length(cumdegree2))=cumdegree2;

        
        
        
        
 loglog(cumdegree,'k o')
        
        hold on;
        
        loglog(cumdegree1,' *')
        hold on;       
        pause(1)

        end
        
        
        
        
        
        
        
end
% t66
    end
%     t53
end
t43
end   
        
          loglog(cumdegree,'k o')
        
        hold on;
        
        loglog(cumdegree2,' *')
        hold on;
        pause(1)
% %         cd=cd+cumdegree;
%       
%         %%
% A=B;       
% n_edges=sum(sum(abs(A)));
% Edges(1,t1)=length(find(A==1))/2;
% Edges(2,t1)=length(find(A==-1))/2;
% 
% p(1,t1)=length(find(A==1))/(sum(sum(abs(A))));
% 
% 
% n=length(A);
% % for ii=1:n
% %     for j=1:n
% %         for k=1:n
% %             if(A(ii,j)==1)&&(A(ii,k)==1)&&(A(k,j)==1)
% %                 T(1,i)=T(1,i)+1;
% %             end
% %             if(A(ii,j)~=0)&&(A(ii,k)~=0)&&(A(k,j)~=0)
% %                 delta(1,i)=delta(1,i)+1;
% %             end
% %             if((A(ii,j)==1)&&(A(ii,k)==-1)&&(A(k,j)==-1))||((A(ii,j)==-1)&&(A(ii,k)==1)&&(A(k,j)==-1))||((A(ii,j)==-1)&&(A(ii,k)==-1)&&(A(k,j)==1))
% %                 T(2,i)=T(2,i)+1;
% %             end
% %             if((A(ii,j)==1)&&(A(ii,k)==1)&&(A(k,j)==-1))||((A(ii,j)==-1)&&(A(ii,k)==1)&&(A(k,j)==1))||((A(ii,j)==1)&&(A(ii,k)==-1)&&(A(k,j)==1))
% %                 T(3,i)=T(3,i)+1;
% %             end
% %             if((A(ii,j)==-1)&&(A(ii,k)==-1)&&(A(k,j)==-1))
% %                 T(4,i)=T(4,i)+1;
% %             end
% %         end
% %     end
% % end
% xx=0;
% t3=0; % All +
% t0=0; % All -
% t2=0; % 2 plus one negative
% t4=0; % 2 negative one plus
% xxx=0;      
% result=A;
% for ii=1:length(result)
%     for j=ii+1:length(result)
%         if abs(result(ii,j))==1
%             for k=j+1:length(result)
%                 if abs(result(ii,k))==1 && abs(result(j,k))==1
%                     xx=xx+1;
%                     
%                     if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==1
%                         t3=t3+1;
%                     else if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==1
%                             t2=t2+1;
%                             
%                         else if result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==1
%                                 t4=t4+1;
%                                 
%                                 
%                             else if result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==-1
%                                     t0=t0+1;
%                                  end
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end
% 
% T(1,t1)=t3;%+++
% T(2,t1)=t2;%++-
% T(3,t1)=t4;%+--
% T(4,t1)=t0;%---
% 
% delta(t1)=t0+t4+t2+t3;
% 
% count=0;
% B=A;
% while(1)
% 
%     while(1)
%     r1=randi(n,1);
%     c1=randi(n,1);
%     if(B(r1,c1)~=0)
%         break;
%     end
%     end
%     
%     while(1)
%     r2=randi(n,1);
%     c2=randi(n,1);
%     if(B(r2,c2)~=0)
%         break;
%     end
%     end
%     
%     temp=B(r1,c1);
%     B(r1,c1)=B(r2,c2);
%     B(c1,r1)= B(r1,c1);
%     B(r2,c2)=temp;
%     B(c2,r2)=B(r2,c2);
%     count=count+1;
% if(count>=n_edges)
%     break;
% end
% 
% end
% 
% 
% xx=0;
% t3=0; % All +
% t0=0; % All -
% t2=0; % 2 plus one negative
% t4=0; % 2 negative one plus
% xxx=0;      
% result=B;
% for ii=1:length(result)
%     for j=ii+1:length(result)
%         if abs(result(ii,j))==1
%             for k=j+1:length(result)
%                 if abs(result(ii,k))==1 && abs(result(j,k))==1
%                     xx=xx+1;
%                     
%                     if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==1
%                         t3=t3+1;
%                     else if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==1
%                             t2=t2+1;
%                             
%                         else if result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==1
%                                 t4=t4+1;
%                                 
%                                 
%                             else if result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==-1
%                                     t0=t0+1;
%                                  end
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end
% d=t0+t4+t2+t3;
% 
% pT_0(1,t1)=t3/d;%+++
% pT_0(2,t1)=t2/d;%++-
% pT_0(3,t1)=t4/d;%+--
% pT_0(4,t1)=t0/d;%---
% 
% 
% 
% 
% % 
%  pT(:,t1)=T(:,t1)./delta(t1);
% % pT_0(1,i)=(Edges(1,i)/sum(Edges(:,i)))*((Edges(1,i)-1)/(sum(Edges(:,i))-1))*((Edges(1,i)-2)/(sum(Edges(:,i))-2));
% % pT_0(2,i)=(Edges(1,i)/sum(Edges(:,i)))*((Edges(2,i))/(sum(Edges(:,i))-1))*((Edges(2,i)-1)/(sum(Edges(:,i))-2));
% % pT_0(3,i)=(Edges(1,i)/sum(Edges(:,i)))*((Edges(1,i)-1)/(sum(Edges(:,i))-1))*((Edges(2,i))/(sum(Edges(:,i))-2));
% % pT_0(4,i)=(Edges(2,i)/sum(Edges(:,i)))*((Edges(2,i)-1)/(sum(Edges(:,i))-1))*((Edges(2,i)-2)/(sum(Edges(:,i))-2));
% ET(1,t1)=pT_0(1,t1)*delta(t1);
% ET(2,t1)=pT_0(2,t1)*delta(t1);
% ET(3,t1)=pT_0(3,t1)*delta(t1);
% ET(4,t1)=pT_0(4,t1)*delta(t1);
% 
% sT(1,t1)=(T(1,t1)-ET(1,t1))/(delta(t1)*pT_0(1,t1)*(1-pT_0(1,t1)))^(0.5);
% sT(2,t1)=(T(2,t1)-ET(2,t1))/(delta(t1)*pT_0(2,t1)*(1-pT_0(2,t1)))^(0.5);
% sT(3,t1)=(T(3,t1)-ET(3,t1))/(delta(t1)*pT_0(3,t1)*(1-pT_0(3,t1)))^(0.5);
% sT(4,t1)=(T(4,t1)-ET(4,t1))/(delta(t1)*pT_0(4,t1)*(1-pT_0(4,t1)))^(0.5);
% 
% 
% 
% % save('T_ex.mat','T');
% % save('pT_ex.mat','pT');
% % save('ET_ex.mat','ET');
% % save('sT_ex.mat','sT');
% % save('pT_0_ex.mat','pT_0');
% % save('p_ex.mat','p');
% % save('Edges_ex.mat','Edges');
% % save('delta_ex.mat','delta');
% 
% save('T.mat','T');
% save('pT.mat','pT');
% save('ET.mat','ET');
% save('sT.mat','sT');
% save('pT_0.mat','pT_0');
% save('p.mat','p');
% save('Edges.mat','Edges');
% save('delta.mat','delta');
%         
%         
%         
%         
% end
%         index=1:length(cumdegree);
%         loglog(index,cd/10,'o');
%         hold on;
%         
%         
%           D=sum(A+B)   ;    
%         x=D;
%         [alpha, xmin, L]=plfit(x);
%         h=plplot(x, xmin, alpha,1);
%         [p,gof]=plpva(x, xmin) ;

% 
% 
% 
% load('elec.mat');
% 
% t2=0;
% t3=0;
% t4=0;
% t0=0;
% T=zeros(4,n);
% result=A;
% 
%         for i=1:n	
%         x=find(result(i,:));
%              for j=1:length(x)
%                  y=find(result(x(j),:));
%                         for k=1:length(y)	
%                             if(y(k)~=i)
%                                    if result(i,x(j))==1 && result(i,y(k))==1 && result(x(j),y(k))==1
%                                     t3=t3+1;
%                                     T(1,i)=T(1,i)+1;
%                                 else if result(i,x(j))==1 && result(i,y(k))==1 && result(x(j),y(k))==-1 || result(i,x(j))==1 && result(i,y(k))==-1 && result(x(j),y(k))==1 || result(i,x(j))==-1 && result(i,y(k))==1 && result(x(j),y(k))==1
%                                         t2=t2+1;
%                                         T(2,i)=T(2,i)+1;
% 
%                                     else if result(i,x(j))==1 && result(i,y(k))==-1 && result(x(j),y(k))==-1 || result(i,x(j))==-1 && result(i,y(k))==1 && result(x(j),y(k))==-1 || result(i,x(j))==-1 && result(i,y(k))==-1 && result(x(j),y(k))==1
%                                             t4=t4+1;
%                                             T(3,i)=T(3,i)+1;
% 
% 
%                                         else if result(i,x(j))==-1 && result(i,y(k))==-1 && result(x(j),y(k))==-1
%                                                 t0=t0+1;
%                                                 T(4,i)=T(4,i)+1;
%                                              end
%                                         end
%                                     end
%                                     end
%                             end
%                         end
%              end

%         end

% load('elec.mat');
% A=abs(A);
% Degree=sum(A);
% 
% D=sum(A);
% 
% D_DistN=zeros(1,max(D));
% 
%     for j=1:n
%         for k=1:n	
%             if(D(1,k)==j)
%                 D_DistN(1,j)=D_DistN(1,j)+1;
%             end
%         end
%     end
%     D_DistN=D_DistN/sum(D_DistN);
% 
%         cumdegree=zeros((max(Degree)),1);
% % cumdegree=zeros(1,n);
%         for i=1:max(Degree)
%             cumdegree(i)=sum(D_DistN(i:max(Degree)));
% %             cumdegree(i)=sum(D_DistN(1:i));
% 
%         end
%         cumdegree=cumdegree/max(cumdegree);
%         loglog(cumdegree,'k o')
%         
        
%          x=sum(abs(A));
%         [alpha, xmin, L]=plfit(x);
%         h=plplot(x, xmin, alpha,1);
%         [p,gof]=plpva(x, xmin) ;
% % 
% p=0:.1:1;
% x=0:.1:1;
% Error=zeros(11,11);
% for i=1:11
%     aa=.7*.5.*x(i)+.7*.5.*(1-x(i)).*p+.3*.5-1/2.2;
%     bb=.7*.5.*x(i)*.25+.7*.5.*(1-x(i)).*p+.3*.5-1/2.2;
%     c=.7*.5.*x(i)*.6;
%     b=.7*.5.*x(i)*.25*.6;
%     Error(i,:)=aa.*bb-b.*c;
% end

  
%    
%  A=AA1-BB1; 
% xx=0;   
% t3=0; % All +
% t0=0; % All -
% t2=0; % 2 plus one negative
% t4=0; % 2 negative one plus
% xxx=0;      
% result=A;
% for ii=1:length(result)
%     for j=ii+1:length(result)
%         if abs(result(ii,j))==1
%             for k=j+1:length(result)
%                 if abs(result(ii,k))==1 && abs(result(j,k))==1
%                     xx=xx+1;
%                     
%                     if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==1
%                         t3=t3+1;
%                     else if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==1
%                             t2=t2+1;
%                             
%                         else if result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==1
%                                 t4=t4+1;
%                                 
%                                 
%                             else if result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==-1
%                                     t0=t0+1;
%                                  end
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end
%    
% 
% t0/(t0+t2+t3+t4)
% t4/(t0+t2+t3+t4)
% t2/(t0+t2+t3+t4)
% t3/(t0+t2+t3+t4)
Elist=FinalElist;
tempElist=Elist';
m=length(tempElist);
tempElist=tempElist';
Elist=zeros(2*m,3);
Elist(1:m,:)=tempElist;
Elist((m+1):2*m,1)=tempElist(:,2);
Elist((m+1):2*m,2)=tempElist(:,1);
Elist((m+1):2*m,3)=sign(tempElist(:,3));
% Elist((m+1):2*m,4)=tempElist(:,4);

n=length(unique(Elist(:,1)));
n1=max(Elist(:,1));
tri1=zeros(1,n1);
tri2=zeros(1,n1);
tri3=zeros(1,n1);
tri4=zeros(1,n1);

parfor i=1:n1
    
    f1=zeros(n1,1);
%     if(mod(i,10000)==0)
%         i
%     end
    temp=find(Elist(:,1)==i);
    if(~isempty(temp))
    x1=Elist(temp,2);
    f1(x1,1)=Elist(temp,3);
    for j=1:length(x1)
        f2=zeros(n1,1);

        if(f1(x1(j))==1)
            temp=find(Elist(:,1)==x1(j));
            if(~isempty(temp))
                x2=Elist(temp,2);
                f2(x2,1)=Elist(temp,3);
                tempx=f1.*f2;
                    tempy=length(find(tempx==-1));
                    tri2(i)=tri2(i)+tempy; % ++- triangles
                    tempy=(find(tempx==1));
                    tempz=f1(tempy);
                    tempz1=length(find(tempz==1));
                    tri1(i)=tri1(i)+tempz1; % +++
                    tempz1=length(find(tempz==-1));
                    tri3(i)=tri3(i)+tempz1; % +--
                
%                 tri(1,i)=tri(1,i)+sum(f1.*f2);
        %         for k=1:length(x2)
        %         temp=find(Elist(:,1)==k);
        %         x3=Elist(temp,2);
        %         tri(1,i)=tri(1,i)+length(find(x3==i));

        %         end
            end
        else
            temp=find(Elist(:,1)==x1(j));
            if(~isempty(temp))
                x2=Elist(temp,2);
                f2(x2,1)=Elist(temp,3);
                tempx=f1.*f2;
                    tempy=length(find(tempx==-1));
                    tri3(i)=tri3(i)+tempy; % -+- triangles
                    tempy=(find(tempx==1));
                    tempz=f1(tempy);
                    tempz1=length(find(tempz==1));
                    tri2(i)=tri2(i)+tempz1; % -++
                    tempz1=length(find(tempz==-1));
                    tri4(i)=tri4(i)+tempz1; % ---
%                 tri(1,i)=tri(1,i)+sum(f1.*f2);
        %         for k=1:length(x2)
        %         temp=find(Elist(:,1)==k);
        %         x3=Elist(temp,2);
        %         tri(1,i)=tri(1,i)+length(find(x3==i));

        %         end
            end
            
        end
        
    end
    end

end
% sum(tri1)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4))
% sum(tri2)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4))
% sum(tri3)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4))
% sum(tri4)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4))



xx1=xx1+sum(tri4)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));
xx2=xx2+sum(tri1)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));
xx3=xx3+sum(tri2)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));
xx4=xx4+sum(tri3)/(sum(tri1)+sum(tri2)+sum(tri3)+sum(tri4));


temp=corrcoef(tri1,tri3);

if(abs(temp(1,2))>0)

xy1=xy1+temp(1,2);
abc1=abc1+1;
end

temp=corrcoef(tri1,tri2);

if(abs(temp(1,2))>0)
xy2=xy2+temp(1,2);
abc2=abc2+1;
end

temp=corrcoef(tri1,tri4);

if(abs(temp(1,2))>0)
xy3=xy3+temp(1,2);
abc3=abc3+1;
end

temp=corrcoef(tri3,tri2);

if(abs(temp(1,2))>0)
xy4=xy4+temp(1,2);
abc4=abc4+1;
end

temp=corrcoef(tri3,tri4);

if(abs(temp(1,2))>0)
xy5=xy5+temp(1,2);
abc5=abc5+1;
end

temp=corrcoef(tri2,tri4);

if(abs(temp(1,2))>0)
xy6=xy6+temp(1,2);
abc6=abc6+1;
end












t3=sum(tri1);
t2=sum(tri2);
t0=sum(tri4);
t4=sum(tri3);


a=0;
        for i=1:n
            a=a+x(i)*(x(i)-1)/2;
        end
        
        
        Cs=Cs+((t3+t4)-(t0+t2))/a;
        SG=SG+((t3+t4)-(t0+t2))/((t3+t4)+(t0+t2));
        
        U=U+(1-SG)/(1+SG);









end    


% fraction of different types of triads
T0=xx1/abc % (---)
T3=xx2/abc %(+++)
T2=xx3/abc %(-++)
T1=xx4/abc % (--+)

% correlation between different types of triads

xy1=xy1/abc1;
xy2=xy2/abc2;
xy3=xy3/abc3;
xy4=xy4/abc4;
xy5=xy5/abc5;
xy6=xy6/abc6;

% different balancedness measures
Cs=Cs/abc;
SG=SG/abc;
U=U/abc;
 
cumdegree=cumdegree/max(cumdegree);
        loglog(cumdegree,'k o')
        
        hold on;
    loglog(cumdegree5/abc,' *')
        hold on;    