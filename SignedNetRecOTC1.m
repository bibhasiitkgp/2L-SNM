% function A=signed_MP1(~) 

clc;
  clear;
load('soc-sign-bitcoin.mat');
   
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
  newmanB=0;
 
  Ue=0;
  lambdan=0;
   
   n=length(A);
   
 cumdegree5=zeros(n,1);
   
   
   
   
   xx1=0;
   xx2=0;
   xx3=0;
   xx4=0;
for abc=1:10
  load('soc-sign-bitcoin.mat');
  n=length(A);
for i=1:n
    A(i,i)=0;
    for j=1:n
        if(A(i,j)~=0)
            A(j,i)=A(i,j);
        end
    end
end
   x=sum(abs(A));
   [gamma, xmin, L]=plfit(x);
   
   n=length(A);
   epsilon=0;
   for i=1:n
       epsilon=epsilon+sum(x(1:i))/i;
   end
   
   
   
  
  
xx=0;   
t3=0; % All +
t0=0; % All -
t2=0; % 2 plus one negative
t4=0; % 2 negative one plus
xxx=0;      
result=A;
for ii=1:length(result)
    for j=ii+1:length(result)
        if abs(result(ii,j))==1
            for k=j+1:length(result)
                if abs(result(ii,k))==1 && abs(result(j,k))==1
                    xx=xx+1;
                    
                    if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==1
                        t3=t3+1;
                    else if result(ii,j)==1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==1
                            t2=t2+1;
                            
                        else if result(ii,j)==1 && result(ii,k)==-1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==1 && result(j,k)==-1 || result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==1
                                t4=t4+1;
                                
                                
                            else if result(ii,j)==-1 && result(ii,k)==-1 && result(j,k)==-1
                                    t0=t0+1;
                                 end
                            end
                        end
                    end
                end
            end
        end
    end
end
   


 
if(abc==1)

T0data=t0/(t0+t2+t3+t4)
T3data=t3/(t0+t2+t3+t4)
T2data=t2/(t0+t2+t3+t4)
T1data=t4/(t0+t2+t3+t4)

end
   
   n=length(A);
   B=A;
   PosDegree=zeros(n,1);
   NegDegree=zeros(n,1);
   for i=1:n
     PosDegree(i)=length(find(A(:,i)==1));
     NegDegree(i)=length(find(A(:,i)==-1));
   end
D=x;
Degree=x;
D_DistN=zeros(1,max(D));

    for j=1:n
        for k=1:n	
            if(D(1,k)==j)
                D_DistN(1,j)=D_DistN(1,j)+1;
            end
        end
    end
    D_DistN=D_DistN/sum(D_DistN);

        cumdegree=zeros((max(Degree)),1);
% cumdegree=zeros(1,n);
        for i=1:max(Degree)
            cumdegree(i)=sum(D_DistN(i:max(Degree)));
%             cumdegree(i)=sum(D_DistN(1:i));

        end
       
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
        index=1:length(cumdegree);
        loglog(index,cumdegree,'o k');
        hold on;
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
alpha=corrcoef(PosDegree,NegDegree);

epsilon=(t2)/(epsilon);
a11=1;
a12=alpha(2,1)*a11;
a22=sum(NegDegree)/sum(PosDegree);
a21=alpha(2,1)*a22;
error1=n;
x=0:.1:1;
Error=n;
p=0;
beta=0;
for i=1:length(x)
    beta=x(i);
    for j=1:length(x)
    p=x(j);
   c11=(1-epsilon/p)*a11*beta/2+(1-epsilon/p)*(1-beta)*p*0.8+epsilon;
   c22=(1-epsilon/p)*a22*beta/2+(1-epsilon/p)*(1-beta)*0.8*p/2;
   c12=(1-epsilon/p)*a12*beta/2+(1-epsilon/p)*(1-beta)*p*0.8*(sum(NegDegree)/sum(PosDegree))+epsilon;
   c21=(1-epsilon/p)*a21*beta/2+(1-epsilon/p)*(1-beta)*0.8*(sum(NegDegree)/sum(PosDegree))*p/2+epsilon;
   
   if(Error>=abs((c11-1/(gamma))*(c22-1/(gamma))-c12*c21))
       Error=abs((c11-1/(gamma))*(c22-1/(gamma))-c12*c21);
       p1=p;
       beta1=beta;
   end
   
    end
    
    
    
end






 p1=0.45;
alpha=corrcoef(PosDegree,NegDegree);
cd=zeros(1,n);

beta=0.41;
x2=0.39;
%.38
y2=0.63;
%.61
epsilon1=epsilon;
for t43=1:length(x2)
    for t53=1:length(y2)

for t1=1:1
 beta=x2(t43);
 % .24
epsilon=0.19*(t2+t0)/(t3+t4+t0+t2);
a11=1;
a12=alpha(2,1)*a11;
a21=alpha(2,1)*a22;


A=zeros(n,n);

i=4;
A(1,2)=1;
A(2,1)=1;
A(3,2)=1;
A(2,3)=1;
A(4,2)=1;
A(2,4)=1;
A(1,3)=1;
A(3,1)=1;
A(4,3)=1;
A(3,4)=1;

B=zeros(n,n);

    B(1,4)=1;
    B(4,1)=1;

while(i<n)
 nolink=0;   
    if(i>n-length(find(Degree==1)))
        p1=0.00;
        beta=0.0;
        nolink=1;
        d1=sum(A);
        d2=sum(B);
    % positive links
    v1=a11*d1+a12*d2;
    v1=v1/i;
    v2=a21*d1+a22*d2;
    v2=v2/i;
    
    i=i+1;
    
        f=0;
          while(1)
            for j=1:i-1
%                 if(rand(1,1)<=0.5)
                if(rand(1,1)<=v1(j))&&(B(i,j)==0)
                    f=1;
                    A(i,j)=1;
                    A(j,i)=1;
                    break;
                end
%                 else
%                 if(rand(1,1)<=v2(j))&&(A(i,j)==0)
%                     f=1;
%                     B(i,j)=1;
%                     B(j,i)=1;
%                     break;
%                 end
%                 end
            end
            if(f==1)
                break;
            end
          end
        
        
        
%         i=i-1;
        
        
%         p1=.5; %.028;
    else
        p1=y2(t53);
        beta=x2(t43);
    end
    p2=p1/2;
    
    if(rand(1,1)<=(1-epsilon))%0.95
    if(rand(1,1)<=beta)
    
d1=sum(A);
d2=sum(B);
    % positive links
    v1=a11*d1+a12*d2;
    v1=v1/i;
    v2=a21*d1+a22*d2;
    v2=v2/i;
    
    i=i+1;
    
        f=0;
          while(1)
            for j=1:i-1
                if(rand(1,1)<=0.5)
                if(rand(1,1)<=v1(j))&&(B(i,j)==0)
                    f=1;
                    A(i,j)=1;
                    A(j,i)=1;
                end
                else
                if(rand(1,1)<=v2(j))&&(A(i,j)==0)
                    f=1;
                    B(i,j)=1;
                    B(j,i)=1;
                end
                end
            end
            if(f==1)
                break;
            end
          end
    
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
%         if(rand(1,1)<=sum(NegDegree)/sum(PosDegree+NegDegree))
        if(rand(1,1)<=sum(t4)/sum(t4+t3))
        B(i,j)=1;
        B(j,i)=1;
        Nb1=find(A(:,j)==1);
        Nb2=find(B(:,j)==1);
        for j=1:length(Nb1) %-+-
            if(rand(1,1)<=p2)&&(A(i,Nb1(j))==0)
                B(i,Nb1(j))=1;
                B(Nb1(j),i)=1;
            end
        end
        
        for j=1:length(Nb2) % --+
            if(rand(1,1)<=p1)&&(B(i,Nb2(j))==0)
                A(i,Nb2(j))=1;
                A(Nb2(j),i)=1;
            end
        end
        else
        A(i,j)=1;
        A(j,i)=1;
        Nb1=find(A(:,j)==1);
        Nb2=find(B(:,j)==1);
        for j=1:length(Nb1) %+++
            if(rand(1,1)<=p1)&&(B(i,Nb1(j))==0)
                A(i,Nb1(j))=1;
                A(Nb1(j),i)=1;
            end
        end
        
        for j=1:length(Nb2) %+--
            if(rand(1,1)<=p2)&&(A(i,Nb2(j))==0)
                B(i,Nb2(j))=1;
                B(Nb2(j),i)=1;
            end
        end
        end
        
        
        end
    end
    else
    if(nolink==0)
         d1=sum(A);
        d2=sum(B);
        
    % positive links
%     v1=a11*d1+a12*d2;
%     v1=.5*v1/i;
%     v2=a21*d1+a22*d2;
%     v2=.5*v2/i;
v1=d1/i;
v2=d2/i;
    
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
           Nb1=find(A(:,x)==1);
            Nb2=find(B(:,x)==1);
%             for j=1:i-1
            if(rand(1,1)<0.5)
            for k=1:length(Nb1) %++-
                y=find(A(:,Nb1(k))==1);
                if(~isempty(y))
                
                    y1=randi(length(y),1);
                    for y1=1:length(y)
                        if(rand(1,1)<p2)&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
                            B(x,y(y1))=1;
                            B(y(y1),x)=1;
                            break;
                        end
                    end
                end
            end
            
            for k=1:length(Nb1) %+-+
                y=find(B(:,Nb1(k))==1);
                if(~isempty(y))
                
                    y1=randi(length(y),1);
                    for y1=1:length(y)
                        if(rand(1,1)<p1)&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
                            A(x,y(y1))=1;
                            A(y(y1),x)=1;
                            break;
                        end
                    end
                end
            end
            
            
            else
             for k=1:length(Nb2) %-++
                y=find(A(:,Nb2(k))==1);
                if(~isempty(y))
                    y1=randi(length(y),1);
                    for y1=1:length(y)
                        if(rand(1,1)<p1)&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
                            A(x,y(y1))=1;
                            A(y(y1),x)=1;
                            break;
                        end
                    end
                end
             end
            
             for k=1:length(Nb2) %---
                y=find(B(:,Nb2(k))==1);
                if(~isempty(y))
                    y1=randi(length(y),1);
                    for y1=1:length(y)
                        if(rand(1,1)<p2)&&(A(x,y(y1))==0)&&(B(x,y(y1))==0)
                            B(x,y(y1))=1;
                            B(y(y1),x)=1;
                            break;
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
if(mod(i,1000)==0)
%     
%     [T1,T2,T3,T0]=property(A(1:i,1:i),B(1:i,1:i));
%     BT(1,i/1000)=T1+T3;
%     BT(2,i/1000)=T2;
    
    i;
end
    
end 
 
 AA=A+B;
% BB=A+B;
% A=BB;
% B=AA;
% A=AA;
% 
Degree=sum(AA);

D=sum(AA);

D_DistN=zeros(1,max(D));

    for j=1:n
        for k=1:n	
            if(D(1,k)==j)
                D_DistN(1,j)=D_DistN(1,j)+1;
            end
        end
    end
    D_DistN=D_DistN/sum(D_DistN);

        cumdegree1=zeros((max(Degree)),1);
% cumdegree=zeros(1,n);
        for i=1:max(Degree)
            cumdegree1(i)=sum(D_DistN(i:max(Degree)));
%             cumdegree(i)=sum(D_DistN(1:i));

        end
        cumdegree1=cumdegree1/max(cumdegree1);
        
        
        if(length(cumdegree1)<=length(cumdegree))
      error=sum(abs((1-cumdegree1)-(1-cumdegree(1:length(cumdegree1)))))+sum(cumdegree(1+length(cumdegree1):length(cumdegree)));
        else
      error=sum(abs((1-cumdegree)-(1-cumdegree1(1:length(cumdegree)))))+sum(cumdegree1(1+length(cumdegree):length(cumdegree1)));
        end
        
        
        if (error<error1)
            error1=error;
            cumdegree2=cumdegree1;
            prob=p1;
            Betaa=beta;
            AA1=A;
            BB1=B;
        end
        
                    cumdegree5(1:length(cumdegree2))=cumdegree5(1:length(cumdegree2))+cumdegree2;

        
        
%                 
%     loglog(cumdegree2,' *')
%         hold on;    
%         
%         
%         pause(1)
        
        
        
end
    end
end
        
        
        
   


    

        
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

  
   
  A=AA1-BB1; 
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
% 
% 


%% balancedness


t2=0;
t3=0;
t1=0;
t0=0;
T=zeros(4,n);
result=A;

        for i=1:n	
        x=find(result(i,:));
             for j=1:length(x)
                 y=find(result(x(j),:));
                        for k=1:length(y)	
                            if(y(k)~=i)
                                   if result(i,x(j))==1 && result(i,y(k))==1 && result(x(j),y(k))==1
                                    t3=t3+1;
                                    T(1,i)=T(1,i)+1;
                                   else
                                       if result(i,x(j))==1 && result(i,y(k))==1 && result(x(j),y(k))==-1 || result(i,x(j))==1 && result(i,y(k))==-1 && result(x(j),y(k))==1 || result(i,x(j))==-1 && result(i,y(k))==1 && result(x(j),y(k))==1
                                        t2=t2+1;
                                        T(2,i)=T(2,i)+1;

                                    else
                                        if result(i,x(j))==1 && result(i,y(k))==-1 && result(x(j),y(k))==-1 || result(i,x(j))==-1 && result(i,y(k))==1 && result(x(j),y(k))==-1 || result(i,x(j))==-1 && result(i,y(k))==-1 && result(x(j),y(k))==1
                                            t1=t1+1;
                                            T(3,i)=T(3,i)+1;


                                        else
                                            if result(i,x(j))==-1 && result(i,y(k))==-1 && result(x(j),y(k))==-1
                                                t0=t0+1;
                                                T(4,i)=T(4,i)+1;
                                             end
                                        end
                                    end
                                    end
                            end
                        end
             end

        end
        
     tri1=T(1,:);
     tri2=T(2,:);
     tri3=T(3,:);
     tri4=T(4,:);
        
xx1=xx1+t0/(t0+t2+t3+t1);
xx2=xx2+t3/(t0+t2+t3+t1);
xx3=xx3+t2/(t0+t2+t3+t1);
xx4=xx4+t1/(t0+t2+t3+t1);






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




















        
        B=abs(A);
        degree=sum(B);
        a=0;
        for i=1:n
            a=a+degree(i)*(degree(i)-1)/2;
        end
        
        
        Cs=Cs+((t3+t1)-(t0+t2))/a;
        SG=SG+((t3+t1)-(t0+t2))/((t3+t1)+(t0+t2));
        tempSG=((t3+t1)-(t0+t2))/((t3+t1)+(t0+t2));
        U=U+(1-tempSG)/(1+tempSG);
        Ue=Ue+(1-trace(expm(A))/trace(expm(B)))/(1+trace(expm(A))/trace(expm(B)));
        
       
        D=diag(degree);
        L=D-A;
        
        x=eig(L);
        lambdan=lambdan+min(x);
        
        P=(A+B)/2;
        N=(B-A)/2;
        alpha=2;
        
        x1=max(eig(P+N));
        x2=max(eig(P-N));
        x3=max(x1,x2);
        x4=((eig(alpha*x3*eye(n,n)-(P-N)))./(eig(alpha*x3*eye(n,n)-(P+N))));
        pro=1;
        for i=1:n
            pro=pro*x4(i);
        end
        
        newmanB=newmanB+0.25*log(pro);
        
        














abc







 end    

 
T0=xx1/abc %---
T3=xx2/abc %+++
T2=xx3/abc %-++
T1=xx4/abc %--+
 



xy1=xy1/abc1;
xy2=xy2/abc2;
xy3=xy3/abc3;
xy4=xy4/abc4;
xy5=xy5/abc5;
xy6=xy6/abc6;

Cs=Cs/abc;
SG=SG/abc;
U=U/abc;
Ue=Ue/abc;
lambdan=lambdan/abc;
newmanB=newmanB/abc;
 
 
(figure);
cumdegree=cumdegree/max(cumdegree);
        loglog(cumdegree,'k o')
        
        hold on;
    loglog(cumdegree5/abc,' *')
        hold on;    
        
%         
        pause(1)
