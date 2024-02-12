function pintarRecta(T_0_vect,T_1_vect,X,y)


    Thj=[T_0_vect;T_1_vect];
    Thj=Thj';
    h = (Thj*X')';
    m=size(X,1);%cantidad de datos     
    
    x1=X(:,2);


    figure(100)
    hold on
    plot(x1,y,'x');
    xlabel('Habitantes (en 10K)')
    ylabel('Beneficio (en 10K de euros)')
    
    aux=plot(x1,h(:,1),'r');
    
    for i=1:size(h,2)
        
        ref=figure(100)
        delete(aux)
        aux=plot(x1,h(:,i),'r');
        title(['Datos ex1data1.txt y recta h(',num2str(i),')'])

        
        pause(0.000001)
        
        
    end
    hold off

end

