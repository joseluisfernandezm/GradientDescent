function [J,Thetas_finales,T_0_vect,T_1_vect] =  GradientDescent(X,y,alpha,Thj,n_iters)

%{
La salida del algoritmo deberá ser:

- Un vector de la funcion de coste para cada una de las iteraciones del
  algoritmo.
- Una matriz con los parametros thetaj para cada una de las iteraciones

- Matriz X con los datos de entrenamiento.
- Vector y con las etiquetas de los datos de entrenamiento.
- Coeficiente de aprendizaje α.
- Valores de inicialización de los parámetros θj.
- Número máximo de iteraciones a realizar del algoritmo.

%}

    m=size(X,1);%cantidad de datos       
    h = [zeros(m, 1)];%inicializo h a cero
    
    %Extraigo theta por practicidad
    Theta_0=Thj(1);
    Theta_1=Thj(2);
    
    % for para el numero de iteraciones indicado
    
    for i = 1:n_iters 
        
        % Ec DxG, y vamos guardando las nuevas thetas en cada vector, uno
        % para theta0 y el otro para theta1
        
        T_0_vect(i) = Theta_0 - alpha*(1/m)*sum((h-y).*X(:, 1));
        T_1_vect(i) = Theta_1 - alpha*(1/m)*sum((h-y).*X(:, 2)); 
        
        % Actualizar los Theta_j
        Theta_0 = T_0_vect(i);
        Theta_1 = T_1_vect(i);
        T = [Theta_0, Theta_1]; 
        Thetas_finales = T;%al final del bucle seran las theas optimas
        h = (T*X')';
        J(i) = ComputeCost(T, X, y);
        
    end
   
end

