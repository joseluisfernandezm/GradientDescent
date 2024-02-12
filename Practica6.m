clear all
close all
clc 

%{

CONTENIDOS: CARGAR DATOS, PLOT DE DATOS, FUNCION COSTE 1 VAR, DESCENSO
GRASDIENTE, REPRESENTACION CON SUR, REPRESENTACION CON COUNTOUR,
REPRESENTACION 3D

Para casi cualquier problema de machine learning, antes de comenzar con las
tareas de aprendizaje, suele ser de ayuda plotear nuestrso datos para
intentar comprender mejor el problema al que nos enfrentamos. Para ello
haremos un grafico en 2D para visualizar las muestras ya que cada una viene
dada por la poblacion y el beneficio. 

%}

data=load('ex1data1.txt'); %cargamos los datos del fichero con load

% extraemos los datos en 2 vectores x1 y  para hacerlo generico

x1=data(:,1);%poblacion de la ciudad (Datos de entrenamiento)
y=data(:,2);%beneficio de los restaurantes (Etiquetas o Datos de salida)

%plot de los datos

figure()
plot(x1,y,'x');
title('Datos ex1data1.txt')
xlabel('Habitantes (en 10K)')
ylabel('Beneficio (en 10K de euros)')


% Descenso por Graciente

%{

En esta parte vamos a ajustar los parametros theta que como vimos en teoria
son los que ajustan nuestra recta en el modelo de regresion lineal.

Para ello vamos a usar la tecnica de descenso del gradiente que nos va a
permitir minimizar la funcion de coste para poder tener las thetas lo con
un menor error posible. 

J(theta)=1/2m*sumatorio((h(x)-y)^2)

En esta formula m es el numero de datos que desponemos, h es la recta que
tenemos y la Y es la salida de cada una, y en general J es el error medio
que genera nuestra recta de hipotesis con respecto a los datos que
disponemos.

h(x)=theta0+theta1*x que es la ecuacion de la recta. 

El descenso del gradiente nos dice que lo que tenemos que hacer es ir
actualizando los valores de la theta a partir de la derivada de la funcion
de coste hasta converger y el valor de las thetas sea similar en cada
iteracion.

Pasos:

- Matriz X con los datos de entrenamiento
- Vector y por separado con las etiquetas de los datos de entrenamiento
- Coeficiente de aprendizaje alpha
- Valores de inicializacion de los parametros theta en un vector
- Numero max de iteraciones a realizar por el algoritmo, nos lo tienen que
  dar.

La salida del algoritmo deberá ser:

- Un vector de la funcion de coste para cada una de las iteraciones del
  algoritmo.
- Una matriz con los parametros thetaj para cada una de las iteraciones


%}

%Implementamos el algoritmo de desdenso por gradiente

% Primero almacenamos los datos de entrenamiento en una matiz X, cada fila
% va aser un dato y añadiremos una columna de todo unos que representa Xo

% X=[ones(1,size(data,1));x1'];%OJO CON TRASPONER, se puede hacer asi o con la funcio horzcat
% X=X';

X=horzcat(ones(size(data,1),1),x1);% solo añado x1 porque mis datos de entrenamiento son x, los datos de entrada

Thj=[0 0]; %inicializamos los theta a 0

alpha=0.01;  % fijamos un coeficiente de aprendizaje
% alpha=0.0001;  % fijamos un coeficiente de aprendizaje
% alpha=1;  % fijamos un coeficiente de aprendizaje

% Calculo funcion de coste
J = ComputeCost(Thj,X,y);%me devuelve el error medio entre mi recta de hipotesis y la y

n_iters=1500;% numero de iteraciones, nos lo dan

[J,Thetas_finales,T_0_vect,T_1_vect] =  GradientDescent(X,y,alpha,Thj,n_iters);

figure()
plot(J);
title(['alpha =', num2str(alpha), ' Evolucion de J(Theta)'])
xlabel('N iters')
ylabel('Coste J(Theta)')


figure()
plot(T_0_vect);
title(['alpha =', num2str(alpha), ' Evolucion de Theta0'])
xlabel('N iters')
ylabel('Theta0')

figure()
plot(T_1_vect);
title(['alpha =', num2str(alpha), ' Evolucion de Theta1'])
xlabel('N iters')
ylabel('Theta1')




%{
Pregunta: ¿El algoritmo converge?¿cuantas iteraciones necesita aproimadamente?

La convergencia del algoritmo depende del coeficiente de aprendizaje que se
esté utilizando. Cuanto menor sea este convergerá antes y si es mayor
diverge y necesita mas iteraciones para converger.

El otro parametro importante es el numero de iteraciones si esta no es lo
suficientemente grande no convergera. 

CONFIRMAR CON EL PROFE

si amentamos el numero de iteraciones a 1700 empieza a converger para un
alpha de 0,01

--------- Respuesta de ChatGPT ---------

El coeficiente de aprendizaje determina el tamaño del paso que se toma en cada iteración del algoritmo del descenso del gradiente. Si el coeficiente de aprendizaje es demasiado grande, es posible que el algoritmo no converja, ya que los pasos pueden ser demasiado grandes y el algoritmo puede oscilar en lugar de converger.

Por otro lado, si el coeficiente de aprendizaje es demasiado pequeño, el algoritmo puede converger lentamente, ya que los pasos son demasiado pequeños y el algoritmo necesita más iteraciones para encontrar el mínimo global.

Por lo tanto, la elección del coeficiente de aprendizaje adecuado es un equilibrio entre tomar pasos lo suficientemente grandes para converger rápidamente, pero no tan grandes como para hacer que el algoritmo no converja.
%}


%pintamos h en la grafica de los datos

%version extra viendo como varia la recta h
pintarRecta(T_0_vect,T_1_vect,X,y);

%version rapida
h = (Thetas_finales*X')';

figure()
hold on
title('Datos ex1data1.txt y recta h')
plot(x1,y,'x');
plot(x1,h,'r');
xlabel('Habitantes (en 10K)')
ylabel('Beneficio (en 10K de euros)')

%% Visualizacion en 3D

% Vamos a pintar el aspecto de la funcion de coste J(Theta)

% vamos a evaluar la expresion en una rejilla de 10000 puntos 100x100

Theta0vect=linspace(-10,10,100);
Theta1vect=linspace(-1,4,100);

% generamos una matriz Jmat de tamaño tamxtam

tam=100;%me lo dan

Jmat = zeros(tam, tam);

%cada elemento de Jmat corresponde con el valor de la funcion de coste para
%cada uno de los puntos de la rejilla definida por los cectores Theta0vect
%y Theta1vect


for i = 1:tam
    for j = 1:tam
        theta = [Theta0vect(i) Theta1vect(j)];
        Jmat(i, j) = ComputeCost(theta, X, y);
    end

end

figure()
surf(Theta0vect, Theta1vect, Jmat'); %Ojo usar traspuesta
xlabel('Theta 0');
ylabel('Theta 1');
zlabel('J en 3D');
title('Funcion de coste en 3D');

figure()
hold on
contour(Theta0vect, Theta1vect, Jmat', logspace(-2, 3, 20)); %Ojo usar traspuesta
plot(T_0_vect,T_1_vect,'rx')
xlabel('Theta 0');
ylabel('Theta 1');
title('Curvas de nivel de la funcion de coste y camino recorrido por el descenso por gradiente');


