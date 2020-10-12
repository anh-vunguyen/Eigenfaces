%% Auteur : NGUYEN Anh Vu
%% Instructeur :  Oscar ACOSTA
%% FACE RECOGNITION WITH THE METHOD OF EIGENFACES

close all
clc

%% Load the Database of Images
load('ImDatabase.mat');

%% Useful coefficients

% Number of class 
NoC = 30;

% Number of images per class
NoI = 15;

% Dimension of each image
dimImg = 32256;

% Number of largest eigen vectors
nb = 50;

%% Mean (Average) Face
M = mean(A,2);
M1 = reshape(M,[192,168]);
figure('name' , 'Mean Face')
imshow(M1);


%% Matrix of Covariance
x = bsxfun(@minus,A,M); %Normailized Vector
s = x'*x;

%% Eigen vectors and Eigen values
[vectorC, valueC] = eig(s);
ss = diag(valueC);
[ss,iii] = sort(-ss);
vectorC = vectorC(:,iii);

% Eigenfaces
vectorL = [];
for i=1:1:nb
    vectorL(:,i) = x*vectorC(:,i);
end

% Show Eigenfaces
figure;
eigenface=[];
for i=1:1:25
    eigenface= reshape(vectorL(:,i),192,168);
    subplot(5,5,i);
    imshow(eigenface, []);
    title(['Eigen Face #', num2str(i)]);
end

% Calculate Coefficients
Coeff = [];
for i=1:1:NoI*NoC
    Coeff(:,i) = x(:,i)'*vectorL;
end
%% Face recognition test

% Input Face
input = 'yaleB01_P00A+000E+00.pgm';
inputImg_8 = imread(input);
inputImg_db = im2double(inputImg_8);
figure ('name', 'Test Image')
imshow(inputImg_db);
inputImg_db  = reshape(inputImg_db,[192*168,1]);
inputImg_db  = bsxfun(@minus,inputImg_db,M);

% Calculate the coefficients of Input Face
inputCoeff = (inputImg_db'*vectorL);
inputCoeff = inputCoeff';

% Reconstruction
weight = inputCoeff/(NoI*NoC*50);
Re = M;
for j=1:1:nb
   Re = Re + vectorL(:,j)*weight(j,1);
end
Re = reshape(Re,[192,168]);
figure('name','Reconstructed Face');
imshow(Re);

% Result
comp_Coeff = zeros(NoC*NoI,1);
tmp_res =0;
for k=1:1:NoI*NoC
    for t=1:1:nb
       tmp_res = tmp_res + (inputCoeff(t,1)-Coeff(t,k))*(inputCoeff(t,1)-Coeff(t,k)); 
    end
    comp_Coeff(k,1) = tmp_res;
    tmp_res = 0;
end

[MinCoeff,I] = min(comp_Coeff);
comp_Coeff_sorted = sort(comp_Coeff);
MaxCoeff = max(comp_Coeff_sorted(1:NoI*2));
if (MinCoeff < MaxCoeff*0.8)
    msgbox('Recognized in the Database')
else
    msgbox('Unrecognized in the Database')
end

% Show the appropriate image in the Database
Org = reshape(A(:,I),[192,168]);
figure('name','Database Image');
imshow(Org);



