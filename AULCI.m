function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,rank_AULCI]...
=AULCI(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,nayai)
      rng(0)

REGRESSION=0;
CLASSIFIER=1;

T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   

TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
       
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                                 

start_time_train=cputime;
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
tempH=InputWeight*P;

x = abs(InputWeight);
pa =nayai; 
                                          %   Release input of training data 
CIFcn1 = @(x,pa)std(x(:),'omitnan')/sqrt(sum(~isnan(x(:)))) * ...
tinv(abs([0,1]-(1-pa/100)/2),sum(~isnan(x(:)))-1) + mean(x(:),'omitnan'); 
CI= CIFcn1(x,pa);


 if mod(NumberofHiddenNeurons,2)==0
     up = NumberofHiddenNeurons/2;
    down=NumberofHiddenNeurons/2; 
 else 
     up =(int16(NumberofHiddenNeurons/2))-1;
     down=(int16(NumberofHiddenNeurons/2));
%    
 end
 
       Biasup = CI(2)+(CI(2)-1).* rand(down,1); 
      Biasdown = CI(1)+(CI(1)-0).* rand(up,1); 
 
              Bias=[Biasdown;Biasup];
Bias = 1 ./ (1 + exp(-Bias));

 ind=ones(1,NumberofTrainingData);
 Bias=Bias(:,ind); 
 
clear x;
tempH=tempH+(Bias);
[U1,S1,V1] = svd(P);
s21 = diag(S1);
 rank_AULCI = nnz(s21); 

 clear P;
 % Calculate the rank using the number of nonzero singular values
switch lower(ActivationFunction)
    case {10,1,'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case {20,2,'sin'}
         H = sin(tempH);    
         case {30,3,'hardlim'}
              H = double(hardlim(tempH)); 
             
case {40,'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {50,'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
end
clear tempH;                                        

OutputWeight=pinv(H') * T';  
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train    ;    %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))  ;  
%   Calculate training accuracy (RMSE) for regression case
end
clear H;
%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
%clear TV.P;             %   Release input of testing data             
 ind=ones(1,NumberofTestingData);
 Bias=Bias(:,ind); 
tempH_test=tempH_test +Bias;
switch lower(ActivationFunction)
    case {10,1,'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {20,2,'sin'}
        H_test = sin(tempH_test);    
        case {30,3,'hardlim'}
                H_test = double(hardlim(tempH_test));  
                case {40,'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {50,'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
                                               
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test  ;         %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY))   ;         %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [~, label_index_expected]=max(T(:,i));
        [~, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
%              xlswrite('TrainingAccuracy.xls',TrainingAccuracy)
    for i = 1 : size(TV.T, 2)
        [~, label_index_expected]=max(TV.T(:,i));
        [~, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
        
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);  
%         xlswrite('TrainingAccuracy.xls',TestingAccuracy)

end