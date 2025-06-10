clc;
clear; close all;
%% Load Feature Matrix
featurePath = 'C:\Users\brth229\OneDrive - University of Kentucky\Research Projects\ErRP\results\features';
combined_data = load(fullfile(featurePath,'combinedFeaturesDWT.mat')).combined_data;
disp('Features are extracted succeffully.')
resultsPath = 'C:\Users\brth229\OneDrive - University of Kentucky\Research Projects\ErRP\results';
%% Columns for Features and Sessions
subColIndex = 1;
sessColIndex = 2;
classColIndex = 3;
BPFColIndex = 5;
%% Session Determination
subjectNumber = unique(cell2mat(combined_data(2:end,subColIndex)));
sessions = unique(cell2mat(combined_data(2:end,sessColIndex)));
% Use ndgrid to create all combinations
[X, Y] = ndgrid(subjectNumber, sessions);
combinations = [X(:), Y(:)];
%% Result Cell
resultCell(1,1:9) = {'Subject','Session','Feature',...
    'LSVM: Training Metrics','LSVM: Testing Metrics','Best C',...
    'LDA: Training Metrics','LDA Testing Metrics','Best Gamma'};
resultCellCounter = 1;
%% Loop starts
for sessCombi = 1:length(combinations)
    currentSubject = combinations(sessCombi,1);
    currentSession = combinations(sessCombi,2);
    currentCombIndices = find(cell2mat(combined_data(2:end,subColIndex))==currentSubject & cell2mat(combined_data(2:end,sessColIndex))==currentSession)+1;
    disp(['<strong>Subject: ' num2str(currentSubject) ' Session: ' num2str(currentSession) ' .</strong>'])
    %% Class or Output Matrix
    currentY = cell2mat(combined_data(currentCombIndices,classColIndex));
    classRatio = sum(currentY==0)/sum(currentY==1);
    disp(['Class Distribution (Class 1/Class 0): ' num2str(sum(currentY==1)) '/' num2str(sum(currentY==0))])
    %% Start Training and Testing
    %% Features Loop
    featureNames = {'DWT1','DWT2','DWT3','DWT4','DWT5','DWT6','DWT7','DWT8','DWT9'};
    for featurei = 1:length(featureNames)
        resultCellCounter = resultCellCounter+1;
        disp(['<strong>Feature ' num2str(featurei) '/' num2str(length(featureNames)) ': ' featureNames{featurei} ' has started.</strong>'])
        currentX = cell2mat(combined_data(currentCombIndices,BPFColIndex+featurei));
        %% Training and test start
        rng(42);
        numFolds =5;
        trainTestCV = cvpartition(currentY,'KFold',numFolds,'Stratify',true);
        %% Folds metric variables
        foldTrainMetricsLSVM=zeros(numFolds+2,5);
        foldTestMetricsLSVM=zeros(numFolds+2,5);
        foldBestC = zeros(numFolds,1);
        foldTrainMetricsLDA=zeros(numFolds+2,5);
        foldTestMetricsLDA=zeros(numFolds+2,5);
        foldBestGamma = zeros(numFolds,1);
        %% Loop over folds
        for foldi =1:numFolds
            %% Splitting the train and test
            trainIndices = training(trainTestCV,foldi);
            train_input= currentX(trainIndices,:); % Training Data
            train_labels=currentY(trainIndices); %Training Labels

            testIndices = test(trainTestCV,foldi);
            test_input = currentX(testIndices,:);  % Testing data
            test_labels = currentY(testIndices);  % Testing labels

            %% Handling Imabalance in the Class Distribution
            %             [trainingInput_Min,trainingLabel_Min] = ADASYN(train_input,train_labels);
            %             [balancedTrainingInput,balancedTrainingLabels] = Utility_Functions.randomOversample(train_input,train_labels);
            [balancedTrainingInput,balancedTrainingLabels] = Utility_Functions.randomUndersample(train_input,train_labels);
            %             balancedTrainingInput = [train_input;trainingInput_Min];
            %             balancedTrainingLabels = [train_labels;trainingLabel_Min];
            disp('Distribution of class 1/class 0:')
            disp([' Before undersampling: ' num2str(sum(train_labels==1)) '/' num2str(sum(train_labels==0))])
            disp([' After undersampling: ' num2str(sum(balancedTrainingLabels==1)) '/' num2str(sum(balancedTrainingLabels==0))])
            %% Implementation of SVM
            optOPtionsLSVM = struct('AcquisitionFunctionName','expected-improvement-plus',...
                'ShowPlots',0,...
                'Verbose',0,...
                'Kfold',5,'UseParallel',true);
            LSVMModel = fitcsvm(balancedTrainingInput,balancedTrainingLabels,...
                'KernelFunction','linear',...
                'Standardize',true,...
                'OptimizeHyperparameters','BoxConstraint',...
                'HyperparameterOptimizationOptions',optOPtionsLSVM);
            bestC = LSVMModel.BoxConstraints(1);
            foldBestC(foldi,1) = bestC;
            trainingPredictedLabels = predict(LSVMModel,balancedTrainingInput);
            [accTrainLSVM,acc0TrainLSVM,acc1TrainLSVM,F10TrainLSVM,F11TrainLSVM] = Utility_Functions.binaryClassMetrics(trainingPredictedLabels,balancedTrainingLabels);
            foldTrainMetricsLSVM(foldi,:) =  [accTrainLSVM,acc0TrainLSVM,acc1TrainLSVM,F10TrainLSVM,F11TrainLSVM];
            testPredictedLabels = predict(LSVMModel,test_input);
            [accTestLSVM,acc0TestLSVM,acc1TestLSVM,F10TestLSVM,F11TestLSVM] = Utility_Functions.binaryClassMetrics(testPredictedLabels,test_labels);
            foldTestMetricsLSVM(foldi,:) = [accTestLSVM,acc0TestLSVM,acc1TestLSVM,F10TestLSVM,F11TestLSVM];
            disp(['Fold ' num2str(foldi) '/' num2str(numFolds) ' is completed for <strong>LSVM</strong>.'])
            %% Implementation LDA
            optOptionsLDA = struct( ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'ShowPlots',             0, ...
                'Verbose',               0, ...
                'Kfold',                 5, ...
                'UseParallel',           true  ...  % if you have the Parallel Toolbox
                );
            LDAmdl = fitcdiscr( ...
                balancedTrainingInput, ...       % N×d raw features
                balancedTrainingLabels, ...      % N×1 labels (0/1)
                'DiscrimType',               'linear', ...
                'OptimizeHyperparameters',   'Gamma', ...
                'HyperparameterOptimizationOptions', optOptionsLDA ...
                );
            bestGamma = LDAmdl.Gamma;
            foldBestGamma(foldi,1) = bestGamma;
            trainingPredictedLabels = predict(LDAmdl,balancedTrainingInput);
            [accTrainLDA,acc0TrainLDA,acc1TrainLDA,F10TrainLDA,F11TrainLDA] = Utility_Functions.binaryClassMetrics(trainingPredictedLabels,balancedTrainingLabels);
            foldTrainMetricsLDA(foldi,:) = [accTrainLDA,acc0TrainLDA,acc1TrainLDA,F10TrainLDA,F11TrainLDA];
            testPredictedLabels = predict(LDAmdl,test_input);
            [accTestLDA,acc0TestLDA,acc1TestLDA,F10TestLDA,F11TestLDA] = Utility_Functions.binaryClassMetrics(testPredictedLabels,test_labels);
            foldTestMetricsLDA(foldi,:) = [accTestLDA,acc0TestLDA,acc1TestLDA,F10TestLDA,F11TestLDA];
            disp(['Fold ' num2str(foldi) '/' num2str(numFolds) ' is completed for <strong>LDA</strong>.'])
        end
        %% Folds Mean & STD
        foldTrainMetricsLSVM(6,:) = mean(foldTrainMetricsLSVM(1:numFolds,:),1);
        foldTrainMetricsLSVM(7,:) = std(foldTrainMetricsLSVM(1:numFolds,:),0,1);
        foldTestMetricsLSVM(6,:) = mean(foldTestMetricsLSVM(1:numFolds,:),1);
        foldTestMetricsLSVM(7,:) = std(foldTestMetricsLSVM(1:numFolds,:),0,1);

        foldTrainMetricsLDA(6,:) = mean(foldTrainMetricsLDA(1:numFolds,:),1);
        foldTrainMetricsLDA(7,:) = std(foldTrainMetricsLDA(1:numFolds,:),0,1);
        foldTestMetricsLDA(6,:) = mean(foldTestMetricsLDA(1:numFolds,:),1);
        foldTestMetricsLDA(7,:) = std(foldTestMetricsLDA(1:numFolds,:),0,1);
        %% result cell
        resultCell(resultCellCounter,1:3) = {currentSubject,currentSession,featureNames{featurei}};
        resultCell(resultCellCounter,4:6) = {foldTrainMetricsLSVM,foldTestMetricsLSVM,foldBestC};
        resultCell(resultCellCounter,7:9) = {foldTrainMetricsLDA,foldTestMetricsLDA,foldBestGamma};
    end
    clc;
end

%% Taking Averages for All 12 Sessions
lastRowIndex = length(resultCell);
for featurei = 1:length(featureNames)
    featureCombIndices = 1+featurei:length(featureNames):lastRowIndex;
    combinationInfo = resultCell(featureCombIndices,3);
    same = all(cellfun(@isequal, combinationInfo, repmat(combinationInfo(1,:), size(combinationInfo,1), 1)), 2);
    if ~any(same)
        error('Not the same combinations')
    end
    resultCell(lastRowIndex+featurei,2:3) = {'Average',resultCell(1+featurei,3)};
    %     currentbAccLSVM=cell2mat(cellfun(@(A) mean(A(1:5,2:3),2), resultCell(featureCombIndices,5),'UniformOutput',false));
    %     currentbAccReshaped = mean(reshape(currentbAccLSVM,5,[])',2);
    %     [stdBAccLSVM,meanBAccLSVM] = std(currentbAccReshaped,0,1);
    %     resultCell(lastRowIndex+featurei,10)={[meanBAccLSVM;stdBAccLSVM]};
    %
    %     currentbAccLDA=cell2mat(cellfun(@(A) mean(A(1:5,2:3),2), resultCell(featureCombIndices,8),'UniformOutput',false));
    %     currentbAccReshaped = mean(reshape(currentbAccLDA,5,[])',2);
    %     [stdBAccLDA,meanBAccLDA] = std(currentbAccReshaped,0,1);
    %     resultCell(lastRowIndex+featurei,11)={[meanBAccLDA;stdBAccLDA]};


    AllSessionsLSVM= cell2mat(cellfun(@(A) A(6,:), resultCell(featureCombIndices,5),'UniformOutput',false));
    resultCell(lastRowIndex+featurei,11) ={AllSessionsLSVM};
    resultCell(lastRowIndex+featurei,5) = {[mean(AllSessionsLSVM,1);std(AllSessionsLSVM,0,1)]};
    AllSessionsLDA= cell2mat(cellfun(@(A) A(6,:), resultCell(featureCombIndices,8),'UniformOutput',false));
    resultCell(lastRowIndex+featurei,12) ={AllSessionsLDA};
    resultCell(lastRowIndex+featurei,8) = {[mean(AllSessionsLDA,1);std(AllSessionsLDA,0,1)]};
    [~,pLDALSVM] = ttest(AllSessionsLDA,AllSessionsLSVM,'Dim',1);
    resultCell(lastRowIndex+featurei,10) ={pLDALSVM};
end
%% Add comments in the resultCell
resultCell(end+1,2) = {'LSVM results is for RBF SVM. Checking if RBF SVM improves performance or not'};

%% Statistical Evaluation
%Feature Vs Feature for BACC
ErrPLSVM = resultCell{62,11}(:,1); 
ShEnLSVM = resultCell{63,11}(:,1);
DWTLSVM = resultCell{64,11}(:,1);
PSDLSVM = resultCell{65,11}(:,1);
SpectEnLSVM = resultCell{66,11}(:,1);
[~,p12]=ttest(ErrPLSVM,ShEnLSVM);[~,p13]=ttest(ErrPLSVM,DWTLSVM);[~,p14]=ttest(ErrPLSVM,PSDLSVM);[~,p15]=ttest(ErrPLSVM,SpectEnLSVM);
[~,p23]=ttest(ShEnLSVM,DWTLSVM);[~,p24]=ttest(ShEnLSVM,PSDLSVM);[~,p25]=ttest(ShEnLSVM,SpectEnLSVM);
[~,p34]=ttest(DWTLSVM,PSDLSVM);[~,p35]=ttest(DWTLSVM,SpectEnLSVM);
[~,p45]=ttest(PSDLSVM,SpectEnLSVM);
FFLSVM = [nan,p12,p13,p14,p15
    p12,nan,p23,p24,p25
    p13,p23,nan,p34,p35
    p14,p24,p34,nan,p45
    p15,p25,p35,p45,nan];

ErrPLDA = resultCell{62,12}(:,1); 
ShEnLDA = resultCell{63,12}(:,1);
DWTLDA = resultCell{64,12}(:,1);
PSDLDA = resultCell{65,12}(:,1);
SpectEnLDA = resultCell{66,12}(:,1);
[~,p12]=ttest(ErrPLDA,ShEnLDA);[~,p13]=ttest(ErrPLDA,DWTLDA);[~,p14]=ttest(ErrPLDA,PSDLDA);[~,p15]=ttest(ErrPLDA,SpectEnLDA);
[~,p23]=ttest(ShEnLDA,DWTLDA);[~,p24]=ttest(ShEnLDA,PSDLDA);[~,p25]=ttest(ShEnLDA,SpectEnLDA);
[~,p34]=ttest(DWTLDA,PSDLDA);[~,p35]=ttest(DWTLDA,SpectEnLDA);
[~,p45]=ttest(PSDLDA,SpectEnLDA);
FFLDA = [nan,p12,p13,p14,p15
    p12,nan,p23,p24,p25
    p13,p23,nan,p34,p35
    p14,p24,p34,nan,p45
    p15,p25,p35,p45,nan];

%% Save Result Cell
overallResultPath = fullfile(resultsPath,'overall');
[~,~,~]=mkdir(overallResultPath);
dateAndTime = datetime('now');
currentDate = char(datetime("today"));
resultCellFileName = ['resultCell_DWT_' char(datetime("today")) '_' num2str(hour(dateAndTime)) num2str(minute(dateAndTime)) '.mat'];
save(fullfile(overallResultPath,resultCellFileName),"resultCell")
disp('<strong>Results are saved.</strong>')
%% END OF SCRIPT