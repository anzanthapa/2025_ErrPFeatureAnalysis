clc;
clear; close all;
%% Load Feature Matrix
featurePath = 'C:\Users\brth229\OneDrive - University of Kentucky\Research Projects\ErRP\results\features';
combined_data = load(fullfile(featurePath,'combinedFeatures.mat')).combined_data;
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
    featureNames = {'ErrP','Shannon Entropy','DWT','PSD','SpectEn','AR'};
    for featurei = 1:6
        resultCellCounter = resultCellCounter+1;
        disp(['<strong>Feature ' num2str(featurei) '/6 has started.</strong>'])
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
    AllSessionsLSVM= cell2mat(cellfun(@(A) A(6,:), resultCell(featureCombIndices,5),'UniformOutput',false));
    resultCell(lastRowIndex+featurei,5) = {[mean(AllSessionsLSVM,1);std(AllSessionsLSVM,0,1)]};
    AllSessionsLDA= cell2mat(cellfun(@(A) A(6,:), resultCell(featureCombIndices,8),'UniformOutput',false));
    resultCell(lastRowIndex+featurei,8) = {[mean(AllSessionsLDA,1);std(AllSessionsLDA,0,1)]};
end
%% Add comments in the resultCell
resultCell(end+1,2) = {'LSVM results is for RBF SVM. Checking if RBF SVM improves performance or not'};
%% Save Result Cell
overallResultPath = fullfile(resultsPath,'overall');
[~,~,~]=mkdir(overallResultPath);
dateAndTime = datetime('now');
currentDate = char(datetime("today"));
resultCellFileName = ['resultCell_' char(datetime("today")) '_' num2str(hour(dateAndTime)) num2str(minute(dateAndTime)) '.mat'];
save(fullfile(overallResultPath,resultCellFileName),"resultCell")
disp('<strong>Results are saved.</strong>')
%% END OF SCRIPT