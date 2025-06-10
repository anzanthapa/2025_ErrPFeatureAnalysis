clear; clc; close all; % clear workspace; clear command window;closes all figures
%% Directories
data_path = 'C:\Users\brth229\OneDrive - University of Kentucky\Research Projects\ErRP\data';
resultsPath = 'C:\Users\brth229\OneDrive - University of Kentucky\Research Projects\ErRP\results';
%% Finding All 12.mat files from data path
data_filenames = {dir(fullfile(data_path, '*.mat')).name};
%% Epoch Parameters
% extraction window interval (in seconds) for EEG will be relative to the onset of an event
window_start = 0; % in seconds relative to the event onset
window_size = 0.5; % in seconds
fs = 512;
selected_channels = {'FC1','C1','CPz','FC2','FCz','Cz','C2',}; % Now accepts a cell array of channel names
% selected_channels = {'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', ...
%                  'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', ...
%                  'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', ...
%                  'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', ...
%                  'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', ...
%                  'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', ...
%                  'P10', 'PO8', 'PO4', 'O2'};

%% Epoch Extraction
% Create cell array to combine all relevant data. Useful for doing both subject wise or session wise classification
% Column 1: Subject; Column 2: Session; Column 3: Class Label; Column 4: Preprocessed EEG Data; Column 5: BPF EEG Data;  
combined_data = cell(1,11);
subColIndex = 1;
sessColIndex = 2;
classColIndex = 3;
HPFColIndex = 4;
BPFColIndex = 5;
combined_data(1,:) = {'Subject','Session','Class','HPF EEG','BPF EEG','DWT1','DWT2','DWT3','DWT4','DWT5','DWT6','DWT7','DWT8','DWT9'};
%% Files Loop
sample_counter = 1;
for filei=1:length(data_filenames) 
    disp(['<strong>File #' num2str(filei) ' has started.</strong>'])
    current_data = load(fullfile(data_path,data_filenames{filei})).run;
    for runi = 1:length(current_data)
        disp([' Run ' num2str(runi) '/' num2str(length(current_data))])
        current_run_eeg_raw = current_data{1,runi}.eeg;
        current_run_fs = current_data{1,runi}.header.SampleRate;
        [current_run_HPF_eeg,current_run_BPF_eeg] = Utility_Functions.preprocess_eeg(current_run_eeg_raw,current_run_fs);
        current_run_channels = current_data{1,runi}.header.Label;
        selected_channel_indexes = find(cellfun(@(x) any(strcmp(x, selected_channels)), current_run_channels));
        
        if length(selected_channel_indexes)~=length(selected_channels)
            error('Channel selection is not correct');
        end
        %% Extracting Class 1: Error Events
        type6_indices = current_data{1,runi}.header.EVENT.TYP==6;
        type9_indices = current_data{1,runi}.header.EVENT.TYP==9;
        current_error_event_pos = current_data{1,runi}.header.EVENT.POS(type6_indices | type9_indices);
        for eventi = 1:length(current_error_event_pos)
            start_sample = current_error_event_pos(eventi)+window_start*current_run_fs;
            end_sample = start_sample+(window_size*current_run_fs)-1;
            sample_counter=sample_counter+1;
            combined_data{sample_counter,subColIndex} = current_data{1,runi}.header.Subject;
            combined_data{sample_counter,sessColIndex} = current_data{1,runi}.header.Session;
            combined_data{sample_counter,classColIndex} = 1;
            combined_data{sample_counter,HPFColIndex} = current_run_HPF_eeg(start_sample:end_sample,selected_channel_indexes)';
            combined_data{sample_counter,BPFColIndex} = current_run_BPF_eeg(start_sample:end_sample,selected_channel_indexes)';

        end

        %% Extracting Class 0: Correct Events
        type5_indices = current_data{1,runi}.header.EVENT.TYP==5;
        type10_indices = current_data{1,runi}.header.EVENT.TYP==10;
        current_correct_event_pos = current_data{1,runi}.header.EVENT.POS(type5_indices | type10_indices);
        for eventi = 1:length(current_correct_event_pos)
            start_sample = current_correct_event_pos(eventi)+window_start*current_run_fs;
            end_sample = start_sample+(window_size*current_run_fs)-1;
            sample_counter=sample_counter+1;
            combined_data{sample_counter,subColIndex} = current_data{1,runi}.header.Subject;
            combined_data{sample_counter,sessColIndex} = current_data{1,runi}.header.Session;
            combined_data{sample_counter,classColIndex} = 0;
            combined_data{sample_counter,HPFColIndex} = current_run_HPF_eeg(start_sample:end_sample,selected_channel_indexes)';
            combined_data{sample_counter,BPFColIndex} = current_run_BPF_eeg(start_sample:end_sample,selected_channel_indexes)';
        end
        disp([' Total Correct Correct/Error Events: ' num2str(length(current_correct_event_pos)) '/' num2str(length(current_error_event_pos))])
    end
end
%% Plotting ErrP/Correct
ErrPFolder = fullfile(resultsPath,'ErrP');
[~,~,~] = mkdir(ErrPFolder);
subjectNums = unique(cell2mat(combined_data(2:end,subColIndex)));
for subi = 1:length(subjectNums)
    currentSubject = subjectNums(subi);
    for sessi=1:2
        currentSessionCorrectIndices = find(cell2mat(combined_data(2:end,subColIndex))==currentSubject & cell2mat(combined_data(2:end,sessColIndex))==sessi & cell2mat(combined_data(2:end,classColIndex))==0)+1;
        currentSessionErrPIndices = find(cell2mat(combined_data(2:end,subColIndex))==currentSubject & cell2mat(combined_data(2:end,sessColIndex))==sessi & cell2mat(combined_data(2:end,classColIndex))==1)+1;
        correctSamplesHPF = permute(cat(3,combined_data{currentSessionCorrectIndices,HPFColIndex}),[3,1,2]);
        ErrPSamplesHPF = permute(cat(3,combined_data{currentSessionErrPIndices,HPFColIndex}),[3,1,2]);
        correctSamplesBPF = permute(cat(3,combined_data{currentSessionCorrectIndices,BPFColIndex}),[3,1,2]);
        ErrPSamplesBPF = permute(cat(3,combined_data{currentSessionErrPIndices,BPFColIndex}),[3,1,2]);
        for chi=1:length(selected_channels)
            % HPF Filtered
            correctSamplesPerChannel = squeeze(correctSamplesHPF(:,chi,:));
            ErrPSamplesPerChannel = squeeze(ErrPSamplesHPF(:,chi,:));
            [currentStdErrP,currentMeanErrP] = std(ErrPSamplesPerChannel,0,1);
            [currentStdCorrect,currentMeanCorrect] = std(correctSamplesPerChannel,0,1);
            HPFfig = figure('Visible','off');
            time_axis = (0:window_size*fs-1)/fs;
            hold on;
            plot(time_axis,currentMeanErrP,'LineWidth',2,'Color','red','DisplayName','Error')
            plot(time_axis,currentMeanErrP-currentStdErrP,'LineWidth',2,'Color','red','LineStyle','--')
            plot(time_axis,currentMeanErrP+currentStdErrP,'LineWidth',2,'Color','red','LineStyle','--')
            plot(time_axis,currentMeanCorrect,'LineWidth',2,'Color','green','DisplayName','Correct')
            plot(time_axis,currentMeanCorrect-currentStdCorrect,'LineWidth',2,'Color','green','LineStyle','--')
            plot(time_axis,currentMeanCorrect+currentStdCorrect,'LineWidth',2,'Color','green','LineStyle','--')
            plot(time_axis,currentMeanErrP-currentMeanCorrect,'LineWidth',2,'Color','black','DisplayName','Error-Correct')
            hold off;
            legend()
            xlabel('Time relative to the event onset [s]');
            ylabel('Amplitude [\muV]');ylim([-20,20]);
            title(['Channel Name:' selected_channels{chi}])
            fileName = ['Fig_ErrPCorrect_CAR_HPF_BSF_Sub' num2str(subi) '_Session' num2str(sessi) '_' selected_channels{chi}];
            saveas(gcf, fullfile(ErrPFolder, [fileName '.fig']));
            print(gcf, fullfile(ErrPFolder, [fileName '.tif']), '-dtiff', '-r300');

            % Band Pass Filtered
            correctSamplesPerChannel = squeeze(correctSamplesBPF(:,chi,:));
            ErrPSamplesPerChannel = squeeze(ErrPSamplesBPF(:,chi,:));
            [currentStdErrP,currentMeanErrP] = std(ErrPSamplesPerChannel,0,1);
            [currentStdCorrect,currentMeanCorrect] = std(correctSamplesPerChannel,0,1);
            BPFfig = figure('Visible','off');
            time_axis = (0:window_size*fs-1)/fs;
            hold on;
            plot(time_axis,currentMeanErrP,'LineWidth',2,'Color','red','DisplayName','Error')
            plot(time_axis,currentMeanErrP-currentStdErrP,'LineWidth',2,'Color','red','LineStyle','--')
            plot(time_axis,currentMeanErrP+currentStdErrP,'LineWidth',2,'Color','red','LineStyle','--')
            plot(time_axis,currentMeanCorrect,'LineWidth',2,'Color','green','DisplayName','Correct')
            plot(time_axis,currentMeanCorrect-currentStdCorrect,'LineWidth',2,'Color','green','LineStyle','--')
            plot(time_axis,currentMeanCorrect+currentStdCorrect,'LineWidth',2,'Color','green','LineStyle','--')
            plot(time_axis,currentMeanErrP-currentMeanCorrect,'LineWidth',2,'Color','black','DisplayName','Error-Correct')
            hold off;
            legend()
            xlabel('Time relative to the event onset [s]');
            ylabel('Amplitude [\muV]');ylim([-20,20]);
            title(['Channel Name:' selected_channels{chi}])
            fileName = ['Fig_ErrPCorrect_CAR_BPF_Sub' num2str(subi) '_Session' num2str(sessi) '_' selected_channels{chi}];
            saveas(gcf, fullfile(ErrPFolder, [fileName '.fig']));
            print(gcf, fullfile(ErrPFolder, [fileName '.tif']), '-dtiff', '-r300');
        end
        disp(['Figure is saved for Subject ' num2str(subi) ' Session ' num2str(sessi) '.'])
    end
end

%% Load existing combinedFeatures . mat
featurePath = fullfile(resultsPath,'features');
if 0
    combined_data = load(fullfile(featurePath,'combinedFeatures.mat')).combined_data;
end
%% Extracting Features for Each Sample
for triali = 2:size(combined_data,1)
    currentTrialEEGHPF = combined_data{triali,HPFColIndex};
    [dwt1,dwt2,dwt3,dwt4,dwt5,dwt6,dwt7,dwt8,dwt9]=Utility_Functions.calculate_DWT_cases(currentTrialEEGHPF);
    % Assign all features in one line
    [combined_data{triali,BPFColIndex+1:BPFColIndex+9}] = deal(dwt1,dwt2,dwt3,dwt4,dwt5,dwt6,dwt7,dwt8,dwt9);
    disp(['Trial ' num2str(triali-1) '/' num2str(size(combined_data,1)-1) ' is completed.'])
end

%% Saving the Combined File with features
fileName = 'combinedFeaturesDWT.mat';
[~,~,~]=mkdir(featurePath);
save(fullfile(featurePath,fileName),"combined_data");
disp('<strong>Features are saved.</strong>')
%% END OF SCRIPT
