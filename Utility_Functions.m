classdef Utility_Functions
    %UTILITY_FUNCTIONS Summary of this class goes here
    %   This class will have the functions used for ErrP analysis.
    methods(Static)

        %% HPF
        function [BSF50_signal,BPF_signal] = preprocess_eeg(raw_eeg,fs)
            HPF_signal = zeros(size(raw_eeg));
            BSF50_signal = zeros(size(raw_eeg));
            BPF_signal = zeros(size(raw_eeg));
            CAREEG = raw_eeg - mean(raw_eeg,2);
            for chi = 1:size(raw_eeg,2)
                [HPF_signal(:,chi),~] = highpass(CAREEG(:,chi)',0.1,fs,"ImpulseResponse","iir",'StopbandAttenuation',60);
                [BSF50_signal(:,chi),~] = bandstop(HPF_signal(:,chi)',[49.9,50.1],fs,"ImpulseResponse","iir",'StopbandAttenuation',60);
                [BPF_signal(:,chi),~] = bandpass(CAREEG(:,chi)',[1,10],fs,"ImpulseResponse","iir",'StopbandAttenuation',60);
            end
        end

        %% DWT
        function [dwt_feature]=calculate_DWT(inputEEG)
            waveletname='db4';
            decomplevel=5;
            dwt_feature=[];
            for chi=1:size(inputEEG,1)
                wpd_trees=wpdec(inputEEG(chi,:),decomplevel,waveletname);
                dwt_feature_array1=wpcoef(wpd_trees,[decomplevel,0]); % 0-8
                dwt_feature_array2=wpcoef(wpd_trees,[decomplevel,1]); % 8-16
                %                 dwt_feature_array3=wpcoef(wpd_trees,[decomplevel,2]); % 16-24
                %                 dwt_feature_array4=wpcoef(wpd_trees,[decomplevel,3]); % 24-32
                %                 dwt_feature_array5=wpcoef(wpd_trees,[decomplevel,4]); % 32-40
                %                 dwt_feature_array6=wpcoef(wpd_trees,[decomplevel,5]); % 40-48
                dwt_feature=[dwt_feature,dwt_feature_array1,dwt_feature_array2];
            end
        end

        function [dwt1,dwt2,dwt3,dwt4,dwt5,dwt6,dwt7,dwt8,dwt9]=calculate_DWT_cases(inputEEG)
            waveletname='db4';
            dwt1=[];dwt2=[];dwt3=[];dwt4=[];dwt5=[];dwt6=[];dwt7=[];dwt8=[];dwt9=[];

            for chi=1:size(inputEEG,1)
                % Case 1
                wpd_trees=wpdec(inputEEG(chi,:),5,waveletname);
                a1=wpcoef(wpd_trees,[1,0]);
                d1=wpcoef(wpd_trees,[1,1]);
                dwt1=[dwt1,a1,d1];
                % Case 2
                a2 = wpcoef(wpd_trees,[2,0]);
                d2 = wpcoef(wpd_trees,[2,1]);
                dwt2=[dwt2,d1,a2,d2];
                % Case 3
                dwt3=[dwt3,a2,d2];
                % Case 4
                a3=wpcoef(wpd_trees,[3,0]);
                d3= wpcoef(wpd_trees,[3,1]);
                dwt4=[dwt4,a3,d3];
                % Case 5
                a4=wpcoef(wpd_trees,[4,0]);
                d4=wpcoef(wpd_trees,[4,1]);
                dwt5=[dwt5,d1,d2,d3,a4];
                % Case 6 & 7
                dwt6=[dwt6,d1,d2,d3,d4,a4];
                dwt7=[dwt7,a4,d4];
                % Case 8
                a5=wpcoef(wpd_trees,[5,0]);
                d5=wpcoef(wpd_trees,[5,1]);
                dwt8=[dwt8,d1,d2,d3,d4,d5,a5];
                % Case 9
                dwt9=[dwt9,a5,d5];
            end
        end
        %% PSD
        function psd_feature=calculate_PSD(inputEEG,fs)
            L=size(inputEEG,2);
            psd_feature_array=zeros(size(inputEEG,1),L/2);
            for chi=1:size(inputEEG,1)
                fftsig=fft(inputEEG(chi,:));
                psd=(1/(fs*L))*abs(fftsig).^2;
                psd_onesided = psd(1:L/2+1);
                psd_onesided(2:end-1)=2*psd_onesided(2:end-1);
                psd_feature_array(chi,:)=psd_onesided(2:end);
            end
            psd_feature = reshape(psd_feature_array',1,[]);
        end
        function psd_feature=calculate_PSD_DTA(inputEEG,fs)
            L=512;
            psd_feature_array=zeros(size(inputEEG,1),14);
            f_oneSided = (1:L/2)*(fs/L); %Excluding DC
            for chi=1:size(inputEEG,1)
                fftsig=fft(inputEEG(chi,:),L);
                psd=(1/(fs*L))*abs(fftsig).^2;
                psd_onesided = psd(1:L/2+1);
                psd_onesided(2:end-1)=2*psd_onesided(2:end-1);
                PSDDTA = psd_onesided(2:end);
                psd_feature_array(chi,:)=PSDDTA(f_oneSided>0 & f_oneSided<=14);
            end
            psd_feature = reshape(psd_feature_array',1,[]);
        end
        %% Spectral Entropy
        function [spectEn] = func_compute_SpectEn(inputEEG,fs)
            % Compute Spectral Entropy for each row (channel) of the EEG
            % Segment
            % inputEEG: Input EEG signal (size: numChannels x EEG samples)
            % spectEn: Output spectral entropy (size: 1 x numChannels)
            L=size(inputEEG,2);
            PSD=zeros(size(inputEEG,1),L/2);
            for chi=1:size(inputEEG,1)
                fftsig=fft(inputEEG(chi,:));
                psd=(1/(fs*L))*abs(fftsig).^2;
                psd_onesided = psd(1:L/2+1);
                psd_onesided(2:end-1)=2*psd_onesided(2:end-1);
                PSD(chi,:)=psd_onesided(2:end);
            end
            % Ensure PSD values are positive to avoid log issues
            PSD(PSD == 0) = eps; % Replace zero values with machine epsilon
            % Compute total power per row (sum over frequency bins)
            totalPSD = sum(PSD, 2); % Sum along columns (across frequencies)

            % Compute normalized power (probability distribution)
            PWi = PSD./ totalPSD;

            % Compute Spectral Entropy
            spectEn = -sum(PWi .* log(PWi), 2)'; % Sum across frequency dimension using natural log (ln)

        end
        %% Shannon's Entropy
        function entropy_feature = calculate_Entropy(inputEEG,fs)
            [b, a] = butter(4, [4 8]./(fs/2), 'bandpass');  % Design filter once
            num_channels = size(inputEEG, 1);
            entropy_feature_array = zeros(1,num_channels);  % Preallocate
            for chi = 1:num_channels
                % Filter to extract theta band
                theta_band = filtfilt(b, a, inputEEG(chi, :));
                %                 currentChannelEEG = inputEEG(chi,:);
                % Use default number of bins if not specified (Used Sturges
                % rule)
                n_bins = round(1 + log2(length(theta_band)));
                % Histogram to estimate probabilities
                [counts, ~] = histcounts(theta_band, n_bins);
                probabilities = counts./ sum(counts);

                % Remove zero entries
                probabilities = probabilities(probabilities > 0);

                % Shannon entropy
                entropy_feature_array(1,chi) = -sum(probabilities .* log2(probabilities));
            end

            entropy_feature = entropy_feature_array;
        end
        %% Random Undersampling
        function [X_bal, Y_bal] = randomUndersample(trainingX, trainingY)
            % undersample01  Randomly undersample majority class (0) to match minority (1)
            %
            %   [X_bal, Y_bal] = undersample01(trainingX, trainingY)
            %
            %   Inputs:
            %     - trainingX : N×d feature matrix
            %     - trainingY : N×1 label vector with values 0 or 1 (class 1 is minority)
            %
            %   Outputs:
            %     - X_bal : M×d feature matrix after undersampling (M = 2·#(class 1))
            %     - Y_bal : M×1 label vector after undersampling
            %
            %   This keeps all class 1 samples and randomly selects the same
            %   number of class 0 samples (without replacement).

            % Find indices of each class
            idx0 = find(trainingY == 0);  % majority
            idx1 = find(trainingY == 1);  % minority
            n1   = numel(idx1);

            % If no minority or already balanced, return unchanged
            if n1 == 0 || numel(idx0) == n1
                X_bal = trainingX;
                Y_bal = trainingY;
                return;
            end
            rng(42)
            % Randomly pick n1 samples from class 0
            keep0 = randsample(idx0, n1, false);

            % Combine with all class 1 indices
            finalIdx = [keep0; idx1];

            % Extract balanced set and shuffle
            X_bal = trainingX(finalIdx, :);
            Y_bal = trainingY(finalIdx);

            perm = randperm(size(X_bal, 1));
            X_bal = X_bal(perm, :);
            Y_bal = Y_bal(perm);
        end



        %% Binary Class Metrics
        function [baccuracy, acc_class0, acc_class1, F1_class0, F1_class1] = binaryClassMetrics(predictions, trueLabels)

            if nargin ==0
                y_true = [ones(1,67),zeros(1,13),ones(1,33),zeros(1,12)];
                y_pred = [ones(1,67),zeros(1,13),zeros(1,33),ones(1,12)];

                fprintf('Running internal test for binaryClassMetrics…\n');
                [acc, acc0, acc1, F10, F11] = Utility_Functions.binaryClassMetrics(y_pred, y_true);
                disp([acc, acc0, acc1, F10, F11])
                confusionmat(y_true,y_pred)
                return;
            end

            % Ensure column vectors
            predictions = predictions(:);
            trueLabels = trueLabels(:);

            % Compute overall accuracy
            accuracy = sum(predictions == trueLabels) / numel(trueLabels);

            % Class 0
            TP0 = sum(predictions == 0 & trueLabels == 0);
            FP0 = sum(predictions == 0 & trueLabels == 1);
            FN0 = sum(predictions == 1 & trueLabels == 0);

            precision0 = TP0 / (TP0 + FP0 + eps);
            recall0 = TP0 / (TP0 + FN0 + eps);
            F1_class0 = 2 * (precision0 * recall0) / (precision0 + recall0 + eps);

            % Class 1
            TP1 = sum(predictions == 1 & trueLabels == 1);
            FP1 = sum(predictions == 1 & trueLabels == 0);
            FN1 = sum(predictions == 0 & trueLabels == 1);

            precision1 = TP1 / (TP1 + FP1 + eps);
            recall1 = TP1 / (TP1 + FN1 + eps);
            F1_class1 = 2 * (precision1 * recall1) / (precision1 + recall1 + eps);

            % Compute class specific accuracies
            N0 = sum(trueLabels==0);
            acc_class0 = TP0/(N0+eps);
            N1 = sum(trueLabels==1);
            acc_class1 = TP1/(N1+eps);
            baccuracy = (acc_class0+acc_class1)/2;
        end

    end
end
%% END OF SCRIPT

